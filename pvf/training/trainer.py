"""TrainingRecipe (everything except the architecture and the data) and the trainer.

The trainer knows models only through the contract in models/base.py:
    model(forecast_batch) -> {"forecast", optional "auxiliary_loss"}
    model.is_trainable, model.quantile_levels, optional model.on_epoch_start(epoch, n_epochs)
"""
from __future__ import annotations

import copy
import itertools
import os
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from ..data.pipeline import make_data_loader
from .losses import get_loss
from .optimizers import make_optimizer, make_scheduler


@dataclass
class TrainingRecipe:
    loss: str = "mse"                        # mse | mae | huber | pinball (quantile models always use pinball)
    optimizer: str = "adam"                  # adam | adamw | sgd | rmsprop
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer_options: dict = field(default_factory=dict)
    scheduler: str = "none"                  # none | cosine | step | reduce_on_plateau | one_cycle
    scheduler_options: dict = field(default_factory=dict)
    n_epochs: int = 100
    batch_size: int = 64
    early_stopping_patience: int = 10        # epochs without validation improvement before stopping
    gradient_clip_norm: float | None = 1.0
    mixed_precision: str = "bf16"            # bf16 | fp16 | none  (RTX 30xx and newer: bf16)
    seed: int = 0
    deterministic: bool = False              # True -> bit-identical reruns on the same GPU (slower)
    num_workers: int = 0                     # DataLoader workers; keep 0 on Windows


def seed_everything(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_dtype(recipe: TrainingRecipe, device: torch.device):
    if device.type != "cuda":
        return None
    return {"bf16": torch.bfloat16, "fp16": torch.float16}.get(recipe.mixed_precision)


def move_to_device(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def loss_function_for(model, recipe: TrainingRecipe):
    loss_name = "pinball" if model.quantile_levels else recipe.loss
    loss = get_loss(loss_name)
    return lambda forecast, batch: loss(forecast.float(), batch["target"], batch["target_mask"], model.quantile_levels)


@torch.no_grad()
def predict(model, dataset, recipe: TrainingRecipe, device=None) -> dict:
    """Run a model over a dataset -> numpy arrays: forecast, target, target_mask, smart_persistence, origin_index."""
    device = device or pick_device()
    model.to(device).eval()
    collected = {k: [] for k in ("forecast", "target", "target_mask", "smart_persistence", "origin_index")}
    dtype = autocast_dtype(recipe, device)
    for batch in make_data_loader(dataset, 4 * recipe.batch_size, shuffle=False, num_workers=recipe.num_workers):
        batch = move_to_device(batch, device)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            collected["forecast"].append(model(batch)["forecast"].float().cpu())
        for key in ("target", "target_mask", "smart_persistence", "origin_index"):
            collected[key].append(batch[key].cpu())
    return {k: torch.cat(v).numpy() for k, v in collected.items()}


def train_model(model, data_bundle, recipe: TrainingRecipe, checkpoint_path=None, log=print) -> list[dict]:
    """Fit on data_bundle.datasets["train"], early-stop on "validation", restore the best weights.
    Returns the per-epoch history [{"epoch", "train_loss", "validation_loss", "learning_rate", "seconds"}]."""
    seed_everything(recipe.seed, recipe.deterministic)
    device = pick_device()
    model.to(device)
    if not model.is_trainable:
        log(f"{type(model).__name__} has nothing to train - evaluation only")
        return []
    loss_function = loss_function_for(model, recipe)
    train_loader = make_data_loader(data_bundle.datasets["train"], recipe.batch_size, True, recipe.seed,
                                    recipe.num_workers)
    target_domain_batches = None
    if "domain_adaptation" in data_bundle.datasets:
        target_domain_batches = itertools.cycle(make_data_loader(data_bundle.datasets["domain_adaptation"],
                                                                 recipe.batch_size, True, recipe.seed))
    optimizer = make_optimizer(recipe.optimizer, model.parameters(), recipe.learning_rate, recipe.weight_decay,
                               **recipe.optimizer_options)
    scheduler, scheduler_step_on = make_scheduler(recipe.scheduler, optimizer, recipe.n_epochs, len(train_loader),
                                                  **recipe.scheduler_options)
    dtype = autocast_dtype(recipe, device)
    gradient_scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)
    best_validation_loss, best_state, epochs_without_improvement, history = float("inf"), None, 0, []

    for epoch in range(recipe.n_epochs):
        if hasattr(model, "on_epoch_start"):
            model.on_epoch_start(epoch, recipe.n_epochs)
        model.train()
        started, loss_sum, n_batches = time.time(), 0.0, 0
        for batch in train_loader:
            batch = move_to_device(batch, device)
            if target_domain_batches is not None:
                batch["target_domain_input"] = model.input_adapter(move_to_device(next(target_domain_batches), device))
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                output = model(batch)
                loss = loss_function(output["forecast"], batch)
                if "auxiliary_loss" in output:
                    loss = loss + output["auxiliary_loss"].float()
            optimizer.zero_grad(set_to_none=True)
            gradient_scaler.scale(loss).backward()
            if recipe.gradient_clip_norm:
                gradient_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.gradient_clip_norm)
            gradient_scaler.step(optimizer)
            gradient_scaler.update()
            if scheduler_step_on == "batch":
                scheduler.step()
            loss_sum, n_batches = loss_sum + loss.item(), n_batches + 1

        validation = predict(model, data_bundle.datasets["validation"], recipe, device)
        validation_loss = loss_function(torch.from_numpy(validation["forecast"]),
                                        {k: torch.from_numpy(validation[k]) for k in ("target", "target_mask")}).item()
        if scheduler_step_on == "epoch":
            scheduler.step()
        elif scheduler_step_on == "validation_loss":
            scheduler.step(validation_loss)
        history.append({"epoch": epoch, "train_loss": loss_sum / max(n_batches, 1), "validation_loss": validation_loss,
                        "learning_rate": optimizer.param_groups[0]["lr"], "seconds": time.time() - started})
        log(f"epoch {epoch:3d}  train {history[-1]['train_loss']:.5f}  validation {validation_loss:.5f}  "
            f"lr {history[-1]['learning_rate']:.2e}  ({history[-1]['seconds']:.1f}s)")
        if validation_loss < best_validation_loss - 1e-6:
            best_validation_loss, epochs_without_improvement = validation_loss, 0
            best_state = copy.deepcopy(model.state_dict())
            if checkpoint_path:
                torch.save(best_state, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= recipe.early_stopping_patience:
                log(f"early stop: no validation improvement for {recipe.early_stopping_patience} epochs")
                break
    if best_state is None:
        raise RuntimeError("validation loss was never finite (training diverged) - lower learning_rate or check data")
    model.load_state_dict(best_state)
    return history
