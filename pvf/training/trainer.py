"""TrainingRecipe (everything except the architecture and the data) and the trainer.

The trainer knows models only through the contract in models/base.py:
    model(forecast_batch) -> {"forecast", optional "auxiliary_loss"}
    model.is_trainable, model.quantile_levels, optional model.on_epoch_start(epoch, n_epochs)
Paper-specific recipes: training/paper_recipes.py and docs/training_recipes.md.
"""
from __future__ import annotations

import copy
import itertools
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from ..data.pipeline import make_data_loader
from .losses import get_loss
from .optimizers import make_optimizer, make_scheduler


@dataclass
class TrainingRecipe:
    # loss and optimisation
    loss: str = "mse"                        # mse | mae | huber | pinball (quantile models always use pinball)
    loss_options: dict = field(default_factory=dict)          # e.g. {"delta": 0.05} for huber
    optimizer: str = "adamw"                 # adam | adamw | sgd | rmsprop
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer_options: dict = field(default_factory=dict)     # e.g. {"betas": (0.9, 0.999)}
    scheduler: str = "reduce_on_plateau"     # none | cosine | step | reduce_on_plateau | one_cycle
    scheduler_options: dict = field(default_factory=dict)     # e.g. {"factor": 0.5, "patience": 2}
    warmup_epochs: int = 0                   # linear learning-rate warm-up before the scheduler takes over
    # length and stopping
    n_epochs: int = 100
    early_stopping_metric: str = "validation_loss"   # validation_loss | nmae | nrmse
    early_stopping_patience: int | None = 10         # None -> train exactly n_epochs (keep the best epoch)
    early_stopping_min_improvement: float = 0.0      # in the metric's units
    time_limit_minutes: float | None = None          # stop after this much training time (best epoch kept)
    # batches
    batch_size: int = 64
    gradient_accumulation_steps: int = 1     # effective batch = batch_size x this
    gradient_clip_norm: float | None = 1.0
    mixed_precision: str = "bf16"            # bf16 | fp16 | none
    # data use and initialisation
    train_fraction: float = 1.0              # < 1: random share of training windows (data-efficiency curves)
    initialize_from_run: str | None = None   # run folder whose best weights start this training (transfer)
    freeze_parameter_prefixes: list = field(default_factory=list)   # e.g. ["encoder."]: keep those weights fixed
    seed: int = 0
    num_workers: int = 0                     # DataLoader workers; keep 0 on Windows


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_dtype(recipe: TrainingRecipe, device: torch.device):
    if device.type != "cuda":
        return None
    return {"bf16": torch.bfloat16, "fp16": torch.float16}.get(recipe.mixed_precision)


def move_to_device(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def loss_function_for(model, recipe: TrainingRecipe):
    loss = get_loss("pinball" if model.quantile_levels else recipe.loss)
    return lambda forecast, batch: loss(forecast.float(), batch["target"], batch["target_mask"],
                                        model.quantile_levels, **recipe.loss_options)


@torch.no_grad()
def predict(model, dataset, recipe: TrainingRecipe, device=None) -> dict:
    """Run a model over a dataset -> numpy arrays: forecast, target, target_mask, smart_persistence, origin_index."""
    device = device or pick_device()
    model.to(device).eval()
    collected = {k: [] for k in ("forecast", "target", "target_mask", "smart_persistence", "origin_index")}
    dtype = autocast_dtype(recipe, device)
    for batch in make_data_loader(dataset, recipe.batch_size, shuffle=False, num_workers=recipe.num_workers):
        batch = move_to_device(batch, device)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            collected["forecast"].append(model(batch)["forecast"].float().cpu())
        for key in ("target", "target_mask", "smart_persistence", "origin_index"):
            collected[key].append(batch[key].cpu())
    return {k: torch.cat(v).numpy() for k, v in collected.items()}


def _point_forecast(model, forecast):
    if model.quantile_levels:
        levels = list(model.quantile_levels)
        return forecast[..., levels.index(0.5)] if 0.5 in levels else forecast.mean(-1)
    return forecast


def validation_scores(model, predictions, loss_function) -> dict:
    point = np.clip(_point_forecast(model, predictions["forecast"]), 0, None)
    mask = predictions["target_mask"] > 0
    error = (point - predictions["target"])[mask]
    loss = loss_function(torch.from_numpy(predictions["forecast"]),
                         {k: torch.from_numpy(predictions[k]) for k in ("target", "target_mask")}).item()
    return {"validation_loss": loss, "nmae": 100 * float(np.abs(error).mean()),
            "nrmse": 100 * float(np.sqrt((error ** 2).mean()))}


def prepare_model(model, recipe: TrainingRecipe, log=print):
    """Optional transfer: load weights from another run, freeze chosen parameter groups."""
    if recipe.initialize_from_run:
        weights = Path(recipe.initialize_from_run) / "best_model.pt"
        state = torch.load(weights, map_location="cpu")
        result = model.load_state_dict(state, strict=False)
        log(f"initialised from {weights} (missing {len(result.missing_keys)}, unused {len(result.unexpected_keys)} tensors)")
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in recipe.freeze_parameter_prefixes):
            parameter.requires_grad_(False)
    if recipe.freeze_parameter_prefixes:
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        log(f"frozen {frozen:,} parameters with prefixes {recipe.freeze_parameter_prefixes}")


def train_model(model, data_bundle, recipe: TrainingRecipe, checkpoint_path=None, log=print) -> list[dict]:
    """Fit on data_bundle.datasets["train"], select the best epoch on "validation", restore its weights.
    Returns the per-epoch history (losses, validation nMAE / nRMSE, learning rate, seconds)."""
    seed_everything(recipe.seed)
    device = pick_device()
    if not model.is_trainable:
        model.to(device)
        log(f"{type(model).__name__} has nothing to train - evaluation only")
        return []
    if recipe.early_stopping_metric not in ("validation_loss", "nmae", "nrmse"):
        raise ValueError("early_stopping_metric must be 'validation_loss', 'nmae' or 'nrmse'")
    prepare_model(model, recipe, log)
    model.to(device)
    loss_function = loss_function_for(model, recipe)
    train_dataset = data_bundle.datasets["train"]
    if recipe.train_fraction < 1.0:
        n_keep = max(1, int(len(train_dataset) * recipe.train_fraction))
        keep = np.sort(np.random.default_rng(recipe.seed).choice(len(train_dataset), n_keep, replace=False))
        train_dataset = Subset(train_dataset, keep.tolist())
        log(f"training on {n_keep} of {len(data_bundle.datasets['train'])} windows ({recipe.train_fraction:.0%})")
    train_loader = make_data_loader(train_dataset, recipe.batch_size, True, recipe.seed, recipe.num_workers)
    target_domain_batches = None
    if "domain_adaptation" in data_bundle.datasets:
        target_domain_batches = itertools.cycle(make_data_loader(data_bundle.datasets["domain_adaptation"],
                                                                 recipe.batch_size, True, recipe.seed))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_optimizer(recipe.optimizer, trainable, recipe.learning_rate, recipe.weight_decay,
                               **recipe.optimizer_options)
    steps_per_epoch = -(-len(train_loader) // recipe.gradient_accumulation_steps)   # ceil: last partial step counts
    scheduler, scheduler_step_on = make_scheduler(recipe.scheduler, optimizer, recipe.n_epochs, steps_per_epoch,
                                                  **recipe.scheduler_options)
    dtype = autocast_dtype(recipe, device)
    gradient_scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)
    best_score, best_state, epochs_without_improvement, history = float("inf"), None, 0, []
    training_started = time.time()

    for epoch in range(recipe.n_epochs):
        if hasattr(model, "on_epoch_start"):
            model.on_epoch_start(epoch, recipe.n_epochs)
        in_warmup = epoch < recipe.warmup_epochs
        if in_warmup:
            for group in optimizer.param_groups:
                group["lr"] = recipe.learning_rate * (epoch + 1) / recipe.warmup_epochs
        model.train()
        started, loss_sum, n_batches = time.time(), 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for batch_number, batch in enumerate(train_loader, start=1):
            batch = move_to_device(batch, device)
            if target_domain_batches is not None:
                batch["target_domain_input"] = model.input_adapter(move_to_device(next(target_domain_batches), device))
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                output = model(batch)
                loss = loss_function(output["forecast"], batch)
                if "auxiliary_loss" in output:
                    loss = loss + output["auxiliary_loss"].float()
            gradient_scaler.scale(loss / recipe.gradient_accumulation_steps).backward()
            loss_sum, n_batches = loss_sum + loss.item(), n_batches + 1
            if batch_number % recipe.gradient_accumulation_steps and batch_number != len(train_loader):
                continue
            if recipe.gradient_clip_norm:
                gradient_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, recipe.gradient_clip_norm)
            gradient_scaler.step(optimizer)
            gradient_scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler_step_on == "batch" and not in_warmup:
                scheduler.step()

        scores = validation_scores(model, predict(model, data_bundle.datasets["validation"], recipe, device),
                                   loss_function)
        if not in_warmup and scheduler_step_on == "epoch":
            scheduler.step()
        elif not in_warmup and scheduler_step_on == "validation_loss":
            scheduler.step(scores["validation_loss"])
        history.append({"epoch": epoch, "train_loss": loss_sum / max(n_batches, 1), **scores,
                        "learning_rate": optimizer.param_groups[0]["lr"], "seconds": time.time() - started})
        log(f"epoch {epoch:3d}  train {history[-1]['train_loss']:.5f}  val loss {scores['validation_loss']:.5f}  "
            f"nMAE {scores['nmae']:.2f}%  nRMSE {scores['nrmse']:.2f}%  lr {history[-1]['learning_rate']:.1e}  "
            f"({history[-1]['seconds']:.1f}s)")
        score = scores[recipe.early_stopping_metric]
        if score < best_score - recipe.early_stopping_min_improvement:
            best_score, epochs_without_improvement = score, 0
            best_state = copy.deepcopy(model.state_dict())
            if checkpoint_path:
                torch.save(best_state, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if recipe.early_stopping_patience and epochs_without_improvement >= recipe.early_stopping_patience:
                log(f"early stop: no {recipe.early_stopping_metric} improvement for {recipe.early_stopping_patience} epochs")
                break
        if recipe.time_limit_minutes and time.time() - training_started > 60 * recipe.time_limit_minutes:
            log(f"time limit: stopped after {recipe.time_limit_minutes} min ({epoch + 1} epochs)")
            break
    if best_state is None:
        raise RuntimeError("validation score was never finite (training diverged) - lower learning_rate or check data")
    model.load_state_dict(best_state)
    return history
