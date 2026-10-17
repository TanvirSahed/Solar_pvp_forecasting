"""Optimizers and learning-rate schedulers, selectable by name (all from torch).

Add one: a line in OPTIMIZERS / SCHEDULERS. Extra arguments come from TrainingRecipe.optimizer_options /
scheduler_options, e.g. scheduler_options={"step_size": 20, "gamma": 0.5} for "step".
"""
from __future__ import annotations

import torch

OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
}

# factory(optimizer, n_epochs, steps_per_epoch, **options) -> scheduler, and when the trainer calls .step()
SCHEDULERS = {
    "none": (lambda opt, n_epochs, steps, **o: None, None),
    "cosine": (lambda opt, n_epochs, steps, **o: torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs, **o),
               "epoch"),
    "step": (lambda opt, n_epochs, steps, step_size=20, gamma=0.5, **o:
             torch.optim.lr_scheduler.StepLR(opt, step_size, gamma, **o), "epoch"),
    "reduce_on_plateau": (lambda opt, n_epochs, steps, factor=0.5, patience=4, **o:
                          torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=factor, patience=patience, **o),
                          "validation_loss"),
    "one_cycle": (lambda opt, n_epochs, steps, **o: torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=opt.param_groups[0]["lr"], epochs=n_epochs, steps_per_epoch=steps, **o), "batch"),
}


def make_optimizer(name: str, parameters, learning_rate: float, weight_decay: float, **options):
    if name not in OPTIMIZERS:
        raise KeyError(f"unknown optimizer {name!r}. Available: {list(OPTIMIZERS)}")
    return OPTIMIZERS[name](parameters, lr=learning_rate, weight_decay=weight_decay, **options)


def make_scheduler(name: str, optimizer, n_epochs: int, steps_per_epoch: int, **options):
    """-> (scheduler or None, step_on) with step_on in {"epoch", "batch", "validation_loss", None}."""
    if name not in SCHEDULERS:
        raise KeyError(f"unknown scheduler {name!r}. Available: {list(SCHEDULERS)}")
    factory, step_on = SCHEDULERS[name]
    return factory(optimizer, n_epochs, steps_per_epoch, **options), step_on
