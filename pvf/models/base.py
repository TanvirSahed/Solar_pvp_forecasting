"""Model contract + registry.

Contract (all models):
    __init__(spec: DataSpec, **hyperparams)
    forward(batch: dict) -> dict with
        "pred":     [B, H]  point forecast of P/P_rated   (or [B, H, Q] if self.quantiles)
        "aux_loss": optional scalar added to the training loss (DANN, regularisers, ...)
        anything else you want to log/inspect (attention, alpha_t, node states ...)

Add a model:  decorate a ForecastModel subclass with @register("my_model") and put a YAML in configs/models/.
"""
from __future__ import annotations

import torch
import torch.nn as nn

MODELS: dict[str, type] = {}


def register(name: str):
    def deco(cls):
        if name in MODELS:
            raise KeyError(f"model {name!r} registered twice")
        MODELS[name] = cls
        cls.registry_name = name
        return cls
    return deco


class ForecastModel(nn.Module):
    trainable: bool = True          # False -> trainer only evaluates (persistence, zero-shot foundation models)
    quantiles: tuple | None = None  # set -> output [B, H, Q] and pinball loss is used

    def __init__(self, spec, quantiles=None, **_):
        super().__init__()
        self.spec = spec
        self.H, self.L = spec.horizon, spec.lookback
        self.quantiles = tuple(quantiles) if quantiles else None

    @property
    def n_q(self) -> int:
        return len(self.quantiles) if self.quantiles else 0


class AnchorResidual(ForecastModel):
    """Physics-anchored residual learning: pred = s(t,h) + f(x). Smart persistence carries the clear-sky
    geometry, the network only learns the cloud/temperature/soiling correction. Wraps ANY model."""

    def __init__(self, inner: ForecastModel):
        super().__init__(inner.spec, inner.quantiles)
        self.inner = inner
        self.trainable = inner.trainable

    def forward(self, batch):
        out = self.inner(batch)
        sp = batch["sp"]
        out["pred"] = out["pred"] + (sp.unsqueeze(-1) if out["pred"].dim() == 3 else sp)
        return out


def build_model(mcfg: dict, spec) -> ForecastModel:
    from . import zoo  # noqa: F401  (populates MODELS)

    mcfg = dict(mcfg)
    name = mcfg.pop("name")
    anchor = mcfg.pop("anchor", None)
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}. Registered: {sorted(MODELS)}")
    model = MODELS[name](spec, **mcfg)
    if anchor == "sp":
        model = AnchorResidual(model)
    return model


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def target_hist(batch, spec) -> torch.Tensor:
    """Past target series [B, L] (normalised P)."""
    return batch["x_hist"][..., spec.target_idx]
