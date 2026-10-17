"""The model contract - the interface between an architecture and the training module.

Every model:
    __init__(data_spec, <hyperparameters as keyword arguments>, input_adapter=None)
    forward(forecast_batch) -> ModelOutput dict     (so a model is directly callable on DataLoader batches)
        runs self.input_adapter (ForecastBatch -> model input), then self.forecast(model_input)
    encode(model_input)   -> representation          (the reusable part; copy/import it to build fused models)
    forecast(model_input) -> {"forecast": [batch, horizon] or [batch, horizon, n_quantiles], ...}

ModelOutput keys the trainer understands:
    "forecast"         required; ac_power / rated power at each horizon step (or quantiles, see quantile_levels)
    "auxiliary_loss"   optional scalar added to the training loss (domain adaptation, regularisers)
Any other key (attention maps, gates, adjacency) is returned for inspection only.
"""
from __future__ import annotations

import torch.nn as nn

from ..model_inputs.input_adapter import InputAdapter

MODEL_REGISTRY: dict[str, type] = {}


def register_model(name: str):
    """Make a model selectable by name (notebook, experiment YAML)."""
    def decorator(model_class):
        if name in MODEL_REGISTRY:
            raise KeyError(f"model name {name!r} registered twice")
        MODEL_REGISTRY[name] = model_class
        model_class.registry_name = name
        return model_class
    return decorator


class ForecastModel(nn.Module):
    is_trainable: bool = True      # False -> the trainer only evaluates it (smart persistence, zero-shot models)

    def __init__(self, data_spec, input_adapter: InputAdapter | None = None, quantile_levels=None):
        super().__init__()
        self.data_spec = data_spec
        self.lookback_steps, self.horizon_steps = data_spec.lookback_steps, data_spec.horizon_steps
        self.quantile_levels = tuple(quantile_levels) if quantile_levels else None
        self.input_adapter = input_adapter or InputAdapter([], model_name=type(self).__name__)
        self.input_adapter.data_spec = data_spec          # lets adapter plots name the feature rows

    @property
    def n_quantiles(self) -> int:
        return len(self.quantile_levels) if self.quantile_levels else 0

    @property
    def n_outputs_per_step(self) -> int:
        return max(1, self.n_quantiles)

    def forward(self, forecast_batch: dict) -> dict:
        return self.forecast(self.input_adapter(forecast_batch))

    def forecast(self, model_input: dict) -> dict:
        raise NotImplementedError

    def shape_output(self, values):
        """[batch, horizon * outputs] or [batch, horizon, outputs] -> [batch, horizon] (point) / [batch, horizon, q]."""
        values = values.reshape(values.shape[0], self.horizon_steps, self.n_outputs_per_step)
        return values if self.n_quantiles else values.squeeze(-1)


class SmartPersistenceResidual(ForecastModel):
    """forecast = smart persistence + model(x). Physics carries the clear-sky shape; the network learns the
    cloud / temperature correction. Wraps ANY model: build_model(..., residual_on_smart_persistence=True)."""

    def __init__(self, inner_model: ForecastModel):
        super().__init__(inner_model.data_spec, inner_model.input_adapter, inner_model.quantile_levels)
        self.inner_model = inner_model
        self.is_trainable = inner_model.is_trainable

    def forward(self, forecast_batch):
        output = self.inner_model(forecast_batch)
        persistence = forecast_batch["smart_persistence"]
        output["forecast"] = output["forecast"] + (persistence.unsqueeze(-1) if self.n_quantiles else persistence)
        return output


def build_model(model_name: str, data_spec, residual_on_smart_persistence: bool = False, **hyperparameters):
    from . import architectures  # noqa: F401  (registers every model)

    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {model_name!r}. Available: {sorted(MODEL_REGISTRY)}")
    model = MODEL_REGISTRY[model_name](data_spec, **hyperparameters)
    return SmartPersistenceResidual(model) if residual_on_smart_persistence else model


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def describe_models() -> "pandas.DataFrame":
    """Every registered model with its hyperparameters (constructor arguments) and their defaults."""
    import inspect

    import pandas as pd

    from . import architectures  # noqa: F401

    rows = []
    for name, model_class in sorted(MODEL_REGISTRY.items()):
        parameters = inspect.signature(model_class.__init__).parameters.values()
        hyperparameters = {p.name: p.default for p in parameters if p.name not in ("self", "data_spec", "input_adapter")}
        rows.append({"model_name": name, "class": model_class.__name__, "trainable": model_class.is_trainable,
                     "module": model_class.__module__.rsplit(".", 1)[-1], "hyperparameters (defaults)": hyperparameters})
    return pd.DataFrame(rows)
