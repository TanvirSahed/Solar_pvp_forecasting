"""Reusable adapter steps (ForecastBatch -> model input). Each step ADDS named tensors and keeps the rest.

Steps adapt to whatever features the DataConfig selected: a requested feature that is not in the history is
skipped with a printed note instead of failing, so changing the feature set never breaks a model.
"""
from __future__ import annotations

import torch

from ..data.features import CALENDAR_FEATURE_NAMES
from .input_adapter import AdapterStep


def measured_feature_names(data_spec) -> list[str]:
    """History features that are measurements or physics, not calendar encodings."""
    return [n for n in data_spec.history_feature_names if n not in CALENDAR_FEATURE_NAMES]


def keep_available(data_spec, requested_names, step_name) -> list[str]:
    available = [n for n in requested_names if n in data_spec.history_feature_names]
    missing = [n for n in requested_names if n not in data_spec.history_feature_names]
    if missing:
        print(f"note: {step_name} skips {missing} (not in history_feature_names)")
    return available


class SelectTargetHistory(AdapterStep):
    description = "past target series only (univariate models)"

    def __init__(self, data_spec):
        self.target_index = data_spec.target_feature_index

    def __call__(self, model_input):
        model_input["target_history"] = model_input["history_features"][..., self.target_index]   # [batch, lookback]
        return model_input


class SplitCalendarFeatures(AdapterStep):
    """Measured values vs calendar features. With no calendar features selected, calendar_history has 0 channels."""
    description = "separate measured values from calendar features"

    def __init__(self, data_spec):
        names = data_spec.history_feature_names
        self.calendar_indices = [i for i, n in enumerate(names) if n in CALENDAR_FEATURE_NAMES]
        self.measurement_indices = [i for i, n in enumerate(names) if n not in CALENDAR_FEATURE_NAMES]

    def __call__(self, model_input):
        history = model_input["history_features"]
        model_input["measurement_history"] = history[..., self.measurement_indices]   # [batch, lookback, n_measured]
        model_input["calendar_history"] = history[..., self.calendar_indices]         # [batch, lookback, n_calendar]
        return model_input


class BuildVariableNodes(AdapterStep):
    """Graph nodes from variables: node i = series of node feature i + the shared features (calendar by default).
    node_feature_names=None -> every measured history feature becomes a node.
    With per-inverter data, write a BuildInverterNodes step instead; the graph model stays unchanged."""
    description = "one graph node per variable: [its own series, shared features]"

    def __init__(self, data_spec, node_feature_names=None, shared_feature_names=None):
        node_names = keep_available(data_spec, node_feature_names or measured_feature_names(data_spec), self.name)
        self.node_names = node_names or [data_spec.target_feature_name]
        shared_names = shared_feature_names if shared_feature_names is not None else CALENDAR_FEATURE_NAMES
        shared_names = [n for n in shared_names if n in data_spec.history_feature_names]
        self.node_indices = [data_spec.history_feature_names.index(n) for n in self.node_names]
        self.shared_indices = [data_spec.history_feature_names.index(n) for n in shared_names]
        self.n_nodes, self.node_input_size = len(self.node_indices), 1 + len(self.shared_indices)

    def __call__(self, model_input):
        history = model_input["history_features"]
        shared = history[..., self.shared_indices]
        model_input["node_series"] = torch.stack(
            [torch.cat([history[..., i:i + 1], shared], -1) for i in self.node_indices], 1)  # [batch, nodes, lookback, F]
        return model_input


class SelectVariableSeries(AdapterStep):
    """Chosen variables as separate series (variate tokens). feature_names=None -> every measured feature.
    The target is always placed first."""
    description = "chosen variables as separate series, target first"

    def __init__(self, data_spec, feature_names=None):
        names = keep_available(data_spec, feature_names or measured_feature_names(data_spec), self.name)
        target = data_spec.target_feature_name
        self.feature_names = [target] + [n for n in names if n != target]
        self.indices = [data_spec.history_feature_names.index(n) for n in self.feature_names]

    def __call__(self, model_input):
        model_input["variable_series"] = model_input["history_features"][..., self.indices].transpose(1, 2)  # [b, n, L]
        return model_input


class AppendFutureToHistory(AdapterStep):
    """PowerMamba-style: one sequence of length lookback+horizon. History channels are 0 in the future part,
    future channels are 0 in the past part."""
    description = "concatenate history and known future into one longer sequence"

    def __call__(self, model_input):
        history, future = model_input["history_features"], model_input["future_features"]
        batch, lookback, horizon = history.shape[0], history.shape[1], future.shape[1]
        model_input["extended_sequence"] = torch.cat([
            torch.cat([history, history.new_zeros(batch, horizon, history.shape[2])], 1),
            torch.cat([history.new_zeros(batch, lookback, future.shape[2]), future], 1)], 2)  # [b, L+H, n_hist+n_fut]
        return model_input


class SummariseWeatherRegime(AdapterStep):
    """Clearness-index statistics over the last day: [mean, std, latest] - a sunny / cloudy / rainy descriptor.
    Needs irradiance and clear_sky_irradiance in the history; without them it returns zeros (regime unknown)."""
    description = "last-day clearness index: mean, std, latest value"

    def __init__(self, data_spec):
        names = data_spec.history_feature_names
        self.has_inputs = "irradiance" in names and "clear_sky_irradiance" in names
        if self.has_inputs:
            self.irradiance_index, self.clear_sky_index = names.index("irradiance"), names.index("clear_sky_irradiance")
        else:
            print("note: SummariseWeatherRegime needs irradiance and clear_sky_irradiance in history -> regime = 0")
        self.steps = data_spec.steps_per_day

    def __call__(self, model_input):
        history = model_input["history_features"][:, -self.steps:]
        if not self.has_inputs:
            model_input["weather_regime"] = history.new_zeros(history.shape[0], 3)
            return model_input
        irradiance, clear_sky = history[..., self.irradiance_index], history[..., self.clear_sky_index]
        is_day = (clear_sky > 0.01).float()
        clearness = (irradiance / clear_sky.clamp_min(0.01)).clamp(0, 1.5) * is_day
        n_day = is_day.sum(1).clamp_min(1)
        mean = clearness.sum(1) / n_day
        std = torch.sqrt(((clearness - mean[:, None]) ** 2 * is_day).sum(1) / n_day)
        model_input["weather_regime"] = torch.stack([mean, std, clearness[:, -1]], 1)                 # [batch, 3]
        return model_input
