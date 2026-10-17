"""InputAdapter: turns the standard ForecastBatch into one model's input, as a list of named steps.

Steps are pure tensor reshaping/selection (no learnable weights), so they can be inspected on their own:

    adapter = model.input_adapter
    adapter                         # GraphMamba input: ForecastBatch -> BuildVariableNodes
    adapter.trace(batch)            # table: step, what it does, every tensor shape it produced
    adapter.plot(batch)             # heatmaps of each step's new tensors for one sample

Write a new step: subclass AdapterStep, implement __call__(model_input) -> model_input (add keys, keep the rest).
"""
from __future__ import annotations

import pandas as pd
import torch


class AdapterStep:
    description = ""

    @property
    def name(self) -> str:
        return type(self).__name__

    def __call__(self, model_input: dict) -> dict:
        raise NotImplementedError


class InputAdapter:
    def __init__(self, steps: list[AdapterStep] | None = None, model_name: str = ""):
        self.steps = list(steps or [])
        self.model_name = model_name
        self.data_spec = None                               # set by the model; used for feature names in plot()

    def __call__(self, batch: dict) -> dict:
        model_input = dict(batch)
        for step in self.steps:
            model_input = step(model_input)
        return model_input

    def __repr__(self) -> str:
        return f"{self.model_name} input: " + " -> ".join(["ForecastBatch"] + [s.name for s in self.steps])

    def _stages(self, batch: dict):
        """Yield (step name, description, model_input after the step, keys the step added or changed)."""
        model_input = dict(batch)
        yield "ForecastBatch", "standard batch from the data pipeline", model_input, list(model_input)
        for step in self.steps:
            new_input = step(dict(model_input))
            changed = [k for k, v in new_input.items() if k not in model_input or v is not model_input[k]]
            yield step.name, step.description, new_input, changed
            model_input = new_input

    def trace(self, batch: dict) -> pd.DataFrame:
        rows = []
        for step_name, description, model_input, changed in self._stages(batch):
            for key in changed:
                value = model_input[key]
                if torch.is_tensor(value):
                    rows.append({"step": step_name, "does": description, "tensor": key, "shape": tuple(value.shape)})
        return pd.DataFrame(rows)

    def _feature_names(self, key):
        if self.data_spec is None:
            return None
        return {"history_features": self.data_spec.history_feature_names,
                "future_features": self.data_spec.future_feature_names}.get(key)

    def plot(self, batch: dict, sample_index: int = 0, max_tensors_per_step: int = 6):
        """One row per step; each new tensor of one sample drawn as a line (1-D) or heatmap (>= 2-D)."""
        import matplotlib.pyplot as plt

        stages = [(s, d, {k: m[k] for k in c if torch.is_tensor(m[k]) and m[k].dim() >= 2}) for s, d, m, c in
                  self._stages(batch)]
        stages = [stage for stage in stages if stage[2]]
        n_columns = min(max_tensors_per_step, max(len(t) for _, _, t in stages))
        figure, axes = plt.subplots(len(stages), n_columns, figsize=(3.2 * n_columns, 2.6 * len(stages)), squeeze=False)
        for row, (step_name, _, tensors) in enumerate(stages):
            for column in range(n_columns):
                axis = axes[row][column]
                if column >= len(tensors):
                    axis.axis("off")
                    continue
                key, value = list(tensors.items())[column]
                sample = value[sample_index].detach().float().cpu()
                if sample.dim() == 1:
                    axis.plot(sample.numpy())
                else:  # time is the second-to-last axis of every tensor here -> time on x, everything else on y
                    axis.imshow(sample.movedim(-2, -1).reshape(-1, sample.shape[-2]).numpy(), aspect="auto",
                                cmap="viridis")
                    axis.set_xlabel("time step", fontsize=6)
                    row_names = self._feature_names(key)
                    if row_names and len(row_names) == sample.shape[-1] and sample.dim() == 2:
                        axis.set_yticks(range(len(row_names)), row_names, fontsize=6)
                axis.set_title(f"{key} {tuple(sample.shape)}", fontsize=8)
                axis.tick_params(labelsize=6)
            axes[row][0].set_ylabel(step_name, fontsize=8)
        figure.suptitle(repr(self), fontsize=9)
        figure.tight_layout()
        return figure
