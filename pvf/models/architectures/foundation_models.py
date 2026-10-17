"""Pretrained time-series foundation models, zero-shot (selection report rows 7-9). Not trained: is_trainable=False.

chronos2  Chronos-2 (R12). Uses past covariates (all history features) and known-future covariates natively.
timesfm   TimesFM 2.5 (R14), univariate on ac_power.
tirex     TiRex (F07; TiRex-2 R13 when released), univariate on ac_power.

Weights are plain files in Data/pretrained/<org>--<name>/ (they travel with the Data folder and load offline).
If missing they are downloaded there once, on first use.
These models like long context: try DataConfig(lookback_steps=512).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ...data.sites import DATA_DIR
from ...model_inputs.adapter_steps import SelectTargetHistory
from ...model_inputs.input_adapter import InputAdapter
from ..base import ForecastModel, register_model

PRETRAINED_WEIGHTS_DIR = DATA_DIR / "pretrained"
WEIGHTS_REPOSITORIES = {
    "chronos2": "amazon/chronos-2",
    "timesfm": "google/timesfm-2.5-200m-pytorch",
    "tirex": "NX-AI/TiRex",
}
DECILE_LEVELS = [round(0.1 * i, 1) for i in range(1, 10)]


def local_weights(repository: str) -> Path:
    """Folder with the repository's weights inside the project; downloaded there (not to the user cache) if absent."""
    if Path(repository).exists():
        return Path(repository)
    folder = PRETRAINED_WEIGHTS_DIR / repository.replace("/", "--")
    if not (folder / ".complete").exists():
        from huggingface_hub import snapshot_download

        print(f"downloading {repository} -> {folder} (one time)")
        snapshot_download(repository, local_dir=folder, allow_patterns=["*.json", "*.safetensors", "*.ckpt", "*.bin"])
        (folder / ".complete").write_text(repository + "\n")
    return folder


class _ZeroShotModel(ForecastModel):
    is_trainable = False
    needs_weight_download = True        # skipped by `pvf smoke` unless named explicitly

    def __init__(self, data_spec, return_quantiles=False, input_adapter=None):
        super().__init__(data_spec, input_adapter or InputAdapter([SelectTargetHistory(data_spec)], type(self).__name__),
                         DECILE_LEVELS if return_quantiles else None)
        self._device_anchor = nn.Parameter(torch.zeros(1), requires_grad=False)   # lets .to(device) move "the model"

    @property
    def device(self):
        return self._device_anchor.device

    def _output(self, deciles: torch.Tensor) -> dict:                  # deciles: [batch, horizon, 9]
        deciles = deciles.float().to(self.device)
        return {"forecast": deciles if self.n_quantiles else deciles[..., DECILE_LEVELS.index(0.5)]}


@register_model("chronos2")
class Chronos2(_ZeroShotModel):
    def __init__(self, data_spec, weights=WEIGHTS_REPOSITORIES["chronos2"], use_covariates=True, batch_size=256,
                 return_quantiles=False, input_adapter=None):
        super().__init__(data_spec, return_quantiles, input_adapter)
        from chronos import Chronos2Pipeline

        pipeline = Chronos2Pipeline.from_pretrained(str(local_weights(weights)),
                                                    device_map="cuda" if torch.cuda.is_available() else "cpu")
        object.__setattr__(self, "pipeline", pipeline)       # not an nn.Module attribute: never saved in checkpoints
        self.use_covariates, self.batch_size = use_covariates, batch_size
        self.model_quantile_levels = [round(q, 2) for q in getattr(pipeline, "quantiles", DECILE_LEVELS)]

    def _chronos_inputs(self, model_input):
        spec = self.data_spec
        history = model_input["history_features"].float().cpu().numpy()
        future = model_input["future_features"].float().cpu().numpy()
        target = model_input["target_history"].float().cpu().numpy()
        if not self.use_covariates:
            return list(target)
        inputs = []
        for i in range(len(target)):
            past = {name: history[i, :, j] for j, name in enumerate(spec.history_feature_names)
                    if name != spec.target_feature_name}
            known_future = {}
            for j, name in enumerate(spec.future_feature_names):
                past_name = name.removesuffix("_oracle")
                if past_name in past:                      # known-future covariate that also has a past part
                    known_future[past_name] = future[i, :, j]
                elif name == "ac_power_yesterday":         # its past part is ac_power shifted by one day
                    past[name] = np.concatenate([np.full(spec.steps_per_day, np.nan), target[i]])[:spec.lookback_steps]
                    known_future[name] = future[i, :, j]
            inputs.append({"target": target[i], "past_covariates": past, "future_covariates": known_future})
        return inputs

    @torch.no_grad()
    def forecast(self, model_input):
        with torch.autocast("cuda", enabled=False):
            outputs = self.pipeline.predict(self._chronos_inputs(model_input),
                                            prediction_length=self.horizon_steps, batch_size=self.batch_size)
        quantiles = torch.stack([o[0] for o in outputs]).permute(0, 2, 1)          # [batch, horizon, n_levels]
        deciles = quantiles[..., [self.model_quantile_levels.index(q) for q in DECILE_LEVELS]]
        return self._output(deciles)


@register_model("timesfm")
class TimesFM(_ZeroShotModel):
    def __init__(self, data_spec, weights=WEIGHTS_REPOSITORIES["timesfm"], return_quantiles=False, input_adapter=None):
        super().__init__(data_spec, return_quantiles, input_adapter)
        import timesfm

        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(str(local_weights(weights)))
        model.compile(timesfm.ForecastConfig(
            max_context=max(512, data_spec.lookback_steps), max_horizon=max(128, data_spec.horizon_steps),
            normalize_inputs=True, use_continuous_quantile_head=True, force_flip_invariance=True,
            infer_is_positive=True, fix_quantile_crossing=True))
        object.__setattr__(self, "timesfm_model", model)

    @torch.no_grad()
    def forecast(self, model_input):
        target = model_input["target_history"].float().cpu().numpy()
        with torch.autocast("cuda", enabled=False):
            point, quantiles = self.timesfm_model.forecast(horizon=self.horizon_steps, inputs=list(target))
        if self.n_quantiles:
            return self._output(torch.from_numpy(np.asarray(quantiles)[..., 1:]))   # [:, :, 0] is the mean
        return {"forecast": torch.from_numpy(np.asarray(point)).float().to(self.device)}


@register_model("tirex")
class TiRex(_ZeroShotModel):
    def __init__(self, data_spec, weights=WEIGHTS_REPOSITORIES["tirex"], model_id="TiRex", return_quantiles=False,
                 input_adapter=None):
        super().__init__(data_spec, return_quantiles, input_adapter)
        from tirex.base import PretrainedModel

        # tirex.load_model() infers the class from a Hugging Face id, so load the local checkpoint through the class
        checkpoint = local_weights(weights) / "model.ckpt"
        object.__setattr__(self, "tirex_model", PretrainedModel.REGISTRY[model_id].from_pretrained(
            str(checkpoint), backend="torch", device="cuda:0" if torch.cuda.is_available() else "cpu"))

    @torch.no_grad()
    def forecast(self, model_input):
        with torch.autocast("cuda", enabled=False):
            deciles, _mean = self.tirex_model.forecast(context=model_input["target_history"].float(),
                                                       prediction_length=self.horizon_steps)
        return self._output(deciles)
