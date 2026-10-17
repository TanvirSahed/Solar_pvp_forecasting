"""Pretrained time-series foundation models, zero-shot (selection report rows 7-9).

chronos2 - Chronos-2 (R12), chronos-forecasting>=2.2. Uses past covariates (G, T, Gcs, calendar) and known-future
           covariates (Gcs, calendar, P_lag24h, oracle G/T if enabled) -> its native covariate support.
timesfm  - TimesFM 2.5 (default, batched) or 3.0 (`version: 3`) (R14), univariate on P.
tirex    - TiRex (F07; TiRex-2 R13 when released on HF), univariate on P.

They are not trained by the trainer (trainable=False); `python -m pvf train -m chronos2` just evaluates them.
Context = data.lookback; these models like long context -> try `data.lookback=512` (or more).
Weights download from Hugging Face on first use (cached in HF_HOME). Fine-tuning hooks: TODO (DT synthetic
history fine-tuning, report rows 7-8) - start from `self.fm`.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..base import ForecastModel, register


class _FM(ForecastModel):
    trainable = False
    external = True       # skipped by `pvf smoke` by default (needs weight download)

    def __init__(self, spec, probabilistic=False, **kw):
        super().__init__(spec, **kw)
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)  # gives .to(device) something to move
        self.probabilistic = probabilistic

    @property
    def dev(self):
        return self._dummy.device

    def _finish(self, q: torch.Tensor, qlevels):
        """q: [B, H, Q] on any device -> output dict (median point or full quantiles)."""
        q = q.float().to(self.dev)
        if self.probabilistic:
            self.quantiles = tuple(float(x) for x in qlevels)
            return {"pred": q}
        return {"pred": q[..., list(np.round(qlevels, 3)).index(0.5)]}


@register("chronos2")
class Chronos2(_FM):
    def __init__(self, spec, checkpoint="amazon/chronos-2", use_covariates=True, batch_size=256, **kw):
        super().__init__(spec, **kw)
        from chronos import Chronos2Pipeline

        object.__setattr__(self, "fm", Chronos2Pipeline.from_pretrained(
            checkpoint, device_map="cuda" if torch.cuda.is_available() else "cpu"))
        self.cov, self.bs = use_covariates, batch_size
        self.qlevels = list(getattr(self.fm, "quantiles", [0.1 * i for i in range(1, 10)]))

    @torch.no_grad()
    def forward(self, batch):
        s = self.spec
        xh, xf = batch["x_hist"].float().cpu().numpy(), batch["x_fut"].float().cpu().numpy()
        P = xh[..., s.target_idx]
        if not self.cov:
            inputs = [p for p in P]
        else:
            inputs = []
            for i in range(len(P)):
                past = {n: xh[i, :, j] for j, n in enumerate(s.hist_names) if n != s.target}
                fut = {}
                for j, n in enumerate(s.fut_names):
                    key = n.replace("_oracle", "")
                    if key in past:                 # known-future covariate with a past part
                        fut[key] = xf[i, :, j]
                    elif n == "P_lag24h":           # yesterday's power: past part = P(t-24h) over the context
                        past[n] = np.concatenate([np.full(s.steps_per_day, np.nan), P[i]])[:s.lookback]
                        fut[n] = xf[i, :, j]
                inputs.append({"target": P[i], "past_covariates": past, "future_covariates": fut})
        with torch.autocast("cuda", enabled=False):
            out = self.fm.predict(inputs, prediction_length=s.horizon, batch_size=self.bs)
        q = torch.stack([o[0] for o in out]).permute(0, 2, 1)       # [B, Q, H] -> [B, H, Q]
        return self._finish(q, self.qlevels)


@register("timesfm")
class TimesFM(_FM):
    def __init__(self, spec, checkpoint=None, version=2.5, **kw):
        super().__init__(spec, **kw)
        import timesfm

        self.version = float(version)
        if self.version >= 3:
            fm = timesfm.TimesFM3Forecaster.from_pretrained(checkpoint or "google/timesfm-3.0-pytorch",
                                                            device="cuda" if torch.cuda.is_available() else "cpu")
        else:
            fm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(checkpoint or "google/timesfm-2.5-200m-pytorch")
            fm.compile(timesfm.ForecastConfig(max_context=max(512, spec.lookback), max_horizon=max(128, spec.horizon),
                                              normalize_inputs=True, use_continuous_quantile_head=True,
                                              force_flip_invariance=True, infer_is_positive=True,
                                              fix_quantile_crossing=True))
        object.__setattr__(self, "fm", fm)

    @torch.no_grad()
    def forward(self, batch):
        P = batch["x_hist"][..., self.spec.target_idx].float().cpu().numpy()
        with torch.autocast("cuda", enabled=False):
            if self.version >= 3:
                pts = np.stack([np.asarray(self.fm.predict(p, self.spec.horizon).point_forecast).reshape(-1)[:self.spec.horizon]
                                for p in P])
                return {"pred": torch.from_numpy(pts).float().to(self.dev)}
            point, quant = self.fm.forecast(horizon=self.spec.horizon, inputs=list(P))
        # quant: [B, H, 10] = mean + deciles 0.1..0.9
        q = torch.from_numpy(np.asarray(quant)[..., 1:])
        out = self._finish(q, [0.1 * i for i in range(1, 10)])
        if not self.probabilistic:
            out["pred"] = torch.from_numpy(np.asarray(point)).float().to(self.dev)
        return out


@register("tirex")
class TiRex(_FM):
    def __init__(self, spec, checkpoint="NX-AI/TiRex", **kw):
        super().__init__(spec, **kw)
        from tirex import load_model

        object.__setattr__(self, "fm", load_model(checkpoint, device="cuda" if torch.cuda.is_available() else "cpu"))

    @torch.no_grad()
    def forward(self, batch):
        P = batch["x_hist"][..., self.spec.target_idx].float()
        with torch.autocast("cuda", enabled=False):
            q, _mean = self.fm.forecast(context=P, prediction_length=self.spec.horizon)
        return self._finish(q, [0.1 * i for i in range(1, 10)])
