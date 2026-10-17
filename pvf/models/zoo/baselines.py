"""Reference / incumbent models.

smart_persistence - clear-sky-scaled persistence, the standard PV reference. Every model must beat it.
mlp, lstm, gru, tcn - classic DL controls (selection report row 1: new architectures must beat these).
dlinear           - "are transformers even needed?" sanity baseline (ref/foundational/F19).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import ForecastModel, register
from ..layers.common import HorizonHead, SeriesDecomp, mlp


@register("smart_persistence")
class SmartPersistence(ForecastModel):
    trainable = False

    def __init__(self, spec, **kw):
        super().__init__(spec, **kw)
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, batch):
        return {"pred": batch["sp"]}


@register("mlp")
class DeepMLP(ForecastModel):
    """Deep MLP: flattened lags -> 256-128-64 -> output (here all H horizons at once)."""

    def __init__(self, spec, hidden=(256, 128, 64), dropout=0.2, use_fut=True, **kw):
        super().__init__(spec, **kw)
        d_in = spec.lookback * spec.n_hist + (spec.horizon * spec.n_fut if use_fut else 0)
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, spec.horizon * max(1, self.n_q)))
        self.net, self.use_fut = nn.Sequential(*layers), use_fut

    def forward(self, batch):
        x = batch["x_hist"].flatten(1)
        if self.use_fut:
            x = torch.cat([x, batch["x_fut"].flatten(1)], 1)
        y = self.net(x).view(x.shape[0], self.H, -1)
        return {"pred": y if self.n_q else y.squeeze(-1)}


class _RNN(ForecastModel):
    cell = nn.LSTM

    def __init__(self, spec, hidden=64, layers=2, dropout=0.1, use_fut=True, **kw):
        super().__init__(spec, **kw)
        self.rnn = self.cell(spec.n_hist, hidden, layers, batch_first=True, dropout=dropout)
        self.head = HorizonHead(hidden, spec.horizon, spec.n_fut, use_fut, n_quantiles=self.n_q, dropout=dropout)

    def forward(self, batch):
        h, _ = self.rnn(batch["x_hist"])
        return {"pred": self.head(h[:, -1], batch["x_fut"])}


@register("lstm")
class LSTM(_RNN):
    cell = nn.LSTM


@register("gru")
class GRU(_RNN):
    cell = nn.GRU


class _CausalBlock(nn.Module):
    def __init__(self, c_in, c_out, k, dil, dropout):
        super().__init__()
        self.pad = (k - 1) * dil
        self.c1 = nn.utils.parametrizations.weight_norm(nn.Conv1d(c_in, c_out, k, dilation=dil))
        self.c2 = nn.utils.parametrizations.weight_norm(nn.Conv1d(c_out, c_out, k, dilation=dil))
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        y = self.drop(F.relu(self.c1(F.pad(x, (self.pad, 0)))))
        y = self.drop(F.relu(self.c2(F.pad(y, (self.pad, 0)))))
        return F.relu(y + self.res(x))


@register("tcn")
class TCN(ForecastModel):
    """TCN (Bai et al. 2018): 4 causal blocks, dilations 1/2/4/8, 32 channels, kernel 3 (receptive field 61)."""

    def __init__(self, spec, channels=32, dilations=(1, 2, 4, 8), kernel=3, dropout=0.1, use_fut=True, **kw):
        super().__init__(spec, **kw)
        blocks, c = [], spec.n_hist
        for d in dilations:
            blocks.append(_CausalBlock(c, channels, kernel, d, dropout))
            c = channels
        self.net = nn.Sequential(*blocks)
        self.head = HorizonHead(channels, spec.horizon, spec.n_fut, use_fut, n_quantiles=self.n_q, dropout=dropout)

    def forward(self, batch):
        h = self.net(batch["x_hist"].transpose(1, 2))
        return {"pred": self.head(h[..., -1], batch["x_fut"])}


@register("dlinear")
class DLinear(ForecastModel):
    """Trend/seasonal linear maps on the target history (+ optional linear future-covariate term)."""

    def __init__(self, spec, kernel=25, use_fut=True, **kw):
        super().__init__(spec, **kw)
        self.decomp = SeriesDecomp(kernel)
        self.lin_s = nn.Linear(spec.lookback, spec.horizon)
        self.lin_t = nn.Linear(spec.lookback, spec.horizon)
        self.fut = nn.Linear(spec.n_fut, 1) if use_fut else None

    def forward(self, batch):
        x = batch["x_hist"][..., self.spec.target_idx:self.spec.target_idx + 1]
        s, t = self.decomp(x)
        y = self.lin_s(s[..., 0]) + self.lin_t(t[..., 0])
        if self.fut is not None:
            y = y + self.fut(batch["x_fut"]).squeeze(-1)
        return {"pred": y}
