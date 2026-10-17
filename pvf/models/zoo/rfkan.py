"""RFKAN - Recurrent Fourier-KAN (Rong et al. 2025, Sci. Rep. -> ref/R16_RFKAN.pdf), re-implemented from Eqs. 3-8.

Layer l at time t:
    h_l(t)   = W_hh h_l(t-1) + W_hz x_l(t)                         (Eq. 3, recurrent kernel nodes)
    x_l+1(t) = sum_i phi_l,i( x_l,i(t), h_l,i(t) )                  (Eq. 4)
    phi(x)   = sum_k a_k cos(k x) + b_k sin(k x)                    (Eq. 8, Fourier series replaces splines)
Two layers, Kolmogorov-Arnold widths n -> 2n+1 -> out (paper Fig. 1).

Not reproduced: the paper's coarse->fine grid-refinement schedule (grid 3..100 every 50 steps); `grid` is a
fixed hyperparameter here. `tanh_state=True` (default off) bounds h for long lookbacks (paper uses L=12).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..base import ForecastModel, register
from ..layers.common import FourierKAN, HorizonHead


class RFKANLayer(nn.Module):
    def __init__(self, d_in, d_out, grid=8, tanh_state=False):
        super().__init__()
        self.W_hz = nn.Linear(d_in, d_in, bias=False)
        self.W_hh = nn.Linear(d_in, d_in, bias=False)
        nn.init.orthogonal_(self.W_hh.weight, gain=0.9)
        self.phi = FourierKAN(2 * d_in, d_out, grid)
        self.norm = nn.LayerNorm(d_out)
        self.tanh = tanh_state

    def forward(self, x):  # [b, L, d_in] -> [b, L, d_out]
        h = x.new_zeros(x.shape[0], x.shape[2])
        zx = self.W_hz(x)
        ys = []
        for t in range(x.shape[1]):
            h = self.W_hh(h) + zx[:, t]
            if self.tanh:
                h = torch.tanh(h)
            ys.append(self.phi(torch.cat([x[:, t], h], -1)))
        return self.norm(torch.stack(ys, 1))


@register("rfkan")
class RFKAN(ForecastModel):
    def __init__(self, spec, grid=8, n_layers=2, tanh_state=False, lookback_used=None, use_fut=True, dropout=0.1, **kw):
        super().__init__(spec, **kw)
        n = spec.n_hist
        self.Lu = lookback_used or spec.lookback          # paper used a short window (12); long L = slow loop
        widths = [n] + [2 * n + 1] * n_layers
        self.layers = nn.ModuleList(RFKANLayer(widths[i], widths[i + 1], grid, tanh_state) for i in range(n_layers))
        self.head = HorizonHead(widths[-1], spec.horizon, spec.n_fut, use_fut, n_quantiles=self.n_q, dropout=dropout)

    def forward(self, batch):
        x = batch["x_hist"][:, -self.Lu:]
        for lyr in self.layers:
            x = lyr(x)
        return {"pred": self.head(x[:, -1], batch["x_fut"])}
