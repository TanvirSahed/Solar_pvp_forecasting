"""State-space (Mamba) family from the selection report (rows 5-6, and the "Concat-Mamba" ablation root).

mamba       - temporal Mamba stack over time steps, all channels concatenated per step ("Concat-Mamba").
smamba      - S-Mamba (F06): variate tokens (iTransformer-style) mixed by a bidirectional Mamba.
powermamba  - PowerMamba (R19, Sec. IV): RevIN -> trend/seasonal decomposition -> linear time projection to E
              -> parallel Mamba (temporal tokens) + inverse Mamba (variate tokens)
              -> concat[x_e, x_m, x_im, x_m + x_im] -> linear to horizon.
              External predictions (Sec. IV-E) are integrated by appending the known-future covariates to the
              sequence (length L+H), zero-filled for channels that have no forecast.
hdcf_mamba  - see hdcf.py (PV-specific, R6).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..base import ForecastModel, register
from ..layers.common import HorizonHead, RevIN, SeriesDecomp
from ..layers.mamba import MambaLayer


@register("mamba")
class MambaTemporal(ForecastModel):
    def __init__(self, spec, d_model=64, n_layers=3, d_state=16, expand=2, dropout=0.1, pool="last",
                 use_fut=True, backend="auto", **kw):
        super().__init__(spec, **kw)
        self.inp = nn.Linear(spec.n_hist, d_model)
        self.layers = nn.Sequential(*[MambaLayer(d_model, d_state, expand=expand, dropout=dropout, backend=backend)
                                      for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.pool = pool
        self.head = HorizonHead(d_model, spec.horizon, spec.n_fut, use_fut, n_quantiles=self.n_q, dropout=dropout)

    def encode(self, x_hist):  # reusable as a temporal block inside hybrids (graph_mamba, fusion ...)
        h = self.norm(self.layers(self.inp(x_hist)))
        return h[:, -1] if self.pool == "last" else h.mean(1)

    def forward(self, batch):
        return {"pred": self.head(self.encode(batch["x_hist"]), batch["x_fut"])}


@register("smamba")
class SMamba(ForecastModel):
    def __init__(self, spec, d_model=128, n_layers=2, d_state=16, dropout=0.1, backend="auto", **kw):
        super().__init__(spec, **kw)
        self.emb_h = nn.Linear(spec.lookback, d_model)
        self.emb_f = nn.Linear(spec.horizon, d_model)
        self.layers = nn.Sequential(*[MambaLayer(d_model, d_state, dropout=dropout, bidirectional=True, ffn=True,
                                                 backend=backend) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, spec.horizon * max(1, self.n_q))

    def forward(self, batch):
        tok = torch.cat([self.emb_h(batch["x_hist"].transpose(1, 2)), self.emb_f(batch["x_fut"].transpose(1, 2))], 1)
        z = self.norm(self.layers(tok))[:, self.spec.target_idx]
        y = self.proj(z).view(z.shape[0], self.H, -1)
        return {"pred": y if self.n_q else y.squeeze(-1)}


@register("powermamba")
class PowerMamba(ForecastModel):
    def __init__(self, spec, embed_len=64, d_state=16, kernel=25, dropout=0.1, revin=True, external=True,
                 backend="auto", **kw):
        super().__init__(spec, **kw)
        self.external = external
        C = spec.n_hist + (spec.n_fut if external else 0)
        T = spec.lookback + (spec.horizon if external else 0)
        self.C, self.T = C, T
        self.revin = RevIN(C) if revin else None
        self.decomp = SeriesDecomp(kernel)
        self.lin_E = nn.Linear(2 * T, embed_len)                     # [T(x); S(x)] (2T) -> E, per channel
        self.drop1, self.drop2 = nn.Dropout(dropout), nn.Dropout(dropout)
        self.mamba = MambaLayer(C, d_state, expand=2, dropout=0.0, backend=backend)        # tokens = E steps
        self.imamba = MambaLayer(embed_len, d_state, expand=2, dropout=0.0, backend=backend)  # tokens = C variates
        self.lin_W = nn.Linear(4 * embed_len, spec.horizon * max(1, self.n_q))

    def forward(self, batch):
        xh, xf = batch["x_hist"], batch["x_fut"]
        b = xh.shape[0]
        if self.external:  # append known future; unknown channels get 0 for future, known get 0 for the past
            x = torch.cat([torch.cat([xh, xh.new_zeros(b, self.H, xh.shape[2])], 1),
                           torch.cat([xh.new_zeros(b, self.L, xf.shape[2]), xf], 1)], 2)
        else:
            x = xh
        if self.revin is not None:
            x = self.revin.norm(x)
        s, t = self.decomp(x)
        xe = self.lin_E(torch.cat([t, s], 1).transpose(1, 2)).transpose(1, 2)   # [b, E, C]
        xm = self.mamba(self.drop1(xe))                                          # temporal tokens
        xim = self.imamba(self.drop2(xe.transpose(1, 2))).transpose(1, 2)        # variate tokens
        xc = torch.cat([xe, xm, xim, xm + xim], 1)                               # [b, 4E, C]
        y = self.lin_W(xc[..., self.spec.target_idx]).view(b, self.H, -1)
        if self.n_q:
            return {"pred": y}
        y = y.squeeze(-1)
        if self.revin is not None:
            y = self.revin.denorm(y, self.spec.target_idx)
        return {"pred": y}
