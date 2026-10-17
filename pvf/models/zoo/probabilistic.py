"""Probabilistic + domain-adaptive models (selection report rows 13-14).

mtsa_qrn - Wang et al. 2026 (Sustainability 18(2):739 -> ref/R08_MTSA-QRN_CQR.pdf), Sec. 3.1-3.4:
           temporal pathway  : multi-scale dilated causal convs -> softmax scale gating (Eq. 9-10) -> MHSA refinement
           meteorological    : GAT over variable nodes (each node = its embedded series)
           weather-adaptive  : fusion weights from a clearness-index regime descriptor (sunny/cloudy/rainy ~ k_t stats)
           quantile head     : non-crossing quantiles (base + cumulative softplus increments); CQR in pvf.eval.conformal
dt_dann  - Zhao 2024 (Sci. Rep. -> ref/R18_DigitalTwin_Transformer_PV.pdf): causal-CNN feature generator (replaces
           positional embedding) -> Transformer -> PV predictor; domain classifier behind a gradient-reversal layer.
           Source = labelled training data ("digital space"), target = unlabelled batches from `data.adapt_months`
           or another site ("physical space"); the trainer supplies them as batch["tgt"].
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import ForecastModel, register
from ..layers.common import HorizonHead, grad_reverse, mlp
from ..layers.graph import DenseGAT

DEFAULT_Q = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


class MonotoneQuantileHead(nn.Module):
    """[b, d] (+x_fut) -> [b, H, Q] with q_1 <= q_2 <= ... (no quantile crossing)."""

    def __init__(self, d_in, horizon, n_fut, n_q, use_fut=True, dropout=0.1):
        super().__init__()
        self.inner = HorizonHead(d_in, horizon, n_fut, use_fut, n_quantiles=n_q, dropout=dropout)

    def forward(self, z, x_fut):
        raw = self.inner(z, x_fut)
        return torch.cat([raw[..., :1], raw[..., :1] + torch.cumsum(F.softplus(raw[..., 1:]), -1)], -1)


@register("mtsa_qrn")
class MTSAQRN(ForecastModel):
    def __init__(self, spec, d_model=64, dilations=(1, 2, 4, 8), kernel=3, n_heads=4, gat_heads=4, dropout=0.1,
                 quantiles=DEFAULT_Q, use_fut=True, **kw):
        super().__init__(spec, quantiles=quantiles, **kw)
        self.inp = nn.Linear(spec.n_hist, d_model)
        self.k = kernel
        self.scales = nn.ModuleList(nn.Conv1d(d_model, d_model, kernel, dilation=d) for d in dilations)
        self.dil = dilations
        self.score = nn.Linear(d_model, 1)                                   # f(.) in Eq. 9
        self.refine = nn.TransformerEncoderLayer(d_model, n_heads, 2 * d_model, dropout, batch_first=True, norm_first=True)
        self.met_idx = [spec.hist_names.index(v) for v in ("P", "G", "T", "Gcs")]
        self.node_emb = nn.Linear(spec.lookback, d_model)
        self.gat = DenseGAT(d_model, d_model, gat_heads, dropout)
        self.iG, self.iGcs = spec.hist_names.index("G"), spec.hist_names.index("Gcs")
        self.regime = mlp(3, 16, 2, 2, 0.0)                                  # weather-adaptive fusion weights
        if self.n_q:
            self.head = MonotoneQuantileHead(d_model, spec.horizon, spec.n_fut, self.n_q, use_fut, dropout)
        else:
            self.head = HorizonHead(d_model, spec.horizon, spec.n_fut, use_fut, dropout=dropout)

    def forward(self, batch):
        xh = batch["x_hist"]
        b = xh.shape[0]
        h = self.inp(xh).transpose(1, 2)                                     # [b, d, L]
        Hs = [conv(F.pad(h, ((self.k - 1) * d, 0))).transpose(1, 2) for conv, d in zip(self.scales, self.dil)]
        Hs = torch.stack([F.gelu(x) for x in Hs], 1)                         # [b, S, L, d]
        alpha = torch.softmax(self.score(Hs.mean(2)), 1)                     # [b, S, 1]   Eq. 9
        H = (alpha.unsqueeze(-1) * Hs).sum(1)                                # Eq. 10
        z_t = self.refine(H)[:, -1]
        nodes = self.node_emb(xh[..., self.met_idx].transpose(1, 2))         # [b, 4, d]
        A = torch.ones(b, len(self.met_idx), len(self.met_idx), device=xh.device)
        z_s = (nodes + self.gat(nodes, A))[:, 0]                             # P node
        g, gcs = xh[:, -self.spec.steps_per_day:, self.iG], xh[:, -self.spec.steps_per_day:, self.iGcs]
        day = (gcs > 0.01).float()
        kt = (g / gcs.clamp_min(0.01)).clamp(0, 1.5) * day
        n = day.sum(1).clamp_min(1)
        mu = kt.sum(1) / n
        sd = torch.sqrt(((kt - mu[:, None]) ** 2 * day).sum(1) / n)
        w = torch.softmax(self.regime(torch.stack([mu, sd, kt[:, -1]], 1)), -1)
        z = w[:, :1] * z_t + w[:, 1:] * z_s
        return {"pred": self.head(z, batch["x_fut"]), "scale_w": alpha.detach(), "fusion_w": w.detach()}


class _CausalConvGen(nn.Module):
    def __init__(self, c_in, d, n_layers=2, k=3):
        super().__init__()
        self.k = k
        self.convs = nn.ModuleList(nn.Conv1d(c_in if i == 0 else d, d, k) for i in range(n_layers))

    def forward(self, x):  # [b, L, c] -> [b, L, d]
        h = x.transpose(1, 2)
        for c in self.convs:
            h = F.gelu(c(F.pad(h, (self.k - 1, 0))))
        return h.transpose(1, 2)


@register("dt_dann")
class DTTransformerDANN(ForecastModel):
    def __init__(self, spec, d_model=64, n_heads=4, n_layers=2, d_ff=128, dropout=0.1, dann_weight=0.1,
                 use_fut=True, **kw):
        super().__init__(spec, **kw)
        self.gen = _CausalConvGen(spec.n_hist, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = HorizonHead(d_model, spec.horizon, spec.n_fut, use_fut, n_quantiles=self.n_q, dropout=dropout)
        self.domain = mlp(d_model, d_model, 1, 2, dropout)
        self.w, self.lam = dann_weight, 0.0

    def on_epoch_start(self, epoch: int, max_epochs: int):
        p = epoch / max(1, max_epochs)
        self.lam = 2.0 / (1.0 + math.exp(-10 * p)) - 1.0                     # Ganin et al. schedule

    def features(self, x_hist):
        return self.enc(self.gen(x_hist))[:, -1]

    def forward(self, batch):
        f = self.features(batch["x_hist"])
        out = {"pred": self.head(f, batch["x_fut"])}
        if self.training and "tgt" in batch and self.w > 0:
            ft = self.features(batch["tgt"]["x_hist"])
            logits = self.domain(grad_reverse(torch.cat([f, ft]), self.lam)).squeeze(-1)
            lbl = torch.cat([torch.zeros(len(f)), torch.ones(len(ft))]).to(logits)
            out["aux_loss"] = self.w * F.binary_cross_entropy_with_logits(logits, lbl)
        return out
