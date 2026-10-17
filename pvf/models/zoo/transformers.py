"""Transformer family from the selection report (rows 2-4).

patchtst     - Nie et al. 2023 (F01): channel-independent patches; head on target channel ("target") or
               on all channel representations ("mix", a cheap fix for the CI limitation noted in the report).
itransformer - Liu et al. 2024 (F02): every variate's whole series is one token; attention across variates.
               Future covariates (clear-sky, calendar, NWP) become extra variate tokens.
timexer      - Wang et al. 2024 (F03): endogenous P -> patch tokens + global token; exogenous series -> variate
               tokens; global token cross-attends to exogenous tokens.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..base import ForecastModel, register
from ..layers.common import HorizonHead, PatchEmbed, RevIN


def encoder(d_model, n_heads, n_layers, d_ff, dropout):
    layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, activation="gelu", batch_first=True,
                                       norm_first=True)
    return nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)


@register("patchtst")
class PatchTST(ForecastModel):
    def __init__(self, spec, patch_len=16, stride=8, d_model=128, n_heads=8, n_layers=3, d_ff=256, dropout=0.2,
                 channels="all", head="target", revin=False, use_fut=True, **kw):
        super().__init__(spec, **kw)
        self.ch = list(range(spec.n_hist)) if channels == "all" else [spec.hist_names.index(c) for c in channels]
        self.t_pos = self.ch.index(spec.target_idx)
        self.revin = RevIN(len(self.ch)) if revin else None
        self.embed = PatchEmbed(patch_len, stride, d_model, spec.lookback, dropout)
        self.enc = encoder(d_model, n_heads, n_layers, d_ff, dropout)
        self.head_mode = head
        d_flat = self.embed.n_patch * d_model
        d_z = d_flat if head == "target" else 256
        if head == "mix":
            self.squeeze = nn.Linear(d_flat, 256 // len(self.ch) + 1)
            d_z = (256 // len(self.ch) + 1) * len(self.ch)
        self.head = HorizonHead(d_z, spec.horizon, spec.n_fut, use_fut, n_quantiles=self.n_q, dropout=dropout)

    def forward(self, batch):
        x = batch["x_hist"][..., self.ch]
        if self.revin is not None:
            x = self.revin.norm(x)
        tok, c = self.embed(x)
        z = self.enc(tok).flatten(1).view(x.shape[0], c, -1)        # [b, C, n_patch*d]
        z = z[:, self.t_pos] if self.head_mode == "target" else self.squeeze(z).flatten(1)
        y = self.head(z, batch["x_fut"])
        if self.revin is not None and not self.n_q:
            y = self.revin.denorm(y, self.t_pos)
        return {"pred": y}


@register("itransformer")
class ITransformer(ForecastModel):
    def __init__(self, spec, d_model=128, n_heads=8, n_layers=3, d_ff=256, dropout=0.1, use_fut_tokens=True, **kw):
        super().__init__(spec, **kw)
        self.emb_h = nn.Sequential(nn.Linear(spec.lookback, d_model), nn.Dropout(dropout))
        self.emb_f = nn.Sequential(nn.Linear(spec.horizon, d_model), nn.Dropout(dropout)) if use_fut_tokens else None
        self.var_id = nn.Parameter(torch.randn(1, spec.n_hist + spec.n_fut, d_model) * 0.02)
        self.enc = encoder(d_model, n_heads, n_layers, d_ff, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, spec.horizon * max(1, self.n_q))

    def forward(self, batch):
        tok = self.emb_h(batch["x_hist"].transpose(1, 2))           # [b, C_h, d]
        if self.emb_f is not None:
            tok = torch.cat([tok, self.emb_f(batch["x_fut"].transpose(1, 2))], 1)
        tok = tok + self.var_id[:, :tok.shape[1]]
        z = self.norm(self.enc(tok))[:, self.spec.target_idx]
        y = self.proj(z).view(z.shape[0], self.H, -1)
        return {"pred": y if self.n_q else y.squeeze(-1)}


class _TimeXerLayer(nn.Module):
    def __init__(self, d, h, d_ff, p):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, h, p, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, h, p, batch_first=True)
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(p), nn.Linear(d_ff, d))
        self.drop = nn.Dropout(p)

    def forward(self, en, ex):  # en: [b, n_patch+1, d] (last = global), ex: [b, C_ex, d]
        h = self.n1(en)
        en = en + self.drop(self.self_attn(h, h, h, need_weights=False)[0])
        g = self.n2(en[:, -1:])
        g = en[:, -1:] + self.drop(self.cross_attn(g, ex, ex, need_weights=False)[0])
        en = torch.cat([en[:, :-1], g], 1)
        return en + self.drop(self.ff(self.n3(en)))


@register("timexer")
class TimeXer(ForecastModel):
    def __init__(self, spec, patch_len=16, d_model=128, n_heads=8, n_layers=2, d_ff=256, dropout=0.1, **kw):
        super().__init__(spec, **kw)
        assert spec.lookback % patch_len == 0, "TimeXer uses non-overlapping patches: lookback % patch_len == 0"
        self.p = patch_len
        n_patch = spec.lookback // patch_len
        self.patch = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, n_patch, d_model) * 0.02)
        self.glob = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.ex_idx = [i for i in range(spec.n_hist) if i != spec.target_idx]
        self.emb_ex_h = nn.Linear(spec.lookback, d_model)
        self.emb_ex_f = nn.Linear(spec.horizon, d_model)
        self.layers = nn.ModuleList(_TimeXerLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear((n_patch + 1) * d_model, spec.horizon * max(1, self.n_q))

    def forward(self, batch):
        xh = batch["x_hist"]
        b = xh.shape[0]
        en = self.patch(xh[..., self.spec.target_idx].unfold(-1, self.p, self.p)) + self.pos
        en = torch.cat([self.drop(en), self.glob.expand(b, -1, -1)], 1)
        ex = torch.cat([self.emb_ex_h(xh[..., self.ex_idx].transpose(1, 2)),
                        self.emb_ex_f(batch["x_fut"].transpose(1, 2))], 1)
        for lyr in self.layers:
            en = lyr(en, ex)
        y = self.proj(self.norm(en).flatten(1)).view(b, self.H, -1)
        return {"pred": y if self.n_q else y.squeeze(-1)}
