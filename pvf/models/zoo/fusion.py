"""Model fusion: build new architectures from registered ones, entirely from config.

    model:
      name: fusion
      mode: gate                 # mean | gate | stack
      deep_supervision: 0.3      # also train every branch on the target (weight 0.3)
      branches:
        - {name: itransformer, d_model: 64}
        - {name: powermamba, anchor: sp}

gate  : horizon-wise softmax weights over branches, conditioned on weather/clear-sky context ->
        "different models win in different regimes" (learned, differentiable precursor of the RL selector).
stack : small MLP over [branch predictions, x_fut] per horizon step.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..base import ForecastModel, build_model, register
from ..layers.common import mlp


@register("fusion")
class Fusion(ForecastModel):
    def __init__(self, spec, branches, mode="gate", deep_supervision=0.0, d_hidden=64, dropout=0.1, **kw):
        super().__init__(spec, **kw)
        self.branches = nn.ModuleList(build_model(dict(b), spec) for b in branches)
        if any(b.n_q for b in self.branches):
            raise ValueError("fusion currently combines point forecasters only")
        K, H = len(self.branches), spec.horizon
        self.mode, self.ds = mode, float(deep_supervision)
        d_ctx = spec.n_hist + H * spec.n_fut
        if mode == "gate":
            self.gate = mlp(d_ctx, d_hidden, H * K, 2, dropout)
        elif mode == "stack":
            self.stack = mlp(K + spec.n_fut, d_hidden, max(1, self.n_q), 2, dropout)
        elif mode != "mean":
            raise ValueError(mode)

    def forward(self, batch):
        outs = [br(batch) for br in self.branches]
        P = torch.stack([o["pred"] for o in outs], -1)                  # [b, H, K]
        if self.mode == "mean":
            y, w = P.mean(-1), None
        elif self.mode == "gate":
            ctx = torch.cat([batch["x_hist"][:, -1], batch["x_fut"].flatten(1)], 1)
            w = torch.softmax(self.gate(ctx).view(-1, self.H, P.shape[-1]), -1)
            y = (w * P).sum(-1)
        else:
            y, w = self.stack(torch.cat([P, batch["x_fut"]], -1)), None
            y = y if self.n_q else y.squeeze(-1)
        res = {"pred": y, "branch_preds": P.detach()}
        if w is not None:
            res["weights"] = w.detach()
        aux = [o["aux_loss"] for o in outs if "aux_loss" in o]
        if aux:
            res["aux_loss"] = sum(aux)
        if self.ds > 0 and self.training:
            res["deep_preds"] = [(o["pred"], self.ds) for o in outs]
        return res
