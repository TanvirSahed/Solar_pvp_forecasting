"""Evidence-driven fusions: PatchTST plus one partner chosen for a weakness measured on the SSP batch
(analysis and results: tools/fusion_study.py and tools/nwp_study.py; the reports are shared separately).

Every fusion is a weather-regime gate over two complete forecasters, trained end to end:

    forecast_h = g_h * patch_tst_h + (1 - g_h) * partner_h,     g_h in [0, 1] for every lead time h

g is predicted from the recent weather regime (last-day clearness mean / std / latest, SummariseWeatherRegime)
and the latest observed history step (power, irradiance, clear sky, time of day), so the mix can change
between cloudy and clear spells and between short and long lead times. Each branch is also trained on its own
(branch_loss_weight), so the gate cannot starve one branch. The output "gate" shows the share given to PatchTST.
"""
from __future__ import annotations

import torch

from ...model_inputs.adapter_steps import SummariseWeatherRegime
from ...model_inputs.input_adapter import InputAdapter
from ..base import ForecastModel, register_model
from ..blocks.common_blocks import make_mlp
from .probabilistic import DigitalTwinTransformerDANN
from .state_space import MambaTemporal
from .transformers import ITransformer, PatchTST

# PatchTST as trained by R7 (its paper recipe), shared by every fusion and by the PatchTST control run
PATCH_TST_OPTIONS = {"model_size": 128, "n_heads": 4, "n_layers": 3, "patch_length": 16, "patch_stride": 8,
                     "dropout": 0.1}


class _RegimeGatedFusion(ForecastModel):
    """gate_reads_future=True: the gate is computed for every lead time from the context AND that step's known-future
    features (clear sky, calendar, and NWP when selected), so a forecast of the coming sky can steer the mix."""

    def __init__(self, data_spec, patch_tst, partner, gate_hidden_size=32, branch_loss_weight=0.3,
                 gate_reads_future=False):
        super().__init__(data_spec, InputAdapter([SummariseWeatherRegime(data_spec)], type(self).__name__))
        self.patch_tst, self.partner = patch_tst, partner
        context_size = 3 + data_spec.n_history_features
        self.gate_reads_future = gate_reads_future
        if gate_reads_future:                                    # one shared MLP applied at every lead time
            self.gate_network = make_mlp(context_size + data_spec.n_future_features, gate_hidden_size, 1,
                                         n_layers=2, dropout=0.1)
        else:                                                    # one MLP that outputs all lead times at once
            self.gate_network = make_mlp(context_size, gate_hidden_size, data_spec.horizon_steps, n_layers=2,
                                         dropout=0.1)
        self.branch_loss_weight = branch_loss_weight

    def _gate_logits(self, model_input):
        context = torch.cat([model_input["weather_regime"], model_input["history_features"][:, -1]], -1)
        if not self.gate_reads_future:
            return self.gate_network(context)
        future = model_input["future_features"]                                   # [batch, horizon, n_future]
        per_step = torch.cat([context.unsqueeze(1).expand(-1, future.shape[1], -1), future], -1)
        return self.gate_network(per_step).squeeze(-1)

    def forecast(self, model_input):
        patch_forecast = self.patch_tst.forecast(model_input)["forecast"]       # [batch, horizon]
        partner_forecast = self.partner.forecast(model_input)["forecast"]
        gate = torch.sigmoid(self._gate_logits(model_input))                     # share given to PatchTST
        output = {"forecast": gate * patch_forecast + (1 - gate) * partner_forecast, "gate": gate.detach()}
        if self.training and self.branch_loss_weight:
            target, mask = model_input["target"], model_input["target_mask"]
            branch_errors = [((f - target) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)
                             for f in (patch_forecast, partner_forecast)]
            output["auxiliary_loss"] = self.branch_loss_weight * sum(branch_errors)
        return output


@register_model("patch_tst_conv_gate")
class PatchTSTConvTransformerGate(_RegimeGatedFusion):
    """Weakness: PatchTST is mid-table on clear-sky windows and under-forecasts slightly.
    Borrowed: DT-DANN's causal-CNN + transformer encoder, among the best on clear and mixed windows (but weak on
    overcast ones, so the gate has to learn when to trust it). Domain adaptation is off: there is no target domain."""

    def __init__(self, data_spec, patch_tst_options=None, partner_options=None, **gate_options):
        partner_options = {"model_size": 64, **(partner_options or {}), "domain_loss_weight": 0.0}
        super().__init__(data_spec, PatchTST(data_spec, **{**PATCH_TST_OPTIONS, **(patch_tst_options or {})}),
                         DigitalTwinTransformerDANN(data_spec, **partner_options), **gate_options)


@register_model("patch_tst_mamba_gate")
class PatchTSTMambaGate(_RegimeGatedFusion):
    """Weakness: PatchTST's 8 h patches blur fast changes; it is weaker in the afternoon (15-18 h) and on ramps.
    Borrowed: the Mamba recurrent state of mamba_temporal / lstm, among the best at the morning and evening ramps
    and on clear windows."""

    def __init__(self, data_spec, patch_tst_options=None, partner_options=None, **gate_options):
        partner_options = {"model_size": 64, "n_layers": 2, "state_size": 16, **(partner_options or {})}
        super().__init__(data_spec, PatchTST(data_spec, **{**PATCH_TST_OPTIONS, **(patch_tst_options or {})}),
                         MambaTemporal(data_spec, **partner_options), **gate_options)


@register_model("patch_tst_variable_gate")
class PatchTSTVariableTokenGate(_RegimeGatedFusion):
    """Weakness: PatchTST is channel-independent - each variable is encoded alone, and with head_reads="target" only
    the power channel reaches the head, so past irradiance and temperature never affect its forecast (the
    limitation flagged in the selection report).
    Borrowed: iTransformer's variable tokens (attention ACROSS variables); its errors are the least correlated
    with PatchTST's and it is strong at 2.5-8 h ahead."""

    def __init__(self, data_spec, patch_tst_options=None, partner_options=None, **gate_options):
        partner_options = {"model_size": 128, "n_heads": 8, "n_layers": 2, "feedforward_size": 256,
                           **(partner_options or {})}
        super().__init__(data_spec, PatchTST(data_spec, **{**PATCH_TST_OPTIONS, **(patch_tst_options or {})}),
                         ITransformer(data_spec, **partner_options), **gate_options)
