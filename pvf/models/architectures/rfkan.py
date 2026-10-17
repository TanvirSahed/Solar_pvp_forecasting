"""RFKAN - Recurrent Fourier-KAN (Rong et al. 2025, Sci. Rep. -> ref/R16_RFKAN.pdf), from Eqs. 3-8.

Layer l at time t:
    memory_l(t)   = W_hh memory_l(t-1) + W_hz x_l(t)                 (Eq. 3, recurrent kernel nodes)
    x_l+1(t)      = FourierKAN([x_l(t), memory_l(t)])                (Eq. 4 and 8, Fourier series replace splines)
Two layers with Kolmogorov-Arnold widths n -> 2n+1 -> 2n+1, last step -> forecast head.

Not reproduced: the paper's coarse-to-fine grid refinement schedule; n_harmonics is fixed. The paper used a
12-step window - long windows make the Python time loop slow, hence recurrent_window_steps.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..base import ForecastModel, register_model
from ..blocks.common_blocks import FourierKANLayer, ForecastHead


class RecurrentFourierKANLayer(nn.Module):
    def __init__(self, input_size, output_size, n_harmonics=8, bound_memory=False):
        super().__init__()
        self.input_to_memory = nn.Linear(input_size, input_size, bias=False)
        self.memory_to_memory = nn.Linear(input_size, input_size, bias=False)
        nn.init.orthogonal_(self.memory_to_memory.weight, gain=0.9)
        self.fourier_kan = FourierKANLayer(2 * input_size, output_size, n_harmonics)
        self.norm = nn.LayerNorm(output_size)
        self.bound_memory = bound_memory

    def forward(self, sequence):                                       # [batch, time, input] -> [batch, time, output]
        memory = sequence.new_zeros(sequence.shape[0], sequence.shape[2])
        memory_inputs = self.input_to_memory(sequence)
        outputs = []
        for t in range(sequence.shape[1]):
            memory = self.memory_to_memory(memory) + memory_inputs[:, t]
            if self.bound_memory:
                memory = torch.tanh(memory)
            outputs.append(self.fourier_kan(torch.cat([sequence[:, t], memory], -1)))
        return self.norm(torch.stack(outputs, 1))


@register_model("rfkan")
class RFKAN(ForecastModel):
    def __init__(self, data_spec, n_harmonics=8, n_layers=2, recurrent_window_steps=48, bound_memory=False,
                 dropout=0.1, use_future_features=True, quantile_levels=None, input_adapter=None):
        super().__init__(data_spec, input_adapter, quantile_levels)
        self.recurrent_window_steps = recurrent_window_steps
        widths = [data_spec.n_history_features] + [2 * data_spec.n_history_features + 1] * n_layers
        self.kan_layers = nn.ModuleList(RecurrentFourierKANLayer(widths[i], widths[i + 1], n_harmonics, bound_memory)
                                        for i in range(n_layers))
        self.forecast_head = ForecastHead(widths[-1], data_spec.horizon_steps, data_spec.n_future_features,
                                          use_future_features, n_outputs_per_step=self.n_outputs_per_step,
                                          dropout=dropout)

    def encode(self, model_input):
        sequence = model_input["history_features"][:, -self.recurrent_window_steps:]
        for layer in self.kan_layers:
            sequence = layer(sequence)
        return sequence[:, -1]

    def forecast(self, model_input):
        representation = self.encode(model_input)
        return {"forecast": self.shape_output(self.forecast_head(representation, model_input["future_features"]))}
