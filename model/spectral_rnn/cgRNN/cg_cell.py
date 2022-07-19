import math

import torch
import torch.nn as nn

from ..manifold_optimization import ManifoldParameter, Manifold
from .util import mod_relu_act, unitary_init


class CGCell(nn.Module):

    def __init__(self, input_size, hidden_size, alpha=0.5, beta=0.5):
        super(CGCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.wg = nn.Parameter(nn.init.xavier_uniform_(torch.empty(2 * hidden_size, hidden_size, dtype=torch.cfloat), gain=0.01))
        self.vg = nn.Parameter(nn.init.xavier_uniform_(torch.empty(2 * hidden_size, input_size, dtype=torch.cfloat), gain=0.01))
        self.bg = nn.Parameter(torch.full((2 * hidden_size,), 4, dtype=torch.cfloat))  # Init to 4 --> Fully open gates

        self.w = ManifoldParameter(unitary_init((hidden_size, hidden_size)), manifold=Manifold.STIEFEL)
        # self.w = nn.Parameter(nn.init.xavier_uniform_(torch.empty(hidden_size, hidden_size, dtype=torch.cfloat), gain=0.01))
        self.v = nn.Parameter(nn.init.xavier_uniform_(torch.empty(hidden_size, input_size, dtype=torch.cfloat), gain=0.01))
        self.b = nn.Parameter(torch.zeros((hidden_size,), dtype=torch.cfloat))  # Init to 0

        self.fg = lambda x: torch.sigmoid(alpha * x.real + beta * x.imag)

        self.fa_offset = nn.Parameter(nn.init.uniform_(torch.empty(1, dtype=torch.float), -0.01, 0.01))
        self.fa = mod_relu_act(self.fa_offset)

    @torch.no_grad()
    def _init_hidden_zero(self, x):
        """
        Inits hidden full with zeros.
        When RNNLayer.learn_hidden_init = True, this method wont be called.
        """

        h = torch.zeros((x.shape[0], self.hidden_size), dtype=torch.cfloat)

        return h

    def forward(self, x, ht_=None):
        if ht_ is None:
            ht_ = self._init_hidden_zero(x).to(x.device)

        # Should be the same as double_memory_gate with real=False in SpectralRNN Repo
        gates = ht_ @ self.wg.T + x @ self.vg.T + self.bg
        g_r, g_z = gates.chunk(2, 1)

        g_r = self.fg(g_r)
        g_z = self.fg(g_z)

        z = (g_r * ht_) @ self.w.T + x @ self.v.T + self.b
        ht = g_z * self.fa(z) + (1 - g_z) * ht_

        return ht
