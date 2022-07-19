from model.spectral_rnn.cgRNN.util import to_complex_activation

import torch
import torch.nn as nn

import math


class ComplexLinear(nn.Module):

    def __init__(self, dim_in, dim_out, fa=lambda x: x):
        super(ComplexLinear, self).__init__()

        stdv = 1. / math.sqrt(dim_out)
        self.w = nn.Parameter(nn.init.xavier_uniform_(torch.empty(dim_out, dim_in, dtype=torch.cfloat)))
        self.b = nn.Parameter(nn.init.uniform_(torch.empty(dim_out, dtype=torch.cfloat), -stdv, stdv))
        self.fa = fa

    def forward(self, x):
        return self.fa(x @ self.w.T + self.b)
