import torch
import torch.nn as nn

from .util import rand_uniform


class PassThroughWrapper(nn.Module):

    def __init__(self, cell):
        super(PassThroughWrapper, self).__init__()

        self.cell = cell

    def forward(self, x, ht_=None):
        ht = self.cell(x, ht_)

        return ht if not isinstance(ht, tuple) else ht[0], ht


class LinearProjectionWrapper(nn.Module):

    def __init__(self, cell, hidden_dim, output_dim, dtype):
        super(LinearProjectionWrapper, self).__init__()

        self.cell = cell

        self.w = nn.Parameter(nn.init.xavier_uniform_(torch.empty(output_dim, hidden_dim, dtype=dtype), gain=0.01))
        self.b = nn.Parameter(nn.Parameter(rand_uniform((output_dim,), -0.01, 0.01, dtype=dtype)))

    def forward(self, x, ht_=None):
        out, ht = self.cell(x, ht_)

        # Perform linear projection
        out = out @ self.w.T + self.b

        return out, ht


class ResidualWrapper(nn.Module):

    def __init__(self, cell):
        super(ResidualWrapper, self).__init__()

        self.cell = cell

    def forward(self, x, ht_=None):
        out, ht = self.cell(x, ht_)

        # Add res connection
        out += x if x.shape[1] == out.shape[1] else \
            torch.cat([x, torch.zeros((x.shape[0], out.shape[1] - x.shape[1])).to(x.device)], dim=1)

        return out, ht
