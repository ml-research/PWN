import math

import torch
import torch.nn as nn

from .util import rand_uniform
from .wrapper import PassThroughWrapper, LinearProjectionWrapper, ResidualWrapper
from ..spectral_rnn_config import SpectralRNNConfig


# Use GPU if avaiable
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class RNNLayer(nn.Module):
    def __init__(self, cell_type, config: SpectralRNNConfig):
        super().__init__()

        input_size = config.input_dim
        hidden_dim = config.hidden_dim
        output_dim = config.compressed_value_dim

        if not config.rnn_layer_config.use_cg_cell:
            input_size *= 2
            hidden_dim *= 2
            output_dim *= 2

        dtype = torch.cfloat if config.rnn_layer_config.use_cg_cell else torch.float

        self.h0 = []
        self.cells = nn.ModuleList()
        for i in range(config.rnn_layer_config.n_layers):
            cell = PassThroughWrapper(cell_type(input_size if i == 0 else hidden_dim, hidden_dim))

            if config.rnn_layer_config.use_linear_projection and i == config.rnn_layer_config.n_layers - 1:
                cell = LinearProjectionWrapper(cell, hidden_dim, output_dim, dtype)

            self.cells.append(cell)

            # Init according to uRNN paper - ensures E[||h0||^2]=1, i.e. "unitary"
            if config.rnn_layer_config.learn_hidden_init:

                boundary = math.sqrt(3 / (2 * hidden_dim))
                h0 = nn.Parameter(rand_uniform((hidden_dim,), -boundary, boundary, dtype=dtype))

                if self.is_gated:
                    self.h0.append(h0)
                else:
                    self.h0.append((h0, nn.Parameter(rand_uniform((hidden_dim,), -boundary, boundary, dtype=dtype))))

        self.config = config

        if self.config.rnn_layer_config.dropout is not None:
            self.dropout = nn.Dropout(p=self.config.rnn_layer_config.dropout)
        else:
            self.dropout = None

    def init_hidden(self, batch_size):
        if self.config.rnn_layer_config.learn_hidden_init:
            return [h0.unsqueeze(0).repeat(batch_size, 1).to(device) for h0 in self.h0] \
                if self.config.rnn_layer_config.is_gated \
                else [tuple((h0_.unsqueeze(0).repeat(batch_size, 1).to(device) for h0_ in h0)) for h0 in self.h0]
        else:
            return [None] * self.config.rnn_layer_config.n_layers

    def forward(self, input_, h_init=None):
        layer_hs = [[] for _ in range(self.config.rnn_layer_config.n_layers)]
        layer_outs = [[] for _ in range(self.config.rnn_layer_config.n_layers)]
        h0 = h_init if h_init is not None else self.init_hidden(input_.shape[0])
        for x_ in torch.unbind(input_, dim=1):
            # If we only get zeros as input, we are only predicting (train & test) --> take earlier output as input
            if len(layer_outs[-1]) > 0 and (
                    (not self.config.rnn_layer_config.use_cg_cell and len(torch.nonzero(x)) == 0) or (
                    self.config.rnn_layer_config.use_cg_cell and
                    len(torch.nonzero(x.real)) == 0 and len(torch.nonzero(x.imag)) == 0)):
                x_ = layer_outs[-1][-1]

            x = x_
            for i, cell in enumerate(self.cells):
                if len(layer_hs[i]) > 0:
                    h = layer_hs[i][-1]
                else:
                    h = h0[i]

                x, h = cell(x, h)

                if i == self.config.rnn_layer_config.n_layers - 1 and self.config.rnn_layer_config.use_residual:
                    x += x_
                elif self.dropout is not None:
                    x = self.dropout(x) if not self.config.rnn_layer_config.use_cg_cell else \
                        torch.complex(self.dropout(x.real), self.dropout(x.imag))

                layer_hs[i].append(h)
                layer_outs[i].append(x)

        return torch.stack(layer_outs[-1], dim=1), layer_hs
