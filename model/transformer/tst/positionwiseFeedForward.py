from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.spectral_rnn.cgRNN.complex_linear import ComplexLinear
from model.spectral_rnn.cgRNN.util import to_complex_activation


class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed Forward Network block from Attention is All You Need.

    Apply two linear transformations to each input, separately but indetically. We
    implement them as 1D convolutions. Input and output have a shape (batch_size, d_model).

    Parameters
    ----------
    d_model:
        Dimension of input tensor.
    d_ff:
        Dimension of hidden layer, default is 2048.
    """

    def __init__(self,
                 d_model: int,
                 d_ff: Optional[int] = 2048,
                 complex: bool = False):
        """Initialize the PFF block."""
        super().__init__()

        self.complex = complex

        linear_layer = ComplexLinear if complex else nn.Linear
        self._linear1 = linear_layer(d_model, d_ff)
        self._linear2 = linear_layer(d_ff, d_model)

        self._activation = to_complex_activation(F.relu) if self.complex else F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate forward the input through the PFF block.

        Apply the first linear transformation, then a relu actvation,
        and the second linear transformation.

        Parameters
        ----------
        x:
            Input tensor with shape (batch_size, K, d_model).

        Returns
        -------
            Output tensor with shape (batch_size, K, d_model).
        """

        return self._linear2(self._activation(self._linear1(x)))
