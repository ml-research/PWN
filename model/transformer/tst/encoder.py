import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .multiHeadAttention import MultiHeadAttention, MultiHeadAttentionChunk, MultiHeadAttentionWindow
from .positionwiseFeedForward import PositionwiseFeedForward


class Encoder(nn.Module):
    """Encoder block from Attention is All You Need.

    Apply Multi Head Attention block followed by a Point-wise Feed Forward block.
    Residual sum and normalization are applied at each step.

    Parameters
    ----------
    d_model:
        Dimension of the input vector.
    q:
        Dimension of all query matrix.
    v:
        Dimension of all value matrix.
    h:
        Number of heads.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    dropout:
        Dropout probability after each MHA or PFF block.
        Default is ``0.3``.
    chunk_mode:
        Swict between different MultiHeadAttention blocks.
        One of ``'chunk'``, ``'window'`` or ``None``. Default is ``'chunk'``.
    """

    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 attention_size: int = None,
                 dropout: float = 0.3,
                 chunk_mode: str = 'chunk',
                 complex: bool = False,
                 native_complex: bool = False):
        """Initialize the Encoder block"""
        super().__init__()

        self.complex = complex
        self.native_complex = native_complex

        chunk_mode_modules = {
            'chunk': MultiHeadAttentionChunk,
            'window': MultiHeadAttentionWindow,
        }

        if chunk_mode in chunk_mode_modules.keys():
            MHA = chunk_mode_modules[chunk_mode]
        elif chunk_mode is None:
            MHA = MultiHeadAttention
        else:
            raise NameError(
                f'chunk_mode "{chunk_mode}" not understood. Must be one of {", ".join(chunk_mode_modules.keys())} or None.')

        self._selfAttention = MHA(d_model, q, v, h, attention_size=attention_size, native_complex=native_complex)
        self._feedForward = PositionwiseFeedForward(d_model, complex=complex)

        self._layerNorm1 = nn.LayerNorm(d_model)
        self._layerNorm2 = nn.LayerNorm(d_model)

        self._dopout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate the input through the Encoder block.

        Apply the Multi Head Attention block, add residual and normalize.
        Apply the Point-wise Feed Forward block, add residual and normalize.

        Parameters
        ----------
        x:
            Input tensor with shape (batch_size, K, d_model).

        Returns
        -------
            Output tensor with shape (batch_size, K, d_model).
        """
        # Self attention
        residual = x
        
        if self.complex:
            if not self.native_complex:
                x_aaa = self._selfAttention(x.real, x.real, x.real)
                x_aab = self._selfAttention(x.real, x.real, x.imag)
                x_aba = self._selfAttention(x.real, x.imag, x.real)
                x_baa = self._selfAttention(x.imag, x.real, x.real)
                x_abb = self._selfAttention(x.real, x.imag, x.imag)
                x_bab = self._selfAttention(x.imag, x.real, x.imag)
                x_bba = self._selfAttention(x.imag, x.imag, x.real)
                x_bbb = self._selfAttention(x.imag, x.imag, x.imag)

                x_a = x_aaa - x_abb - x_bab - x_bba
                x_b = -x_bbb + x_baa + x_aba + x_aab
            else:
               x_ = self._selfAttention(x, x, x)
               x_a, x_b = x.real, x_.imag
            
            x = torch.complex(self._layerNorm1(self._dopout(x_a)), self._layerNorm1(self._dopout(x_b)))
        else:
            x = self._selfAttention(query=x, key=x, value=x)
            x = self._dopout(x)
            x = self._layerNorm1(x + residual)

        # Feed forward
        residual = x
        x = self._feedForward(x)

        if self.complex:
            x = torch.complex(self._layerNorm2(self._dopout(x.real)), self._layerNorm2(self._dopout(x.imag)))
        else:
            x = self._dopout(x)
            x = self._layerNorm2(x + residual)

        return x

    @property
    def attention_map(self) -> torch.Tensor:
        """Attention map after a forward propagation,
        variable `score` in the original paper.
        """
        return self._selfAttention.attention_map
