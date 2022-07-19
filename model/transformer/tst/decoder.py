import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .multiHeadAttention import MultiHeadAttention, MultiHeadAttentionChunk, MultiHeadAttentionWindow
from .positionwiseFeedForward import PositionwiseFeedForward


class Decoder(nn.Module):
    """Decoder block from Attention is All You Need.

    Apply two Multi Head Attention block followed by a Point-wise Feed Forward block.
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
        """Initialize the Decoder block"""
        super().__init__()

        chunk_mode_modules = {
            'chunk': MultiHeadAttentionChunk,
            'window': MultiHeadAttentionWindow,
        }

        self.complex = complex
        self.native_complex = native_complex

        if chunk_mode in chunk_mode_modules.keys():
            MHA = chunk_mode_modules[chunk_mode]
        elif chunk_mode is None:
            MHA = MultiHeadAttention
        else:
            raise NameError(
                f'chunk_mode "{chunk_mode}" not understood. Must be one of {", ".join(chunk_mode_modules.keys())} or None.')

        self._selfAttention = MHA(d_model, q, v, h, attention_size=attention_size, native_complex=native_complex)
        self._encoderDecoderAttention = MHA(d_model, q, v, h, attention_size=attention_size,
                                            native_complex=native_complex)
        self._feedForward = PositionwiseFeedForward(d_model, complex=complex)

        self._layerNorm1 = nn.LayerNorm(d_model)
        self._layerNorm2 = nn.LayerNorm(d_model)
        self._layerNorm3 = nn.LayerNorm(d_model)

        self._dopout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Propagate the input through the Decoder block.

        Apply the self attention block, add residual and normalize.
        Apply the encoder-decoder attention block, add residual and normalize.
        Apply the feed forward network, add residual and normalize.

        Parameters
        ----------
        x:
            Input tensor with shape (batch_size, K, d_model).
        memory:
            Memory tensor with shape (batch_size, K, d_model)
            from encoder output.

        Returns
        -------
        x:
            Output tensor with shape (batch_size, K, d_model).
        """
        # Self attention
        residual = x

        if self.complex:
            if not self.native_complex:
                x_aaa = self._selfAttention(x.real, x.real, x.real, mask="subsequent")
                x_aab = self._selfAttention(x.real, x.real, x.imag, mask="subsequent")
                x_aba = self._selfAttention(x.real, x.imag, x.real, mask="subsequent")
                x_baa = self._selfAttention(x.imag, x.real, x.real, mask="subsequent")
                x_abb = self._selfAttention(x.real, x.imag, x.imag, mask="subsequent")
                x_bab = self._selfAttention(x.imag, x.real, x.imag, mask="subsequent")
                x_bba = self._selfAttention(x.imag, x.imag, x.real, mask="subsequent")
                x_bbb = self._selfAttention(x.imag, x.imag, x.imag, mask="subsequent")

                x_a = x_aaa - x_abb - x_bab - x_bba
                x_b = -x_bbb + x_baa + x_aba + x_aab
            else:
               x_ = self._selfAttention(x, x, x, mask="subsequent")
               x_a, x_b = x.real, x_.imag

            x = torch.complex(self._layerNorm1(self._dopout(x_a)), self._layerNorm1(self._dopout(x_b)))
        else:
            x = self._selfAttention(query=x, key=x, value=x)
            x = self._dopout(x)
            x = self._layerNorm1(x + residual)

        # Encoder-decoder attention
        residual = x
        
        if self.complex:
            if not self.native_complex:
                x_acc = self._encoderDecoderAttention(x.real, memory.real, memory.real)
                x_add = self._encoderDecoderAttention(x.real, memory.imag, memory.imag)
                x_bcd = self._encoderDecoderAttention(x.imag, memory.real, memory.imag)
                x_bdc = self._encoderDecoderAttention(x.imag, memory.imag, memory.real)
                x_acd = self._encoderDecoderAttention(x.real, memory.real, memory.imag)
                x_adc = self._encoderDecoderAttention(x.real, memory.imag, memory.real)
                x_bcc = self._encoderDecoderAttention(x.imag, memory.real, memory.real)
                x_bdd = self._encoderDecoderAttention(x.imag, memory.imag, memory.imag)

                x_a = x_acc - x_add - x_bcd - x_bdc
                x_b = x_acd + x_adc + x_bcc - x_bdd
            else:
               x_ = self._selfAttention(x, memory, memory)
               x_a, x_b = x.real, x_.imag

            x = torch.complex(self._layerNorm2(self._dopout(x_a)), self._layerNorm2(self._dopout(x_b)))
        else:
            x = self._encoderDecoderAttention(query=x, key=memory, value=memory)
            x = self._dopout(x)
            x = self._layerNorm2(x + residual)

        # Feed forward
        residual = x
        x = self._feedForward(x)

        if self.complex:
            x = torch.complex(self._layerNorm3(self._dopout(x.real)), self._layerNorm3(self._dopout(x.imag)))
        else:
            x = self._dopout(x)
            x = self._layerNorm3(x + residual)

        return x
