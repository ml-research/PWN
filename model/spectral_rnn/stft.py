import torch
import torch.nn as nn

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .spectral_rnn_config import SpectralRNNConfig
    from ..transformer import TransformerConfig


# Use GPU if avaiable
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# We must make this a module and a class, so that it can be part of ModuleLists etc.
class STFT(nn.Module):

    def __init__(self, config: ['SpectralRNNConfig', 'TransformerConfig'], pad_mode='constant', onesided=True):
        super(STFT, self).__init__()

        self.value_dim = config.compressed_value_dim
        self.n_fft = config.window_size
        self.hop_length = config.step_width
        self.win_size = config.window_size
        self.pad_mode = pad_mode
        self.onesided = onesided
        self.normalize_fft = config.normalize_fft

        self.sigma = nn.Parameter(torch.full((1,), 0.7))  # Init to 0.7 as in paper
        self.config = config

    @property
    def window(self):
        sigma = self.sigma ** 2

        window = torch.arange(0, self.win_size).to(device)

        N = self.win_size - 1
        return torch.exp(-0.5 * torch.pow((window - N / 2) / (sigma * N / 2), 2))

    def forward(self, x, reverse=False):
        if not reverse:
            return torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_size,
                              window=self.window, pad_mode=self.pad_mode, onesided=self.onesided,
                              return_complex=True, normalized=self.normalize_fft)[:, :self.value_dim]
        else:
            if self.config.removed_freqs > 0:
                zero_freqs = torch.zeros((x.shape[0], self.config.removed_freqs, x.shape[2])).to(device)
                x = torch.cat([x, zero_freqs], dim=1)

            return torch.istft(x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_size,
                               window=self.window, onesided=self.onesided, return_complex=False,
                               normalized=self.normalize_fft)
