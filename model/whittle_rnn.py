from .spectral_rnn import SpectralRNN, SpectralRNNConfig
from .wein import WEin, WEinConfig
from .maf import MAFEstimator
from .cwspn import CWSPN, CWSPNConfig
from .model import Model

import torch
import numpy as np

# Use GPU if avaiable
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class WhittleRNN(Model):

    def __init__(self, s_config: SpectralRNNConfig, e_config: [WEinConfig, CWSPNConfig], estimator='WEin',
                 pass_fc=True):
        self.s_config = s_config
        self.e_config = e_config

        self.pass_fc = pass_fc

        self.srnn = SpectralRNN(self.s_config)
        self.w_estimator = CWSPN(self.e_config) if estimator == 'WCSPN' \
            else (WEinsum(self.e_config) if estimator == 'WEin' else MAFEstimator())

    @property
    def identifier(self):
        """
            Identifier of the respective model - defaults to class name
        """

        return f'{self.__class__.__name__}-{self.w_estimator.identifier}'

    def train(self, x_in, y_in, embedding_sizes, batch_size=256, epochs=80, lr=0.004, lr_decay=0.97):
        self.srnn.train(x_in, y_in, embedding_sizes, batch_size, epochs, lr, lr_decay)
        self.w_estimator.train(x_in, y_in, self.srnn.net.stft, batch_size, epochs // 5)

    @torch.no_grad()
    def predict(self, x, batch_size=1024, pred_label='', mpe=False):
        predictions, f_c = self.srnn.predict(x, batch_size, pred_label=pred_label, return_coefficients=True)

        x_ = {key: x_[:, :, -1] for key, x_ in x.items()}

        if self.pass_fc:
            f_c_ = {key: f.reshape((f.shape[0], -1)) for key, f in f_c.items()}
            ll = self.w_estimator.predict(x_, f_c_, stft_y=False, batch_size=batch_size)
        else:
            ll = self.w_estimator.predict(x_, predictions, stft_y=True, batch_size=batch_size)

        if mpe:
            y_empty = {key: np.zeros((x[key].shape[0], self.srnn.net.amt_prediction_samples)) for key in x.keys()}
            predictions_mpe = self.w_estimator.predict_mpe({key: x_.copy() for key, x_ in x.items()},
                                                           y_empty, batch_size=batch_size)
            lls_mpe = self.w_estimator.predict(x_, {key: v[0] for key, v in predictions_mpe.items()},
                                               stft_y=False, batch_size=batch_size)

            # predictions, likelihoods, likelihoods_mpe, predictions_mpe
            return predictions, ll, lls_mpe, {key: v[1] for key, v in predictions_mpe.items()}

        else:
            return predictions, ll

    def save(self, filepath):
        self.srnn.save(filepath)
        self.w_estimator.save(filepath)

    def load(self, filepath):
        self.srnn.load(filepath)
        self.w_estimator.load(filepath)
        self.w_estimator.stft_module = self.srnn.net.stft
