from ..model import Model
from .tst import Transformer
from ..spectral_rnn.stft import STFT
from .transformer_config import TransformerConfig
from ..spectral_rnn.cgRNN import clip_grad_value_complex_
from util.plot import plot_experiment

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

import pickle
import math

# Use GPU if avaiable
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class TransformerPredictor(Model):

    def __init__(self, config: TransformerConfig):
        self.net = None
        self.final_amt_pred_samples = None

        self.config = config

        self.config.step_width = int(self.config.window_size * self.config.overlap)
        self.config.value_dim = self.config.window_size // 2 + 1

        config.compressed_value_dim = config.value_dim // config.fft_compression
        config.removed_freqs = config.value_dim - config.compressed_value_dim
        config.input_dim = config.compressed_value_dim

    def train(self, x_in, y_in, embedding_sizes, batch_size=128, epochs=1, lr=0.001, lr_decay=0.97):
        """
        Trains the model with the given data_source
        """

        # Model is trained jointly on all groups
        x_ = np.concatenate(list(x_in.values()), axis=0)
        y_ = np.concatenate(list(y_in.values()), axis=0)[:, :, -1]

        if self.config.x_as_labels:
            y_ = np.concatenate([x_[:, 1:, -1], y_], axis=1)

        if self.final_amt_pred_samples is None:
            self.final_amt_pred_samples = next(iter(y_in.values())).shape[1]

        train_data = TensorDataset(torch.from_numpy(x_), torch.from_numpy(y_))
        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=False)

        self.config.embedding_sizes = embedding_sizes
        self.build_net()

        if self.config.use_cached_predictions:
            return

        # Defining loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        # optimizer = torch.optim.RMSprop(self.net.parameters(), lr=lr, alpha=0.9)

        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay)

        self.net.train()
        print(f'Starting Training of {self.identifier} model')
        epoch_times = []
        # Start training loop
        for epoch in range(epochs):

            avg_loss = 0.
            counter = 0
            for x, label in train_loader:
                self.net.zero_grad()
                counter += 1

                out = self.net(x.to(device), label.to(device))
                loss = criterion(out, label.to(device).float())

                loss.backward()

                if self.config.clip_gradient_value > 0:
                    clip_grad_value_complex_(self.net.parameters(), self.config.clip_gradient_value)

                optimizer.step()

                avg_loss += loss.item()

                if counter % 100 == 0:
                    print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter,
                                                                                               len(train_loader),
                                                                                               avg_loss / counter))

            print("Epoch {}/{} Done, Total Loss: {}, Current LR: {}".format(epoch, epochs, avg_loss / len(train_loader),
                                                                         lr_scheduler.get_last_lr()))
            lr_scheduler.step()
        print("Total Training Time: {} seconds".format(str(sum(epoch_times))))

    def predict(self, x, batch_size=1024, pred_label=None, return_coefficients=False):
        """
        Returns predictions for given data_source
        """

        if not self.config.use_cached_predictions:
            predictions = {}
            coefficients = {}

            self.net.eval()
            with torch.no_grad():
                for key, x_ in x.items():
                    predictions[key], coefficients[key] = [], []

                    for i in range(math.ceil(x_.shape[0] / batch_size)):
                        batch = torch.from_numpy(x_[i * batch_size:(i + 1) * batch_size])
                        out = self.net(batch.to(device).float(), return_coefficients=return_coefficients)

                        if return_coefficients:
                            coefficients[key].append(out[1].detach().cpu())
                            out = out[0]

                        predictions[key].append(out[:, -self.final_amt_pred_samples:].detach().cpu())

                    predictions[key] = np.concatenate(predictions[key], axis=0)
                    coefficients[key] = np.concatenate(coefficients[key], axis=0)

            if pred_label is not None:
                with open(pred_label + '.pkl', 'wb') as f:
                     pickle.dump((predictions, coefficients,
                                  self.final_amt_pred_samples, self.net.amt_prediction_samples), f)

        else:
            assert pred_label is not None

            with open(pred_label + '.pkl', 'rb') as f:
                predictions, coefficients, self.final_amt_pred_samples, self.net.amt_prediction_samples = pickle.load(f)

        if return_coefficients:
            return predictions, coefficients
        else:
            return predictions

    def build_net(self):
        in_factor = 1 if complex else 2

        self.net = TransformerNet(self.config, self.config.input_dim * in_factor, self.config.hidden_dim,
                                  self.config.input_dim * in_factor, self.config.q, self.config.k, self.config.heads,
                                  self.config.num_enc_dec, attention_size=self.config.attention_size,
                                  dropout=self.config.dropout, chunk_mode=self.config.chunk_mode, pe=self.config.pe,
                                  complex=self.config.is_complex, native_complex=self.config.native_complex)
        self.net = self.net.to(device)

    # TODO: Now, with the config object, it would probably be easier just saving
    #  the config object together with the model
    def save(self, filepath):
        """
        Saves a model to a given filepath
        """

        # Store model settings
        with open(f'{filepath}_settings.pkl', 'wb') as f:
            pickle.dump({'p': {key: value for key, value in self.__dict__.items() if key != 'net'},
                         'n': {'amt_prediction_samples': self.net.amt_prediction_samples,
                               'amt_prediction_windows': self.net.amt_prediction_windows}}, f)

        # Store net itself
        torch.save(self.net.state_dict(), filepath)

    def load(self, filepath):
        """
        Loads a model from a given filepath
        """

        # Load model settings
        with open(f'{filepath}_settings.pkl', 'rb') as f:
            attributes = pickle.load(f)

            for attr, val in attributes['p'].items():
                setattr(self, attr, val)

            # Create net
            self.build_net()

            for attr, val in attributes['n'].items():
                setattr(self.net, attr, val)

        # Load net itself
        self.net.load_state_dict(torch.load(filepath))


class TransformerNet(Transformer):

    def __init__(self, config: TransformerConfig,
                 d_input: int,
                 d_model: int,
                 d_output: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 attention_size: int = None,
                 dropout: float = 0.3,
                 chunk_mode: str = 'chunk',
                 pe: str = 'original',
                 pe_period: int = None,
                 complex: bool = False,
                 native_complex: bool = False):
        super(TransformerNet, self).__init__(d_input, d_model, d_output, q, v, h, N, attention_size, dropout, chunk_mode,
                                             pe, pe_period, complex=complex, native_complex=native_complex)

        self.d_output = d_output
        self.complex = complex
        self.native_complex = native_complex

        self.stft = STFT(config)

        self.amt_prediction_samples = None
        self.amt_prediction_windows = None

    def forward(self, x: torch.Tensor, y=None, return_coefficients=False):

        if self.amt_prediction_samples is None:
            self.amt_prediction_samples = 144
            self.amt_prediction_windows = 4

        x_ = self.stft(x[:, :, -1]).swapaxes(-2, -1)

        if not self.complex:
            x_ = torch.cat([x_.real, x_.imag], dim=-1)

        f_c = super().forward(x_, self.amt_prediction_windows)[:, -self.amt_prediction_windows:]

        if not self.complex:
            f_c = torch.complex(f_c[:, :, :self.d_output // 2], f_c[:, :, self.d_output // 2:])

        prediction = self.stft(f_c.swapaxes(-2, -1), reverse=True)[:, -self.amt_prediction_samples:]

        return prediction, f_c if return_coefficients else prediction
