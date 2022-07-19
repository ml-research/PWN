from ..model import Model
from .stft import STFT
from .spectral_rnn_config import SpectralRNNConfig
from .manifold_optimization import ManifoldOptimizer
from .cgRNN import RNNLayer, CGCell, ComplexLinear, to_complex_activation, clip_grad_value_complex_
from util.plot import plot_experiment

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

import pickle
import math

# Use GPU if avaiable
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SpectralRNN(Model):
    """
    Simple baseline rnn model based on gru units
    """

    def __init__(self, config: SpectralRNNConfig):

        self.net = None
        self.final_amt_pred_samples = None

        self.config = config

        self.config.step_width = int(self.config.window_size * self.config.overlap)
        self.config.value_dim = self.config.window_size // 2 + 1

    def train(self, x_in, y_in, embedding_sizes, batch_size=128, epochs=1, lr=0.001, lr_decay=0.97):
        """
        Trains the model with the given data_source
        """

        # TODO: Adjustment for complex optimization, needs documentation
        if self.config.rnn_layer_config.use_cg_cell:
            lr /= 4

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
        optimizer = ManifoldOptimizer(self.net.parameters(), lr, torch.optim.RMSprop, alpha=0.9) \
            if self.config.rnn_layer_config.use_cg_cell \
            else torch.optim.RMSprop(self.net.parameters(), lr=lr, alpha=0.9)

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

                # We do it as original in the paper using the value, but is clip_grad_norm_ myb better?
                if self.config.clip_gradient_value > 0:
                    clip_grad_value_complex_(self.net.parameters(), self.config.clip_gradient_value)
                optimizer.step()

                avg_loss += loss.item()

                if counter % 100 == 0:
                    print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter,
                                                                                               len(train_loader),
                                                                                               avg_loss / counter))

            # Not exactly as in original paper where we have a decay step of 390
            # --> to make it comparable, we set the decay rate a bit higher
            print("Epoch {}/{} Done, Total Loss: {}, Current LR: {}".format(epoch, epochs, avg_loss / len(train_loader),
                                                                         lr_scheduler.get_last_lr()))
            lr_scheduler.step()
        print("Total Training Time: {} seconds".format(str(sum(epoch_times))))

    def predict(self, x, batch_size=1024, pred_label='', return_coefficients=False):
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

            with open(pred_label + '.pkl', 'wb') as f:
                 pickle.dump((predictions, coefficients,
                              self.final_amt_pred_samples, self.net.amt_prediction_samples), f)

        else:
            with open(pred_label + '.pkl', 'rb') as f:
                predictions, coefficients, self.final_amt_pred_samples, self.net.amt_prediction_samples = pickle.load(f)

        if return_coefficients:
            return predictions, coefficients
        else:
            return predictions

    def build_net(self):
        self.net = SpectralRNNNet(self.config)
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
                                'amt_prediction_windows': self.net.amt_prediction_windows,}}, f)

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


class SpectralRNNNet(nn.Module):

    def __init__(self, config: SpectralRNNConfig):
        super(SpectralRNNNet, self).__init__()

        assert not config.rnn_layer_config.use_cg_cell or \
               config.rnn_layer_config.use_gated, 'SpectralRNN only supports gated complex cells!'

        config.compressed_value_dim = config.value_dim // config.fft_compression
        config.removed_freqs = config.value_dim - config.compressed_value_dim
        # config.input_dim = sum([config.embedding_dim if e > 0 else config.compressed_value_dim
        #                         for e in config.embedding_sizes])
        config.input_dim = config.compressed_value_dim

        self.stft = STFT(config)

        if config.use_only_ts_input:
            self.f_in = nn.ModuleList([self.stft])
        else:
            self.f_in = nn.ModuleList([nn.Embedding(num_embeddings=e, embedding_dim=config.embedding_dim).to(device)
                                       if e > 1 else (nn.Linear(1, 1) if e == 1 else self.stft)
                                       for e in config.embedding_sizes])

        if config.rnn_layer_config.use_cg_cell:
            cell = CGCell
        elif config.rnn_layer_config.use_gated:
            cell = torch.nn.GRUCell
        else:
            cell = torch.nn.LSTMCell

        self.rnn = RNNLayer(cell, config)

        # Not part of original model
        if config.use_add_linear:
            self.add_pre_act = to_complex_activation(torch.nn.LeakyReLU(0.1))
            self.add_linear = ComplexLinear(config.compressed_value_dim, config.compressed_value_dim, lambda x_: x_)

        self.amt_prediction_samples = None
        self.amt_prediction_windows = None

        self.config = config

    def forward(self, x_in, y_in=None, return_coefficients=False):

        if self.config.use_only_ts_input:
            x = [self.f_in[0](x_in[:, :, -1])]
        else:
            x = [f_in_(x_in[:, :, i].long()) if i in [1, 2] else (f_in_(x_in[:, :, i:i+1].float() / 100.) if i == 0 else f_in_(x_in[:, :, i])) for i, f_in_ in enumerate(self.f_in)]

        stft_len = x[-1].shape[1]

        x[-1] = x[-1].swapaxes(-2, -1)

        encoder_in = x[-1]

        # Keep stats of y for prediction
        if self.amt_prediction_samples is None or self.amt_prediction_windows is None:
            self.amt_prediction_samples = y_in.shape[1]
            self.amt_prediction_windows = self.stft(y_in).shape[-1]

        decoder_in = torch.zeros([encoder_in.shape[0], self.amt_prediction_windows, encoder_in.shape[-1]]).to(device)

        gru_in = torch.cat([encoder_in, decoder_in], dim=-2).type(torch.complex64)\
            .to(device)

        gru_in_extended = gru_in

        if not self.config.rnn_layer_config.use_cg_cell:
            gru_in_extended = torch.cat([gru_in_extended.real, gru_in_extended.imag], dim=-1)

        gru_out, _ = self.rnn(gru_in_extended)

        decoder_out = gru_out[:, -self.amt_prediction_windows:]

        if not self.config.rnn_layer_config.use_cg_cell:
            decoder_out = torch.complex(decoder_out[:, :, :stft_len], decoder_out[:, :, stft_len:])

        if self.config.use_add_linear:
            decoder_out = self.add_pre_act(decoder_out)
            decoder_out = self.add_linear(decoder_out)

        decoder_out_ = decoder_out.swapaxes(-2, -1)

        out = self.stft(decoder_out_, reverse=True)[:, -self.amt_prediction_samples:]

        if return_coefficients:
            return out, decoder_out
        else:
            return out
