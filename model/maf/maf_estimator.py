from ..model import SPN
from .maf import MAF
from .made import MADE
from .maf_config import MAFConfig
from .utils.train import train_one_epoch_maf
from .utils.test import test_maf

import numpy as np
import torch
import torch.nn as nn

import datetime as dt
import pickle

# Use GPU if avaiable
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# TODO: Add sampling and reconstruction (pseudo forecasting) support
class MAFEstimator(SPN):

    def __init__(self, config: MAFConfig = MAFConfig()):
        super(MAFEstimator, self).__init__(use_stft=True)

        self.config = config

        self.use_made = False

        self.n_mades = 1
        self.hidden_dims = [48]
        self.random_order = False
        self.patience = 30  # For early stopping
        self.seed = 290765
        self.plot = True
        # self.config.use_limited_context = 6 * 12
        # self.window_level = True

        self.model = None

        self.input_sizes = None
        self.stft_module = None
        self.amt_windows_per_sequence = None

    @property
    def final_input_sizes(self):
        return (sum(self.input_sizes) * (2 if self.use_stft else 1)) if type(self.input_sizes) == tuple \
            else self.input_sizes

    def train(self, x_in, y_in, stft_module, batch_size=128, epochs=5, lr=1e-4):
        self.stft_module = stft_module

        # Model is trained jointly on all groups
        x_ = np.concatenate(list(x_in.values()), axis=0)[:, :, -1]
        y_ = np.concatenate(list(y_in.values()), axis=0)[:, :, -1]

        # FFT and preparation
        x = self.prepare_input(np.concatenate([x_, y_], axis=-1))

        # FFT Size
        self.input_sizes = x.shape[1]

        # Create MAF
        self.create_net()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-6)

        # Train
        lls = []
        start = dt.datetime.now()
        train_data = torch.utils.data.TensorDataset(x)
        train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=False)
        for epoch_count in range(epochs):
            train_loss = train_one_epoch_maf(self.model, epoch_count, optimizer, train_loader)
            print(f'Epoch {epoch_count} Loss: {train_loss}')

        print((dt.datetime.now() - start).total_seconds())

    def create_net(self):
        if self.use_made:
            self.model = MADE(self.final_input_sizes, self.hidden_dims, gaussian=True).to(device)
        else:
            self.model = MAF(self.final_input_sizes, self.n_mades, self.hidden_dims).to(device)

    def predict(self, x_, y_, stft_y=True, batch_size=128):
        res = {}

        for key in x_.keys():
            x, y = self.prepare_input(x_[key], y_[key], stft_y=stft_y)
            test_data = torch.utils.data.TensorDataset(torch.cat([x, y], dim=1).reshape((x.shape[0], -1)))
            test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=batch_size, drop_last=False)

            res[key] = test_made(self.model, test_loader) if self.use_made else test_maf(self.model, test_loader)

            if False:  # if self.window_level:
                amt_prediction_windows = 4  # TODO: temp
                res[key] = res[key].reshape((-1, 1, self.amt_windows_per_sequence))[:, :, -amt_prediction_windows:].mean(axis=-1)
            else:
                res[key] = torch.nan_to_num(res[key], nan=-1e5)
                res[key] = torch.clamp(res[key], min=-1e5, max=1e5)
                res[key] = (res[key].unsqueeze(dim=-1).cpu(),)

        return res

    def parameters(self):
        return self.model.parameters()

    # TODO: Now, with the config object, it would probably be easier just saving
    #  the config object together with the model
    def save(self, filepath):
        """
        Saves a model to a given filepath
        """

        # Store model settings
        with open(f'{filepath}_maf_settings.pkl', 'wb') as f:
            pickle.dump({key: value for key, value in self.__dict__.items() if key != 'model'}, f)

        # Store the nets itself
        torch.save(self.model, filepath + '_maf')

    def load(self, filepath):
        """
        Loads a model from a given filepath
        """

        # Load model settings
        with open(f'{filepath}_maf_settings.pkl', 'rb') as f:
            attributes = pickle.load(f)

            for attr, val in attributes.items():
                setattr(self, attr, val)

        # Create net
        self.create_net()

        # Load net itself
        self.model = torch.load(filepath + '_maf')
