from ..model import Model
from .model import NBeatsNet

import torch
import numpy as np

import pickle
import math

# Use GPU if avaiable
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class NBeatsPredictor(Model):

    def __init__(self):
        self.net = None
        self.use_cached_predictions = False

        self.build_net()

    def train(self, x_in, y_in, val_x, val_y, embedding_sizes, batch_size=128, epochs=10, lr=0.001, lr_decay=0.97):
        """
        Trains the model with the given data_source
        """

        # Model is trained jointly on all groups
        x_ = np.concatenate(list(x_in.values()), axis=0)[:, :, -1]
        y_ = np.concatenate(list(y_in.values()), axis=0)[:, :, -1]

        val_x_ = np.concatenate([x for x in val_x.values() if len(x) > 0], axis=0)[:, :, -1]
        val_y_ = np.concatenate([y for y in val_y.values() if len(y) > 0], axis=0)[:, :, -1]

        self.net.fit(x_, y_, (val_x_, val_y_), batch_size=batch_size, epochs=epochs)

    def predict(self, x, batch_size=1024, pred_label=None, return_coefficients=False, mpe=False):
        """
        Returns predictions for given data_source
        """

        if not self.use_cached_predictions:
            predictions = {}

            with torch.no_grad():
                for key, x_ in x.items():
                    predictions[key] = self.net.predict(x_[:, :, -1])

            if pred_label is not None:
                with open(pred_label + '_beats.pkl', 'wb') as f:
                     pickle.dump(predictions, f)

        else:
            assert pred_label is not None

            with open(pred_label + '_nbeats.pkl', 'rb') as f:
                predictions = pickle.load(f)

        return predictions

    def build_net(self):
        # TODO: Hardcoded values
        blocks = [NBeatsNet.GENERIC_BLOCK, NBeatsNet.SEASONALITY_BLOCK, NBeatsNet.TREND_BLOCK]
        nb_blocks_per_stack = 3
        hidden_dim = 128
        lr = .5 * 1e-4

        forecast_length = int(1.5 * 24)
        backcast_length = 14 * 24
        thetas_dim = (4, 4, 2)

        self.net = NBeatsNet(device, blocks, forecast_length=forecast_length, backcast_length=backcast_length,
                             thetas_dim=thetas_dim, nb_blocks_per_stack=nb_blocks_per_stack,
                             hidden_layer_units=hidden_dim)
        amt_param = sum([p.numel() for p in self.net.parameters()])
        # self.net.compile('smape', 'adam', lr=lr)
        self.net.compile('mse', 'adam', lr=lr)

    # TODO: Now, with the config object, it would probably be easier just saving
    #  the config object together with the model
    def save(self, filepath):
        """
        Saves a model to a given filepath
        """

        # Store model settings
        with open(f'{filepath}_settings.pkl', 'wb') as f:
            pickle.dump({key: value for key, value in self.__dict__.items() if key != 'net'}, f)

        # Store net itself
        torch.save(self.net.state_dict(), filepath)

    def load(self, filepath):
        """
        Loads a model from a given filepath
        """

        # Load model settings
        with open(f'{filepath}_settings.pkl', 'rb') as f:
            attributes = pickle.load(f)

            for attr, val in attributes.items():
                setattr(self, attr, val)

            # Create net
            self.build_net()

        # Load net itself
        self.net.load_state_dict(torch.load(filepath))
