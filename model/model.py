from abc import ABC, abstractmethod

import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Model(ABC):
    """
    Abstract super class for all prediction models
    """

    @property
    def identifier(self):
        """
        Identifier of the respective model - defaults to class name
        """

        return self.__class__.__name__

    @abstractmethod
    def train(self, x, y, embedding_sizes):
        """
        Trains the model with the given data_source
        """

        pass

    @abstractmethod
    def predict(self, x):
        """
        Returns predictions for given data_source
        """

        pass

    @abstractmethod
    def save(self, filepath):
        """
        Saves a model to a given filepath
        """

        pass

    @abstractmethod
    def load(self, filepath):
        """
        Loads a model from a given filepath
        """

        pass


# noinspection PyAbstractClass
class SPN(Model):

    def __init__(self, use_stft=True):
        self.use_stft = use_stft

        self.stft_module = None
        self.amt_prediction_windows = None
        self.reduced_coefficient_size = None

    @torch.no_grad()
    def prepare_input(self, x_, y_, stft_y=True):
        if not torch.is_tensor(x_) or not torch.is_tensor(y_):
            x_ = torch.from_numpy(x_).float().to(device)
            y_ = torch.from_numpy(y_).to(device)

        if not self.use_stft:
            return x_, y_

        if self.config.use_limited_context is not None:
            x_ = x_[:, -self.config.use_limited_context:]

        if stft_y:
            y_ = y_.float()

            if self.config.prepare_joint:
                if self.amt_prediction_windows is None or self.reduced_coefficient_size is None:
                    y_stft = self.stft_module(y_)
                    self.amt_prediction_windows = y_stft.shape[-1]
                    self.reduced_coefficient_size = y_stft.shape[-2]

                v = self.stft_module(torch.cat([x_, y_], axis=1))
                v_x, v_y = v[..., :-self.amt_prediction_windows], v[..., -self.amt_prediction_windows:]
            else:
                v_x = self.stft_module(x_)
                v_y = self.stft_module(y_)

                if self.amt_prediction_windows is None or self.reduced_coefficient_size is None:
                    self.amt_prediction_windows = v_y.shape[-1]
                    self.reduced_coefficient_size = v_y.shape[-2]

            x = self.prepare_single_input(v_x)
            y = self.prepare_single_input(v_y)

            return torch.stack([x.real, x.imag], dim=-1), torch.stack([y.real, y.imag], dim=-1)

        else:
            assert not self.config.prepare_joint

            v_x = self.stft_module(x_)
            x = self.prepare_single_input(v_x)

            if torch.is_complex(y_):
                y_ = torch.stack([y_.real, y_.imag], dim=-1)

            return torch.stack([x.real, x.imag], dim=-1), y_

    @torch.no_grad()
    def prepare_single_input(self, x_):
        x = x_.swapaxes(-2, -1)

        # If we work on window level, we see each window as an isolated input
        if self.config.window_level:
            return x.reshape((-1, x.shape[-1]))

        # On sequence level, however, all windows of a given input sequence are used as a joint input
        else:
            return x.reshape((x.shape[0], -1))
