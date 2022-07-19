from model.model import SPN
from model.cwspn.weight_nn import WeigthNN
import model.cwspn.region_graph as region_graph
from model.cwspn.cwspn_config import CWSPNConfig
from model.cwspn.rat_spn import RatSpn, SpnArgs, IndexParamProvider

import torch
import numpy as np

import pickle

# Use GPU if avaiable
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CWSPN(SPN):

    def __init__(self, config: CWSPNConfig):
        super(CWSPN, self).__init__(use_stft=True)

        self.rg = None
        self.args = None
        self.spn = None
        self.weight_nn = None
        self.stft_module = None
        self.input_sizes = None

        self.num_sum_params = None
        self.num_leaf_params = None

        self.config = config

    def train(self, x_in, y_in, stft_module, batch_size=256, epochs=10):
        self.stft_module = stft_module

        # Model is trained jointly on all groups
        x_ = np.concatenate(list(x_in.values()), axis=0)[:, :, -1]
        y_ = np.concatenate(list(y_in.values()), axis=0)[:, :, -1]

        # FFT and preparation
        x, y = self.prepare_input(x_, y_)

        # Create Nets
        self.input_sizes = x.shape[1], y.shape[1]
        self.create_net()

        # Optimizer & Loss
        criterion = lambda out: -1 * torch.logsumexp(out, dim=1).mean()
        optimizer = torch.optim.Adam(list(self.spn.parameters()) + list(self.weight_nn.parameters()))

        # Train
        for epoch in range(epochs):
            idx_batches = torch.randperm(x.shape[0], device=device).split(batch_size)

            epoch_loss = 0
            for i, idx in enumerate(idx_batches):
                optimizer.zero_grad()
                batch_x, batch_y = x[idx, :].detach().clone(), y[idx, :].detach().clone()

                sum_params, leaf_params = self.weight_nn(batch_x.reshape((batch_x.shape[0], batch_x.shape[1] *
                                                                          (2 if self.use_stft else 1),)))
                self.args.param_provider.sum_params = sum_params
                self.args.param_provider.leaf_params = leaf_params

                outputs = self.spn(batch_y)
                loss = criterion(outputs)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                if i % 100 == 0:
                    print(f'Epoch {epoch + 1}: Step {i} / {len(idx_batches)}. Avg. LL: {epoch_loss / (i + 1)}')

            print(f'Epoch {epoch + 1} / {epochs} done. Avg. LL: {epoch_loss / len(idx_batches)}')

    def predict(self, x_, y_, stft_y=True, batch_size=256):
        res = {}

        for key in x_.keys():
            x, y = self.prepare_input(x_[key], y_[key], stft_y=stft_y)

            self.spn.eval()
            self.weight_nn.eval()

            ll_cond = []
            idx_batches = torch.arange(x.shape[0], device=device).split(batch_size)
            for i, idx in enumerate(idx_batches):
                batch_x, batch_y = x[idx, :].detach().clone(), y[idx, :].detach().clone()

                sum_params, leaf_params = self.weight_nn(batch_x.reshape((batch_x.shape[0], batch_x.shape[1]
                                                                          * (2 if self.use_stft else 1))))
                self.args.param_provider.sum_params = sum_params
                self.args.param_provider.leaf_params = leaf_params

                outputs = self.spn(batch_y)
                ll_cond.append(torch.logsumexp(outputs, dim=1).detach().cpu())

            ll_cond = np.expand_dims(np.concatenate(ll_cond, axis=0), axis=-1)
            res[key] = (ll_cond,)

        return res

    def predict_ll_per_window(self, x_, y_, stft_y=True, batch_size=256, take_only_first_sample=False):
        res = {}

        for key in x_.keys():
            x, y = self.prepare_input(x_[key], y_[key], stft_y=stft_y)

            self.spn.eval()
            self.weight_nn.eval()

            ll_cond = []
            idx_batches = torch.arange(x.shape[0], device=device).split(batch_size)
            for i, idx in enumerate(idx_batches):
                batch_x, batch_y = x[idx, :].detach().clone(), y[idx, :].detach().clone()

                sum_params, leaf_params = self.weight_nn(batch_x.reshape((batch_x.shape[0], batch_x.shape[1]
                                                                          * (2 if self.use_stft else 1))))
                self.args.param_provider.sum_params = sum_params
                self.args.param_provider.leaf_params = leaf_params

                outs = []
                for j in range(self.amt_prediction_windows):
                    marginalized = [[1
                                    if j * self.reduced_coefficient_size <= k < (j + 1) * self.reduced_coefficient_size
                                    else 0
                                    for k in range(self.amt_prediction_windows * self.reduced_coefficient_size)]]
                    out = self.spn(batch_y, marginalized=torch.tensor(marginalized).to('cuda'))

                    if take_only_first_sample:
                        out = out[0:1]

                    outs.append(torch.logsumexp(out, dim=1).detach().cpu())

                ll_cond.append(np.concatenate(outs, axis=0))

            ll_cond = np.expand_dims(np.concatenate(ll_cond, axis=0), axis=-1)
            res[key] = (ll_cond,)

        return res

    def predict_ll_per_window_with_confidence(self, x_, y_, stft_y=True):
        res, c_l, c_u = {}, {}, {}

        # TODO: temporary
        if self.reduced_coefficient_size is None:
            self.reduced_coefficient_size = 12

        for key in x_.keys():
            x, y = self.prepare_input(x_[key], y_[key], stft_y=stft_y)

            self.spn.eval()
            self.weight_nn.eval()

            sum_params, leaf_params = self.weight_nn(x.reshape((x.shape[0], x.shape[1]
                                                                * (2 if self.use_stft else 1),)))
            self.args.param_provider.sum_params = sum_params
            self.args.param_provider.leaf_params = leaf_params

            outs = []
            confidence_lower = []
            confidence_upper = []
            for j in range(self.amt_prediction_windows):
                w_start = j * self.reduced_coefficient_size
                w_end = (j + 1) * self.reduced_coefficient_size

                marginalized = [[1 if w_start <= k < w_end else 0
                                for k in range(self.amt_prediction_windows * self.reduced_coefficient_size)]]
                out = self.spn(y, marginalized=torch.tensor(marginalized).to('cuda'))
                outs.append(torch.logsumexp(out, dim=1).detach().cpu())

                c_l_ = y[:, w_start:w_end]
                for k in range(c_l_.shape[1]):  # Loop through fc of window
                    for l in range(c_l_.shape[-1]):  # Loop through real and imag parts of fc
                        c_l_[:, k, l]

            ll_cond = np.expand_dims(np.concatenate(outs, axis=0), axis=-1)
            res[key] = (ll_cond,)

        return res, c_l, c_u

    def predict_mpe(self, x_, y_empty, batch_size=256):
        res = {}

        for key in x_.keys():
            x, y = self.prepare_input(x_[key][:, :, -1], y_empty[key])

            self.spn.eval()
            self.weight_nn.eval()

            mpes = []
            mpes_r = []
            idx_batches = torch.arange(x.shape[0], device=device).split(batch_size)
            for i, idx in enumerate(idx_batches):
                batch_x, batch_y = x[idx, :].detach(), y[idx, :].detach()

                sum_params, leaf_params = self.weight_nn(batch_x.reshape((batch_x.shape[0], batch_x.shape[1]
                                                                          * (2 if self.use_stft else 1),)))
                self.args.param_provider.sum_params = sum_params
                self.args.param_provider.leaf_params = leaf_params

                mpe = self.spn.reconstruct_batch(batch_y)
                mpe_ = torch.complex(mpe[..., 0], mpe[..., 1]).reshape(
                    (mpe.shape[0], self.amt_prediction_windows, -1)).swapaxes(-1, -2)
                mpe_r = self.stft_module(mpe_, reverse=True)

                mpes_r.append(mpe_r.detach().cpu())
                mpes.append(mpe.detach().cpu())

            res[key] = (np.concatenate(mpes, axis=0), np.concatenate(mpes_r, axis=0))

        return res

    def parameters(self):
        return list(self.spn.parameters()) + list(self.weight_nn.parameters())

    # TODO: Now, with the config object, it would probably be easier just saving
    #  the config object together with the model
    def save(self, filepath):
        """
        Saves a model to a given filepath
        """

        # Store model settings
        with open(f'{filepath}_wcspn_settings.pkl', 'wb') as f:
            pickle.dump({key: value for key, value in self.__dict__.items() if key != 'spn' and key != 'weight_nn'}, f)

        # Store the nets itself
        torch.save(self.weight_nn.state_dict(), filepath + '_wcspn_weight_nn')

    def load(self, filepath):
        """
        Loads a model from a given filepath
        """

        # Load model settings
        with open(f'{filepath}_wcspn_settings.pkl', 'rb') as f:
            attributes = pickle.load(f)

            for attr, val in attributes.items():
                setattr(self, attr, val)

        # Create net
        self.create_net()

        # Load net itself
        self.weight_nn.load_state_dict(torch.load(filepath + '_wcspn_weight_nn'))

    def create_net(self):
        self.rg = region_graph.RegionGraph(range(self.input_sizes[1]))
        for _ in range(0, self.config.rg_splits):
            self.rg.random_split(self.config.rg_split_size, self.config.rg_split_recursion)

        self.args = SpnArgs()
        self.args.num_sums = self.config.num_sums
        self.args.num_gauss = self.config.num_gauss
        self.args.gauss_min_mean = self.config.gauss_min_mean
        self.args.gauss_max_mean = self.config.gauss_max_mean
        self.args.gauss_min_sigma = self.config.gauss_min_sigma
        self.args.gauss_max_sigma = self.config.gauss_max_sigma
        self.args.param_provider = IndexParamProvider()
        self.spn = RatSpn(1, region_graph=self.rg, name="spn", args=self.args).to(device)
        self.spn.num_params()

        self.num_sum_params, self.num_leaf_params = self.spn.num_params_separately()
        self.weight_nn = WeigthNN(self.input_sizes[0] * (2 if self.use_stft else 1), self.num_sum_params,
                                  self.num_leaf_params, self.config.use_rationals).to(device)
