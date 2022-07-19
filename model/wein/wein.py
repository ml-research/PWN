from ..model import SPN
from .wein_config import WEinConfig
from .EinsumNetwork import Graph, EinsumNetwork

import numpy as np
import torch
import torch.nn as nn

import datetime as dt
import pickle

# Use GPU if avaiable
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class WEin(SPN):

    def __init__(self, config: WEinConfig = WEinConfig()):
        super(WEin, self).__init__()

        self.config = config
        self.graph = None
        self.net = None

        self.stft_module = None

    def train(self, x_in, y_in, stft_module, batch_size=128, epochs=10):
        self.stft_module = stft_module

        # Model is trained jointly on all groups
        x_ = np.concatenate(list(x_in.values()), axis=0)[:, :, -1]
        y_ = np.concatenate(list(y_in.values()), axis=0)[:, :, -1]

        # FFT and preparation
        x_f, y_f = self.prepare_input(x_, y_)
        x = torch.cat([x_f, y_f], dim=1)

        # FFT Size
        self.config.input_size = x.shape[1]

        # Create (W)-Einet
        self.create_net()

        # Train
        lls = []
        start = dt.datetime.now()
        for epoch_count in range(epochs):

            # Evaluate
            self.net.eval()
            train_ll = EinsumNetwork.eval_loglikelihood_batched(self.net, x, batch_size=batch_size)
            lls.append(train_ll / x.shape[0])
            print("[{}] Train LL {}".format(epoch_count, lls[-1]))
            self.net.train()

            idx_batches = torch.randperm(x.shape[0], device=device).split(batch_size)

            total_ll = 0.0
            for idx in idx_batches:
                batch_x = x[idx, :]
                outputs = self.net.forward(batch_x)
                ll_sample = EinsumNetwork.log_likelihoods(outputs)
                log_likelihood = ll_sample.sum()
                log_likelihood.backward()

                self.net.em_process_batch()
                total_ll += log_likelihood.detach().item()

            self.net.em_update()

        self.net.eval()
        train_ll = EinsumNetwork.eval_loglikelihood_batched(self.net, x, batch_size=batch_size)
        lls.append(train_ll / x.shape[0])
        print("[{}] Train LL {}".format(epochs, lls[-1]))

        print((dt.datetime.now() - start).total_seconds())

    def predict(self, x_, y_, stft_y=True, batch_size=256):
        res = {}

        for key, val in x_.items():
            x, y = self.prepare_input(x_[key], y_[key], stft_y=stft_y)

            lls_joint, lls_marginal = [], []
            idx_batches = torch.arange(x.shape[0], device=device).split(batch_size)
            for i, idx in enumerate(idx_batches):
                batch_x, batch_y = x[idx, :].detach(), y[idx, :].detach()
                net_in = torch.cat([batch_x, batch_y], dim=1)

                ll_joint = EinsumNetwork.eval_single_loglikelihood_batched(self.net, net_in, batch_size=batch_size)
                self.net.set_marginalization_idx(list(range(x.shape[1], x.shape[1] + y.shape[1])))
                ll_marginal = EinsumNetwork.eval_single_loglikelihood_batched(self.net, net_in, batch_size=batch_size)
                self.net.set_marginalization_idx([])

                lls_joint.append(ll_joint.detach().cpu().numpy())
                lls_marginal.append(ll_marginal.detach().cpu().numpy())

            lls_joint = np.concatenate(lls_joint, axis=0)
            lls_marginal = np.concatenate(lls_marginal, axis=0)

            res[key] = (lls_joint - lls_marginal, lls_joint, lls_marginal)

            if self.config.window_level:
                raise NotImplementedError()

        return res

    def predict_ll_per_window(self, x_, y_, stft_y=True, batch_size=256):
        res = {}

        for key in x_.keys():
            x, y = self.prepare_input(x_[key], y_[key], stft_y=stft_y)

            self.net.eval()

            ll_cond = []
            idx_batches = torch.arange(x.shape[0], device=device).split(batch_size)
            for i, idx in enumerate(idx_batches):
                batch_x, batch_y = x[idx, :].detach().clone(), y[idx, :].detach().clone()

                outs = []
                for j in range(self.amt_prediction_windows):
                    marginalized = list(range(j * self.reduced_coefficient_size + batch_x.shape[1],
                                              (j + 1) * self.reduced_coefficient_size + batch_x.shape[1]))

                    net_in = torch.cat([batch_x, batch_y], dim=1)

                    ll_joint = EinsumNetwork.eval_single_loglikelihood_batched(self.net, net_in, batch_size=batch_size)
                    self.net.set_marginalization_idx(marginalized)
                    ll_marginal = EinsumNetwork.eval_single_loglikelihood_batched(self.net, net_in,
                                                                                  batch_size=batch_size)
                    self.net.set_marginalization_idx([])

                    outs.append((ll_joint - ll_marginal).cpu()[0])

                ll_cond.append(np.concatenate(outs, axis=0))

            ll_cond = np.expand_dims(np.concatenate(ll_cond, axis=0), axis=-1)
            res[key] = (ll_cond,)

        return res

    def predict_mpe(self, x_, y_empty, batch_size=256):
        res = {}

        for key, val in x_.items():
            x, y = self.prepare_input(x_[key][:, :, -1], y_empty[key])
            self.net.set_marginalization_idx(list(range(x.shape[1], x.shape[1] + y.shape[1])))

            mpes, mpes_r = [], []
            idx_batches = torch.arange(x.shape[0], device=device).split(batch_size)
            for i, idx in enumerate(idx_batches):
                batch_x, batch_y = x[idx, :].detach(), y[idx, :].detach()
                mpe = self.net.mpe(x=torch.cat([batch_x, batch_y], dim=1))[:, x.shape[1]:]

                mpe_ = torch.complex(mpe[..., 0], mpe[..., 1]).reshape(
                    (mpe.shape[0], self.amt_prediction_windows, -1)).swapaxes(-1, -2)
                mpe_r = self.stft_module(mpe_, reverse=True)

                mpes.append(mpe.detach().cpu())
                mpes_r.append(mpe_r.detach().cpu())

            self.net.set_marginalization_idx([])
            res[key] = (np.concatenate(mpes, axis=0), np.concatenate(mpes_r, axis=0))

        return res

    def parameters(self):
        return self.net.parameters()

    # TODO: Now, with the config object, it would probably be easier just saving
    #  the config object together with the model
    def save(self, filepath):
        """
        Saves a model to a given filepath
        """

        # Store model settings
        with open(f'{filepath}_wein_settings.pkl', 'wb') as f:
            pickle.dump({key: value for key, value in self.__dict__.items() if key != 'graph' and key != 'net'}, f)

        Graph.write_gpickle(self.graph, filepath + '_graph.pkl')
        torch.save(self.net, filepath + '_net.mdl')

    def load(self, filepath):
        """
        Loads a model from a given filepath
        """

        # Load model settings
        with open(f'{filepath}_wein_settings.pkl', 'rb') as f:
            attributes = pickle.load(f)

            for attr, val in attributes.items():
                setattr(self, attr, val)

        self.graph = Graph.read_gpickle(filepath + '_graph.pkl')
        self.net = torch.load(filepath + '_net.mdl')

    def create_net(self):
        if self.config.structure['type'] == 'binary-trees':
            self.graph = Graph.random_binary_trees(num_var=self.config.input_size, depth=self.config.structure['depth'],
                                                   num_repetitions=self.config.structure['num_repetitions'])

        args = EinsumNetwork.Args(
            num_var=self.config.input_size,
            num_dims=2,
            num_classes=1,
            num_sums=self.config.K,
            num_input_distributions=self.config.K,
            exponential_family=self.config.exponential_family,
            exponential_family_args=self.config.exponential_family_args,
            online_em_frequency=self.config.online_em_frequency,
            online_em_stepsize=self.config.online_em_stepsize)

        self.net = EinsumNetwork.EinsumNetwork(self.graph, args)
        self.net.initialize()
        self.net.to(device)
