from ..model import Model
from .model import DeepARNet, loss_fn, accuracy_RMSE_

import torch
import numpy as np

import pickle
import math

# Use GPU if avaiable
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class DeepARPredictor(Model):

    def __init__(self):
        self.net = None
        self.use_cached_predictions = False

        self.build_net()

    def train(self, x_in, y_in, val_x, val_y, embedding_sizes, batch_size=256, epochs=10, lr=0.001, lr_decay=0.97):
        """
        Trains the model with the given data_source
        """

        # Model is trained jointly on all groups
        x_ = np.concatenate(list(x_in.values()), axis=0)[..., -1:]
        y_ = np.concatenate(list(y_in.values()), axis=0)[..., -1:]

        all_data = np.concatenate([x_, y_], axis=1)

        x = torch.from_numpy(np.pad(all_data[:, :-1], [(0, 0), (1, 0), (0, 0)])).float().to(device)
        y = torch.from_numpy(all_data[..., -1]).float().to(device)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.net.params.learning_rate)

        self.net.train()

        for epoch in range(self.net.params.num_epochs):
            loss_epoch = np.zeros(len(x))

            idx_batches = torch.randperm(x.shape[0], device=device).split(batch_size)
            for i, idx in enumerate(idx_batches):
                batch_x, batch_y = x[idx].detach().clone(), y[idx, :].detach().clone()

                # Drop last
                if len(batch_x) != batch_size:
                    continue

                optimizer.zero_grad()

                train_batch = batch_x.permute(1, 0, 2).to(torch.float32)  # not scaled
                labels_batch = batch_y.permute(1, 0).to(torch.float32)  # not scaled

                loss = torch.zeros(1, device=device)
                hidden = self.net.init_hidden(batch_size).to(device)
                cell = self.net.init_cell(batch_size).to(device)

                for t in range(self.net.params.train_window):
                    # if z_t is missing, replace it by output mu from the last time step
                    zero_index = (train_batch[t, :, 0] == 0)
                    if t > 0 and torch.sum(zero_index) > 0:
                        train_batch[t, zero_index, 0] = mu[zero_index]

                    mu, sigma, hidden, cell = self.net(train_batch[t].unsqueeze_(0).clone(), 0, hidden, cell)

                    loss += loss_fn(mu, sigma, labels_batch[t])

                loss.backward()
                optimizer.step()
                loss = loss.item() / self.net.params.train_window  # loss per timestep
                loss_epoch[i] = loss

                if i % 10 == 0:
                    print(f'train_loss: {loss} batch: {i}')

            print(f'epoch {epoch} finished, loss {loss_epoch.mean()}')

        return loss_epoch

    def predict(self, x, batch_size=350, pred_label=None, return_coefficients=False, mpe=False):
        """
        Returns predictions for given data_source
        """

        if not self.use_cached_predictions:
            predictions = {}

            with torch.no_grad():
                for key, x_ in x.items():
                    x_t = torch.tensor(x_, device=device)[:, :, -1:]

                    predictions[key] = []
                    # idx_batches = torch.arange(x_.shape[0], device=device).split(batch_size)
                    i = 1
                    while i == 1:
                        i += 1
                    # for i, idx in enumerate(idx_batches):
                        batch_x = x_t.detach().clone()

                        test_batch = batch_x.permute(1, 0, 2).to(torch.float32).to(device)
                        # v_batch = v.to(torch.float32).to(params.device)
                        # labels = labels.to(torch.float32).to(device)
                        batch_size = test_batch.shape[1]
                        hidden = self.net.init_hidden(batch_size).to(device)
                        cell = self.net.init_cell(batch_size).to(device)

                        for t in range(self.net.params.test_predict_start):
                            # if z_t is missing, replace it by output mu from the last time step
                            zero_index = (test_batch[t, :, 0] == 0)
                            if t > 0 and torch.sum(zero_index) > 0:
                                test_batch[t, zero_index, 0] = mu[zero_index]

                            mu, sigma, hidden, cell = self.net(test_batch[t].unsqueeze(0), 0, hidden, cell)
                            # input_mu[:, t] = v_batch[:, 0] * mu + v_batch[:, 1]
                            # input_sigma[:, t] = v_batch[:, 0] * sigma

                        sample_mu, sample_sigma = self.net.test(
                            torch.cat([test_batch, torch.zeros((self.net.params.predict_steps, test_batch.shape[1],
                                                                test_batch.shape[2]), device=device)]),
                            0, 0, hidden, cell)

                        batch_preds = sample_mu.detach().cpu().numpy()
                        predictions[key].append(batch_preds)

                    predictions[key] = np.concatenate(predictions[key], axis=0)

            if pred_label is not None:
                with open(pred_label + '_deep_ar.pkl', 'wb') as f:
                     pickle.dump(predictions, f)

        else:
            assert pred_label is not None

            with open(pred_label + '_deep_ar.pkl', 'rb') as f:
                predictions = pickle.load(f)

        return predictions

    def build_net(self):
        # CARE: Hardcoded values!
        in_len = 20 * 24
        out_len = 2 * 24
        params = {
            "learning_rate": 1e-4,
            "batch_size": 256,
            "lstm_layers": 3,
            "num_epochs": 10,
            "train_window": int(in_len + out_len),
            "test_window":  int(in_len + out_len),
            "predict_start": int(in_len - 1),
            "test_predict_start": int(in_len - 1),
            "predict_steps": int(out_len),
            "num_class": 370,
            "cov_dim": 0,
            "lstm_hidden_dim": 172,
            "embedding_dim": 0,
            "sample_times": 200,
            "lstm_dropout": 0.1,
            "predict_batch": 256
        }

        self.net = DeepARNet(dotdict(params))
        self.net.to(device)
        amt_param = sum([p.numel() for p in self.net.parameters()])
        print(f'PARAMS: {amt_param}')

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
