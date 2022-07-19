from .spectral_rnn.spectral_rnn import SpectralRNN, SpectralRNNConfig
from .spectral_rnn.manifold_optimization import ManifoldOptimizer
from .spectral_rnn.cgRNN import clip_grad_value_complex_
from .transformer import TransformerPredictor, TransformerConfig
from .maf import MAFEstimator
from .cwspn import CWSPN, CWSPNConfig
from .model import Model

import numpy as np
import torch
import torch.nn as nn

# Use GPU if avaiable
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PWN(Model):

    def __init__(self, s_config: SpectralRNNConfig, c_config: CWSPNConfig, train_spn_on_gt=True,
                 train_spn_on_prediction=False, train_rnn_w_ll=False, weight_mse_by_ll=None, always_detach=False,
                 westimator_early_stopping=5, step_increase=False, westimator_stop_threshold=.5,
                 westimator_final_learn=2, ll_weight=0.5, ll_weight_inc_dur=20, use_transformer=False, use_maf=False,
                 smape_target=False):

        assert train_spn_on_gt or train_spn_on_prediction
        assert not train_rnn_w_ll or train_spn_on_gt

        self.srnn = SpectralRNN(s_config) if not use_transformer else TransformerPredictor(
            TransformerConfig(normalize_fft=True, window_size=s_config.window_size,
                              fft_compression=s_config.fft_compression))
        self.westimator = CWSPN(c_config) if not use_maf else MAFEstimator()

        self.train_spn_on_gt = train_spn_on_gt
        self.train_spn_on_prediction = train_spn_on_prediction
        self.train_rnn_w_ll = train_rnn_w_ll
        self.weight_mse_by_ll = weight_mse_by_ll
        self.always_detach = always_detach

        self.westimator_early_stopping = westimator_early_stopping
        self.westimator_stop_threshold = westimator_stop_threshold
        self.westimator_final_learn = westimator_final_learn
        self.ll_weight = ll_weight
        self.ll_weight_inc_dur = ll_weight_inc_dur
        self.step_increase = step_increase
        self.use_transformer = use_transformer
        self.use_maf = use_maf
        self.smape_target = smape_target

    def train(self, x_in, y_in, val_x, val_y, embedding_sizes, batch_size=256, epochs=70, lr=0.004, lr_decay=0.97):

        # TODO: Adjustment for complex optimization, needs documentation
        if self.srnn.config.rnn_layer_config.use_cg_cell:
            lr /= 4

        if type(self.srnn) == TransformerPredictor:
            lr /= 10

        # Model is trained jointly on all groups
        x_ = np.concatenate(list(x_in.values()), axis=0)
        y_ = np.concatenate(list(y_in.values()), axis=0)[:, :, -1]

        if self.srnn.final_amt_pred_samples is None:
            self.srnn.final_amt_pred_samples = next(iter(y_in.values())).shape[1]

        # Init SRNN
        x = torch.from_numpy(x_).float().to(device)
        y = torch.from_numpy(y_).float().to(device)
        self.srnn.config.embedding_sizes = embedding_sizes
        self.srnn.build_net()

        self.westimator.stft_module = self.srnn.net.stft

        # Init westimator
        westimator_x_prototype, westimator_y_prototype = self.westimator.prepare_input(x[:1, :, -1], y[:1])
        self.westimator.input_sizes = westimator_x_prototype.shape[1], westimator_y_prototype.shape[1]
        self.westimator.create_net()

        prediction_loss = lambda error: error.mean()
        ll_loss = lambda out: -1 * torch.logsumexp(out, dim=1).mean()
        if self.smape_target:
            smape_adjust = 2  # Move all values into the positive space
            p_base_loss = lambda out, label: 2 * (torch.abs(out - label) /
                                                  (torch.abs(out + smape_adjust) +
                                                   torch.abs(label + smape_adjust))).mean(axis=1)
        # MSE target
        else:
            p_base_loss = lambda out, label: nn.MSELoss(reduction='none')(out, label).mean(axis=1)

        srnn_parameters = list(self.srnn.net.parameters())
        westimator_parameters = self.westimator.parameters()

        amt_param = sum([p.numel() for p in self.srnn.net.parameters()])
        amt_param_w = sum([p.numel() for p in self.westimator.parameters()])

        srnn_optimizer = ManifoldOptimizer(srnn_parameters, lr, torch.optim.RMSprop, alpha=0.9) \
            if type(self.srnn.config) == SpectralRNNConfig and self.srnn.config.rnn_layer_config.use_cg_cell \
            else torch.optim.RMSprop(srnn_parameters, lr=lr, alpha=0.9)
        westimator_optimizer = torch.optim.Adam(westimator_parameters, lr=1e-4)

        if self.train_rnn_w_ll:
            current_ll_weight = 0
            ll_weight_history = []
            ll_weight_increase = self.ll_weight / self.ll_weight_inc_dur
        elif self.train_spn_on_prediction:
            def ll_loss_pred(out, error):
                return (-1 * torch.logsumexp(out, dim=1) * (error ** -2)).mean() * 1e-4

        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=srnn_optimizer, gamma=lr_decay)

        westimator_losses = []
        srnn_losses = []
        srnn_losses_p = []
        srnn_losses_ll = []

        stop_cspn_training = False
        westimator_patience_counter = 0
        westimator_losses_epoch = []

        self.srnn.net.train()

        if hasattr(self.westimator, 'spn'):
            self.westimator.spn.train()
            self.westimator.weight_nn.train()
        else:
            self.westimator.model.train()

        if self.srnn.config.use_cached_predictions:
            import pickle
            with open('srnn_train.pkl', 'rb') as f:
                all_predictions, all_f_cs = pickle.load(f)

                all_predictions = torch.from_numpy(all_predictions).to(device)
                all_f_cs = torch.from_numpy(all_f_cs).to(device)

        val_errors = []
        print(f'Starting Training of {self.identifier} model')
        for epoch in range(epochs):
            idx_batches = torch.randperm(x.shape[0], device=device).split(batch_size)

            if self.train_rnn_w_ll:
                ll_weight_history.append(current_ll_weight)

            srnn_loss_p_e = 0
            srnn_loss_ll_e = 0

            pi = torch.tensor(np.pi)
            srnn_loss_e = 0
            westimator_loss_e = 0
            for i, idx in enumerate(idx_batches):
                batch_x, batch_y = x[idx].detach().clone(), y[idx, :].detach().clone()
                batch_westimator_x, batch_westimator_y = self.westimator.prepare_input(batch_x[:, :, -1], batch_y)

                if self.srnn.config.use_cached_predictions:
                    batch_p = all_predictions.detach().clone()[idx]
                    batch_fc = all_f_cs.detach().clone()[idx]

                if self.train_spn_on_gt:
                    westimator_optimizer.zero_grad()
                    if not stop_cspn_training or epoch >= epochs - self.westimator_final_learn:
                        out_w, _ = self.call_westimator(batch_westimator_x, batch_westimator_y)

                        if hasattr(self.westimator, 'spn'):
                            gt_ll = out_w
                            westimator_loss = ll_loss(gt_ll)
                            westimator_loss.backward()
                            westimator_optimizer.step()
                        else:
                            if self.westimator.use_made:
                                raise NotImplementedError  # MADE not implemented here

                            u, log_det = out_w

                            negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
                            negloglik_loss += 0.5 * self.westimator.final_input_sizes * np.log(2 * pi)
                            negloglik_loss -= log_det
                            negloglik_loss = torch.mean(negloglik_loss)

                            negloglik_loss.backward()
                            westimator_loss = negloglik_loss.item()
                            westimator_optimizer.step()
                            westimator_optimizer.zero_grad()

                    else:
                        westimator_loss = westimator_losses_epoch[-1]

                    westimator_loss_e += westimator_loss.detach()

                # Also zero grads for westimator, s.t. old grads dont influence the optimization
                srnn_optimizer.zero_grad()
                westimator_optimizer.zero_grad()

                if not self.srnn.config.use_cached_predictions:
                    prediction, f_c = self.srnn.net(batch_x, batch_y, return_coefficients=True)
                else:
                    prediction, f_c = batch_p, batch_fc

                if self.westimator.use_stft:
                    prediction_ll, w_in = self.call_westimator(batch_westimator_x, f_c.reshape((f_c.shape[0], -1))
                                        if self.train_rnn_w_ll and not self.always_detach else f_c.reshape((f_c.shape[0], -1)).detach())
                else:
                    prediction_ll, w_in = self.call_westimator(batch_westimator_x, prediction
                                        if self.train_rnn_w_ll and not self.always_detach else prediction.detach())

                error = p_base_loss(prediction, batch_y)
                p_loss = prediction_loss(error)

                if self.train_rnn_w_ll:
                    l_loss = ll_loss(prediction_ll)

                    if self.weight_mse_by_ll is None:
                        srnn_loss = (1 - current_ll_weight) * p_loss + current_ll_weight * l_loss
                    else:
                        local_ll = torch.logsumexp(prediction_ll, dim=1)
                        local_ll = local_ll - local_ll.max()  # From 0 to -inf
                        local_ll = local_ll / local_ll.min()  # From 0 to 1 -> low LL is 1, high LL is 0: Inverse Het
                        local_ll = local_ll / local_ll.mean()  # Scale it to mean = 1

                        if self.weight_mse_by_ll == 'het':
                            # Het: low LL is 0, high LL is 1
                            local_ll = local_ll.max() - local_ll

                        srnn_loss = p_loss * (self.ll_weight - current_ll_weight) + \
                                    current_ll_weight * (error * local_ll).mean()
                else:
                    srnn_loss = p_loss
                    l_loss = 0

                if not self.srnn.config.use_cached_predictions:
                    srnn_loss.backward()

                    if self.srnn.config.clip_gradient_value > 0:
                        clip_grad_value_complex_(srnn_parameters, self.srnn.config.clip_gradient_value)

                    srnn_optimizer.step()

                if self.train_spn_on_prediction:
                    if hasattr(self.westimator, 'spn'):
                        westimator_loss = ll_loss_pred(prediction_ll, error.detach())

                        westimator_loss.backward()
                        westimator_optimizer.step()
                    else:
                        if type(prediction_ll) == tuple:
                            u, log_det = prediction_ll

                            negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
                            negloglik_loss += 0.5 * self.westimator.final_input_sizes * np.log(2 * pi)
                            negloglik_loss -= log_det
                            negloglik_loss = torch.mean(negloglik_loss * (error ** -2)) * 1e-4

                        else:
                            mu, logp = torch.chunk(prediction_ll, 2, dim=1)
                            u = (w_in - mu) * torch.exp(0.5 * logp)

                            negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
                            negloglik_loss += 0.5 * w_in.shape[1] * np.log(2 * pi)
                            negloglik_loss -= 0.5 * torch.sum(logp, dim=1)

                            negloglik_loss = torch.mean(negloglik_loss)

                        negloglik_loss.backward()
                        westimator_loss = negloglik_loss.item()
                        westimator_optimizer.step()

                    westimator_loss_e += westimator_loss.detach()

                l_loss = l_loss.detach() if not type(l_loss) == int else l_loss
                srnn_loss_p_e += p_loss.item()
                srnn_loss_ll_e += l_loss
                srnn_loss_e += srnn_loss.item()

                westimator_losses.append(westimator_loss.detach().cpu().numpy())
                srnn_losses.append(srnn_loss.detach().cpu().numpy())
                srnn_losses_p.append(p_loss.detach().cpu().numpy())
                srnn_losses_ll.append(l_loss)

                if (i + 1) % 10 == 0:
                    print(f'Epoch {epoch + 1} / {epochs}: Step {(i + 1)} / {len(idx_batches)}. '
                          f'Avg. WCSPN Loss: {westimator_loss_e / (i + 1)} '
                          f'Avg. SRNN Loss: {srnn_loss_e / (i + 1)}')

            lr_scheduler.step()

            if epoch < self.ll_weight_inc_dur and self.train_rnn_w_ll:
                if self.step_increase:
                    current_ll_weight = 0
                else:
                    current_ll_weight += ll_weight_increase
            elif self.train_rnn_w_ll:
                current_ll_weight = self.ll_weight

            westimator_loss_epoch = westimator_loss_e / len(idx_batches)
            srnn_loss_epoch = srnn_loss_e / len(idx_batches)
            print(f'Epoch {epoch + 1} / {epochs} done.'
                  f'Avg. WCSPN Loss: {westimator_loss_epoch} '
                  f'Avg. SRNN Loss: {srnn_loss_epoch}')

            print(f'Avg. SRNN-Prediction-Loss: {srnn_loss_p_e / len(idx_batches)}')
            print(f'Avg. SRNN-LL-Loss: {srnn_loss_ll_e / len(idx_batches)}')

            if len(westimator_losses_epoch) > 0 and not stop_cspn_training and \
                    not westimator_loss_epoch < westimator_losses_epoch[-1] - self.westimator_stop_threshold and not \
                    self.train_spn_on_prediction:
                westimator_patience_counter += 1

                print(f'Increasing patience counter to {westimator_patience_counter}')

                if westimator_patience_counter >= self.westimator_early_stopping:
                    stop_cspn_training = True
                    print('WCSPN training stopped!')

            else:
                westimator_patience_counter = 0

            westimator_losses_epoch.append(westimator_loss_epoch)

            if False and epoch % 3 == 0:
                pred_val, _ = self.predict({key: x for key, x in val_x.items() if len(x) > 0}, mpe=False)
                self.srnn.net.train()
                self.westimator.spn.train()
                self.westimator.weight_nn.train()

                val_mse = np.mean([((p - val_y[key][:, :, -1]) ** 2).mean() for key, p in pred_val.items()])
                val_errors.append(val_mse)

        if not self.srnn.config.use_cached_predictions:
            predictions = []
            f_cs = []

            with torch.no_grad():
                print(x.shape[0] // batch_size + 1)
                for i in range(x.shape[0] // batch_size + 1):
                    batch_x, batch_y = x[i * batch_size:(i + 1) * batch_size], \
                                       y[i * batch_size:(i + 1) * batch_size]
                    prediction, f_c = self.srnn.net(batch_x, batch_y, return_coefficients=True)

                    predictions.append(prediction)
                    f_cs.append(f_c)

            predictions = torch.cat(predictions, dim=0).detach().cpu().numpy()
            f_cs = torch.cat(f_cs, dim=0).detach().cpu().numpy()

            import pickle
            with open('srnn_train.pkl', 'wb') as f:
                pickle.dump((predictions, f_cs), f)

        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size': 48, 'figure.figsize': (60, 40)})
        index = list(range(len(westimator_losses)))
        plt.ylabel('LL')
        plt.plot(index, westimator_losses, label='WCSPN-Loss (Negative LL)', color='blue')
        plt.plot(index, srnn_losses_ll, label='SRNN-Loss (Negative LL)', color='green')
        plt.legend(loc='upper right')

        ax2 = plt.twinx()
        ax2.set_ylabel('MSE', color='red')
        ax2.plot(index, srnn_losses, label='SRNN-Loss Total', color='magenta')
        ax2.plot(index, srnn_losses_p, label='SRNN-Loss Prediction', color='red')
        ax2.legend(loc='upper left')

        plt.savefig('res/plots/0_PWN_Training_losses')

        plt.clf()
        plt.plot(val_errors)
        plt.savefig('res/plots/0_PWN_Val_MSE')
        print(val_errors)

        if self.train_rnn_w_ll:
            plt.clf()
            plt.plot(ll_weight_history)
            plt.ylabel('SRNN LL-Loss Weight (percentage of total loss)')
            plt.title('LL Weight Warmup')
            plt.savefig('res/plots/0_PWN_LLWeightWarmup')

    @torch.no_grad()
    def predict(self, x, batch_size=1024, pred_label='', mpe=False):
        predictions, f_c = self.srnn.predict(x, batch_size, pred_label=pred_label, return_coefficients=True)

        x_ = {key: x_[:, :, -1] for key, x_ in x.items()}

        f_c_ = {key: f.reshape((f.shape[0], -1)) for key, f in f_c.items()}

        if self.westimator.use_stft:
            ll = self.westimator.predict(x_, f_c_, stft_y=False, batch_size=batch_size)
        else:
            ll = self.westimator.predict(x_, predictions, batch_size=batch_size)

        if mpe:
            y_empty = {key: np.zeros((x[key].shape[0], self.srnn.net.amt_prediction_samples)) for key in x.keys()}
            predictions_mpe = self.westimator.predict_mpe({key: x_.copy() for key, x_ in x.items()},
                                                           y_empty, batch_size=batch_size)
            lls_mpe = self.westimator.predict(x_, {key: v[0] for key, v in predictions_mpe.items()},
                                              stft_y=False, batch_size=batch_size)

            # predictions, likelihoods, likelihoods_mpe, predictions_mpe
            return predictions, ll, lls_mpe, {key: v[1] for key, v in predictions_mpe.items()}

        else:
            return predictions, ll

    def save(self, filepath):
        self.srnn.save(filepath)
        self.westimator.save(filepath)

    def load(self, filepath):
        self.srnn.load(filepath)
        self.westimator.load(filepath)
        self.westimator.stft_module = self.srnn.net.stft

    def call_westimator(self, x, y):

        y_ = torch.stack([y.real, y.imag], dim=-1) if torch.is_complex(y) else y

        if hasattr(self.westimator, 'spn'):
            sum_params, leaf_params = self.westimator.weight_nn(x.reshape((x.shape[0], x.shape[1] *
                                                                           (2 if self.westimator.use_stft else 1),)))
            self.westimator.args.param_provider.sum_params = sum_params
            self.westimator.args.param_provider.leaf_params = leaf_params

            return self.westimator.spn(y_), y_
        else:
            val_in = torch.cat([x, y_], dim=1).reshape((x.shape[0], -1))
            return self.westimator.model(val_in), val_in
