from model import PWN
from model.spectral_rnn import SpectralRNNConfig
from model.cwspn import CWSPNConfig
from util.plot import cmp_plot
from data_source import Mackey, ReadPowerPKL
from preprocessing import ZScoreNormalization

import torch
import numpy as np
import matplotlib.pyplot as plt

import pickle

plot_base_path = 'res/plots/'
model_base_path = 'res/models/'
experiments_base_path = 'res/experiments/'

config = SpectralRNNConfig()
config.normalize_fft = True
config.use_add_linear = False
config.rnn_layer_config.use_gated = True
config.rnn_layer_config.use_cg_cell = False
config.rnn_layer_config.use_residual = True
config.rnn_layer_config.learn_hidden_init = False
config.rnn_layer_config.use_linear_projection = True

config.window_size = 96
config.fft_compression = 4

config.use_cached_predictions = False

config_c = CWSPNConfig()

with open('./res/data_cache_pwr_long.pkl', 'rb') as f:
    train_x, train_y, test_x, test_y, train_y_values, test_y_values, \
    column_names, embedding_sizes, preprocessing = pickle.load(f)

# tmax = 1024 * 60
use_pred_as_input = True

former_context_timespan = preprocessing.context_timespan
former_prediction_timespan = preprocessing.prediction_timespan
preprocessing = ZScoreNormalization((0,), 2, 1, [True, True, True, False], context_timespan=15 * 96,
                                    prediction_timespan=int(40 * 96), timespan_step=96,
                                    single_group=False, multivariate=False, retail=False)

model = PWN(config, config_c, train_spn_on_gt=False, train_spn_on_prediction=True, ll_weight=1e-5,
                        ll_weight_inc_dur=50, westimator_early_stopping=8)

model_filepath = f'{model_base_path}07_19_2022_15_43_15__PWN-ReadPowerPKL'  # change the file name after executing training.py
model.load(model_filepath)

x_raw, y_raw, _, _, _, _, _ = preprocessing.apply(ReadPowerPKL().data)

for key in [2018.0]:  # x_raw.keys():
    x = {'All': x_raw[key][-2:-1]}
    y = {'All': y_raw[key][-2:-1]}

    model.srnn.final_amt_pred_samples = preprocessing.prediction_timespan
    model.srnn.net.amt_prediction_samples = preprocessing.prediction_timespan
    model.srnn.net.amt_prediction_windows = model.srnn.net.stft(torch.from_numpy(y['All'][..., -1]).to('cuda')).shape[-1]

    predictions, f_c = model.srnn.predict(x, 256, pred_label='Mackey_LT', return_coefficients=True)

    lls = []
    mses = []
    lls_mpe = []
    all_window_lls = []
    cspn_width = model.westimator.amt_prediction_windows
    for i in range(preprocessing.prediction_timespan // former_prediction_timespan):
        time_start = i * former_prediction_timespan
        time_start_input = (i - 1) * former_prediction_timespan
        time_start_predictions = (i - former_context_timespan // former_prediction_timespan) * former_prediction_timespan

        spectral_start = i * (cspn_width - 1)

        local_p = predictions['All'][:, time_start:time_start + former_prediction_timespan]
        local_y = y['All'][:, time_start:time_start + former_prediction_timespan, -1]

        if use_pred_as_input:
            if i == 0:
                x_ = {'All': x['All'][..., -1]}
            elif i < former_context_timespan / former_prediction_timespan:
                x_ = {'All': np.concatenate([
                    x['All'][:, time_start:, -1],
                    predictions['All'][:, :time_start]], axis=1)}
            else:
                x_ = {'All': predictions['All'][:, time_start_predictions:time_start_predictions + former_context_timespan]}
        else:
            if i == 0:
                x_ = {'All': x['All'][..., -1]}
            elif i < former_context_timespan / former_prediction_timespan:
                x_ = {'All': np.concatenate([
                    x['All'][:, time_start:, -1],
                    y['All'][:, :time_start, -1]], axis=1)}
            else:
                x_ = {'All': y['All'][:, time_start_predictions:time_start_predictions + former_context_timespan, -1]}

        f_c_ = {'All': f_c['All'][:, spectral_start: spectral_start + cspn_width].reshape((f_c['All'].shape[0], -1))}

        ll = model.westimator.predict(x_, f_c_, stft_y=False)
        lls.append(ll['All'][0][0])

        mpe = model.westimator.predict_mpe({key: np.expand_dims(x, axis=-1) for key, x in x_.items()},
                                           {'All': np.zeros_like(local_y)})
        ll_mpe = model.westimator.predict(x_, {key: v[0] for key, v in mpe.items()}, stft_y=False)
        lls_mpe.append(ll_mpe['All'][0][0])

        window_lls = model.westimator.predict_ll_per_window(x_, f_c_, stft_y=False)
        all_window_lls.append(window_lls['All'][0])

        mse = (local_p - local_y) ** 2
        mses.append(mse.mean())

    all_window_lls = [window for pred in all_window_lls for window in pred]

    pred_ll = torch.zeros(preprocessing.prediction_timespan).cuda()

    all_window_lls_merged = []
    for i, window in enumerate(all_window_lls):
        if 0 < i < len(all_window_lls) - 1 and (i + 1) % cspn_width == 0:
            continue

        if 0 < i < len(all_window_lls) - 1 and i % cspn_width == 0:
            all_window_lls_merged.append((window + all_window_lls[i - 1]) / 2)
        else:
            all_window_lls_merged.append(window)

    for i, window in enumerate(all_window_lls_merged):
        w = window[0] * model.srnn.net.stft.window

        if i == 0:
            pred_ll[:config.step_width] += w[config.step_width:]
        elif i == len(all_window_lls_merged) - 1:
            pred_ll[-config.step_width:] += w[:config.step_width]
        else:
            pred_ll[(i - 1) * config.step_width:(i + 1) * config.step_width] += w

    # We use the LR-Ratio here, based on train min / max ll
    min_ll = -482.12857
    max_ll = 264.2689
    pred_lr = -2 * (pred_ll - max_ll)
    max_lr_span = -2 * (min_ll - max_ll)

    # Taking the square root here is the same as taking the square root of both likelihood ratios
    confidence = .5 * torch.sqrt(pred_lr / max_lr_span).detach().cpu().numpy()

    plt.plot(lls_mpe)
    plt.title('MPE LLs')
    plt.show()
    plt.clf()

    plt.plot(confidence)
    plt.title(f'Pred Confidence: {key}')
    plt.show()
    plt.clf()

    plt.plot(lls, label='LLs', color='blue')
    ax2 = plt.twinx()
    ax2.plot(mses, label='MSEs', color='red')
    ax2.legend('upper right')
    plt.legend('upper left')
    plt.title(f'LLs vs MSE: {key}')
    plt.show()
    plt.clf()

    llrs_lower, llrs_upper = predictions['All'][0] - confidence, predictions['All'][0] + confidence

    center = preprocessing.context_timespan // 2
    plt.arrow(center - 200, llrs_lower.min() - 0.05, -center + 230, 0, length_includes_head=True,
              head_starts_at_zero=False, color='grey', head_width=0.25, head_length=45)
    plt.arrow(center + 200, llrs_lower.min() - 0.05, center - 230, 0, length_includes_head=True,
              head_starts_at_zero=False, color='grey', head_width=0.25, head_length=45)
    plt.text(center, llrs_lower.min() - 0.05, 'Context', ha='center', va='center',
             bbox=dict(facecolor='none', edgecolor='none'), fontdict={'color': 'grey'})

    center = preprocessing.context_timespan + former_prediction_timespan // 2
    plt.text(preprocessing.context_timespan + former_prediction_timespan // 2, llrs_lower.min() - 0.05, 'Standard\nPrediction',
             ha='center', va='center',
             bbox=dict(facecolor='none', edgecolor='none'), fontdict={'color': 'grey'})

    center = preprocessing.context_timespan + former_prediction_timespan + \
             (model.srnn.final_amt_pred_samples - former_prediction_timespan) // 2
    plt.arrow(center - 535, llrs_lower.min() - 0.05, -(model.srnn.final_amt_pred_samples - former_prediction_timespan) // 2 + 565, 0,
              length_includes_head=True, head_starts_at_zero=False, color='grey', head_width=0.25, head_length=45)
    plt.arrow(center + 535, llrs_lower.min() - 0.05, (model.srnn.final_amt_pred_samples - former_prediction_timespan) // 2 - 565, 0,
              length_includes_head=True, head_starts_at_zero=False, color='grey', head_width=0.25, head_length=45)
    plt.text(center, llrs_lower.min() - 0.05, 'Long-Range Prediction', ha='center', va='center',
             bbox=dict(facecolor='none', edgecolor='none'), fontdict={'color': 'grey'})
    plt.vlines([preprocessing.context_timespan, preprocessing.context_timespan + former_prediction_timespan], ymin=-30, ymax=30,
               colors='grey', linestyles='dashed',
               linewidth=5.5)

    plt.plot(range(preprocessing.context_timespan + model.srnn.final_amt_pred_samples),
             np.concatenate([x['All'][0, :, -1], y['All'][0, :, -1]]), label='Ground Truth', color='C1', linewidth=5.5)
    plt.plot(range(preprocessing.context_timespan, preprocessing.context_timespan + model.srnn.final_amt_pred_samples),
             predictions['All'][0], label='Prediction', color='C0', linestyle='dashed', linewidth=5.5)
    plt.fill_between(range(preprocessing.context_timespan,
                           preprocessing.context_timespan + model.srnn.final_amt_pred_samples),
                     llrs_lower, llrs_upper, alpha=0.2, facecolor='green', label='LLRS')
    plt.ylim([llrs_lower.min() - 1.2, llrs_upper.max() + 0.2])
    leg = plt.legend(loc='upper left')

    for legobj in leg.legendHandles:
        legobj.set_linewidth(5.5)

    plt.xlabel('Time')
    plt.ylabel('Power (z-score normalized)')
    plt.savefig(f'Uncertainty_Power.pdf', bbox_inches='tight')
    plt.show()
    plt.clf()
