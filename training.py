from data_source import BasicSelect, Mackey, ReadPowerPKL, ReadM4
from preprocessing import *
from model import WhittleRNN, PWN, PWNEM
from model.n_beats import NBeatsPredictor
from model.cwspn import CWSPN, CWSPNConfig
from model.wein import WEin, WEinConfig
from model.spectral_rnn import SpectralRNN, SpectralRNNConfig
from model.wein.EinsumNetwork.ExponentialFamilyArray import NormalArray, MultivariateNormalArray
from evaluation import *
from util.plot import plot_experiment, cmp_plot, ll_plot
from util.dataset import split_dataset
from util.store_experiments import save_experiment

import pickle
from datetime import datetime

import numpy as np

plot_base_path = 'res/plots/'
model_base_path = 'res/models/'
experiments_base_path = 'res/experiments/'

config = SpectralRNNConfig()
# config.normalize_fft = True
config.use_add_linear = False
config.rnn_layer_config.use_gated = True
config.rnn_layer_config.use_cg_cell = False
config.rnn_layer_config.use_residual = True
config.rnn_layer_config.learn_hidden_init = False
config.rnn_layer_config.use_linear_projection = True
config.rnn_layer_config.dropout = 0.1

config.hidden_dim = 128
config.rnn_layer_config.n_layers = 2

config.use_cached_predictions = False

config_w = WEinConfig()
config_w.exponential_family = MultivariateNormalArray  # NormalArray
config_w.window_level = False
config_w.mpe_prediction = False
config_w.structure = {'type': 'binary-trees', 'depth': 4, 'num_repetitions': 5}
config_w.exponential_family_args = {'min_var': 1e-4, 'max_var': 4.}
config_w.prepare_joint = False
config_w.K = 2

config_c = CWSPNConfig()
config_c.num_gauss = 2
config_c.num_sums = 4
config_c.rg_splits = 8
config_c.rg_split_recursion = 2
config_c.gauss_min_sigma = 1e-4
config_c.gauss_max_sigma = 1. * 4
config_c.use_rationals = True

manual_split = True

m4_key = 'Daily'
m4_settings = {
    'Hourly': {'window_size': 24, 'fft_compression': 2, 'context_timespan': int(20 * 24),
               'prediction_timespan': int(2 * 24), 'timespan_step': int(.5 * 24)},  # 700 Min Context
    'Daily': {'window_size': 14, 'fft_compression': 2, 'context_timespan': int(5 * 14),
              'prediction_timespan': int(1 * 14), 'timespan_step': int(.5 * 14)},  # 93 Min Context
    'Weekly': {'window_size': 14, 'fft_compression': 2, 'context_timespan': int(4.5 * 14),
               'prediction_timespan': int(14), 'timespan_step': int(.5 * 14)},  # 80 Min Context
    'Monthly': {'window_size': 18, 'fft_compression': 1, 'context_timespan': int(6 * 18),
                'prediction_timespan': int(1 * 18), 'timespan_step': int(.5 * 18)},  # 42 Min Context
    'Quarterly': {'window_size': 8, 'fft_compression': 1, 'context_timespan': int(4 * 8),
                  'prediction_timespan': int(1 * 8), 'timespan_step': int(.5 * 8)},  # 16 Min Context
    'Yearly': {'window_size': 6, 'fft_compression': 1, 'context_timespan': int(4 * 6),
               'prediction_timespan': int(1 * 6), 'timespan_step': int(.5 * 6)}  # 13 Min Context
}

config.window_size = 96  # m4_settings[m4_key]['window_size']
config.fft_compression = 4  # m4_settings[m4_key]['fft_compression']

# Each experiment is a tuple (data_source, preprocessing, model, evaluation_metrics, plots, load, use_cached)
experiments = [
    (ReadPowerPKL(), ZScoreNormalization((0,), 3, 2, [True, True, True, False], min_group_size=0,
                                         context_timespan=15 * 96, prediction_timespan=int(1.5 * 96),
                                         timespan_step=96, single_group=False, multivariate=False, retail=False),
     PWN(config, config_c, train_spn_on_gt=False, train_spn_on_prediction=True, train_rnn_w_ll=False,
            always_detach=True),
     [CorrelationError(), MAE(), MSE(), RMSE()],
     {'train': False, 'reversed': True, 'll': False, 'single_ll': False, 'mpe': False},
     None, False)]

for i, (data_source, preprocessing, model, evaluation_metrics, plots, load, use_cached) in enumerate(experiments):
    print(f'Starting experiment {i}')

    start_time = datetime.now()
    model_name = f'{model.identifier}-{data_source.get_identifier()}'
    experiment_id = f'{start_time.strftime("%m_%d_%Y_%H_%M_%S")}__{model_name}'

    if not use_cached:
        print('Getting fresh data...')
        data = data_source.data

        print(f'Received {len(data)} rows of data')
        print(f'--- Query finished after {(datetime.now() - start_time).microseconds} microseconds ---')

        train_x, train_y, test_x, test_y, column_names, embedding_sizes, last_sequence_labels = \
            preprocessing.apply(data, data_source, manual_split=True)

        train_y_values = {key: y[..., -1] if len(y) > 0 else y for key, y in train_y.items()}
        test_y_values = {key: y[..., -1] if len(y) > 0 else y for key, y in test_y.items()}

        with open('data_cache.pkl', 'wb') as f:
            pickle.dump((train_x, train_y, test_x, test_y, train_y_values, test_y_values,
                         column_names, embedding_sizes, preprocessing), f)

    else:
        print('Loading cached data...')

        with open('data_cache.pkl', 'rb') as f:
            train_x, train_y, test_x, test_y, train_y_values, test_y_values, \
            column_names, embedding_sizes, preprocessing = pickle.load(f)

    print(f'--- Preprocessing finished after {(datetime.now() - start_time).microseconds} microseconds ---')

    if load is None:
        print('Training model...')
        model.train(train_x, train_y, test_x, test_y, embedding_sizes)
        print(f'--- Training finished after {(datetime.now() - start_time).microseconds} microseconds ---')

    else:
        print('Loading model...')
        model_filepath = f'{model_base_path}{load}'
        model.load(model_filepath)

    model_filepath = f'{model_base_path}{experiment_id}'
    model.save(model_filepath)
    print(f'Model saved to {model_filepath}')

    test_x = {key: val for key, val in test_x.items() if len(val) > 0}
    test_y = {key: val for key, val in test_y.items() if len(val) > 0}
    test_y_values = {key: val for key, val in test_y_values.items() if len(val) > 0}

    predictions = model.predict(test_x, mpe=plots['mpe'], pred_label='test_pred')
    print('Test predictions done')

    if plots['train']:
        train_predictions = model.predict(train_x, mpe=plots['mpe'], pred_label='train_pred')

        print('Train predictions done')

    if type(predictions) is tuple:
        if plots['mpe']:
            predictions, likelihoods, likelihoods_mpe, predictions_mpe = predictions

            if plots['train']:
                train_predictions, train_likelihoods, \
                    train_likelihoods_mpe, train_predictions_mpe = train_predictions
        else:
            predictions, likelihoods = predictions

            if plots['train']:
                train_predictions, train_likelihoods = train_predictions

        if plots['ll']:
            likelihoods_gt = model.westimator.predict({key: x_[:, :, -1] for key, x_ in test_x.items()}, test_y_values)
            ll_length = len(next(iter(likelihoods_gt.values())))

        if plots['train']:
            likelihoods_gt_train = model.westimator.predict({key: x_[:, :, -1] for key, x_ in train_x.items()}, train_y_values)

    else:
        likelihoods = None
        train_likelihoods = None

    index_test = {key: np.array([list(range(p.shape[-1]))]) for key, p in predictions.items()}

    if plots['train']:
        index_train = {key: np.array([list(range(p.shape[-1]))]) for key, p in train_predictions.items()}

    if plots['reversed']:
        predictions_reversed = preprocessing.reverse(predictions)
        test_y_values_reversed = preprocessing.reverse(test_y_values)

        if plots['train']:
            train_predictions_reversed = preprocessing.reverse(train_predictions)
            train_y_values_reversed = preprocessing.reverse(train_y_values)

    else:
        print('Note: As "reversed" is deactivated, '
              'MSE will be calculated on the preprocessed instead of the original values')

    plot_filepath_base = f'{plot_base_path}{experiment_id}'

    eval_results = {}
    for eval_metric in evaluation_metrics:
        metric_name = eval_metric.__class__.__name__
        result_test = eval_metric.calculate(predictions_reversed, test_y_values_reversed, likelihoods) \
            if plots['reversed'] else eval_metric.calculate(predictions, test_y_values, likelihoods)

        eval_results[metric_name] = result_test
        # print(f'{metric_name}: {result_test}')

        if type(eval_metric) in (MSE, SMAPE) and plots['mpe']:
            try:
                if reversed:
                    mpe_pred_reversed = preprocessing.reverse(predictions_mpe)

                    mpe_error = eval_metric.calculate(mpe_pred_reversed, test_y_values_reversed, likelihoods)
                else:
                    mpe_error = eval_metric.calculate(predictions_mpe, test_y_values, likelihoods)

                print(f'{metric_name} MPE Average: {sum([x for x in mpe_error.values()]) / len(mpe_error)}')

                if plots['train']:
                    if reversed:
                        mpe_pred_reversed_train = preprocessing.reverse(train_predictions_mpe)

                        mpe_error_train = eval_metric.calculate(mpe_pred_reversed_train,
                                                                train_y_values_reversed, None)
                    else:
                        mpe_error_train = eval_metric.calculate(train_predictions_mpe, train_y_values, None)

                    print(f'{metric_name} MPE Average Train: '
                          f'{sum([x for x in mpe_error_train.values()]) / len(mpe_error_train)}')

            except NameError:
                print('MPE not avaiable!')

        if type(result_test) is dict:
            print(f'{metric_name} Average: {sum([x for x in result_test.values()]) / len(result_test)}')
        else:
            print(f'{metric_name} Average: {result_test.mean()}')

            from collections.abc import Iterable

            if isinstance(result_test, Iterable):
                import matplotlib.pyplot as plt

                plt.clf()
                plt.plot(sorted(result_test))
                plt.savefig(f'{plot_filepath_base}_ce')
                plt.clf()

        if plots['train']:
            result_train = eval_metric.calculate(train_predictions_reversed, train_y_values_reversed,
                                                 train_likelihoods) if plots['reversed'] else \
                eval_metric.calculate(train_predictions, train_y_values, train_likelihoods)
            print(f'Train: {metric_name}: {result_train}')

            if type(result_test) is dict:
                print(f'Train {metric_name} Average: {sum([x for x in result_train.values()]) / len(result_test)}')
            else:
                print(f'Train {metric_name} Average: {result_train.mean()}')

    print(f'--- Evaluation finished after {(datetime.now() - start_time).microseconds} microseconds ---')

    rolling = 30

    if plots['single_ll']:
        print('Plotting single LL...')

        cmp_plot(f'{experiment_id}: Test_PredictionLL', f'{plot_filepath_base}_ll_test',
                 predictions, likelihoods, test_y_values, columns=6, rolling=2)
        cmp_plot(f'{experiment_id}: Test_GroundTruthLL', f'{plot_filepath_base}_ll_test_gt',
                 predictions, likelihoods_gt, test_y_values, columns=6, rolling=2)

        if plots['train']:
            cmp_plot(f'{experiment_id}: Train_PredictionLL', f'{plot_filepath_base}_ll_train',
                     train_predictions, train_likelihoods, train_y_values)
            cmp_plot(f'{experiment_id}: Train_GroundTruthLL', f'{plot_filepath_base}_ll_train_gt',
                     train_predictions, likelihoods_gt_train, train_y_values)

    if plots['ll'] or plots['mpe']:
        try:
            p_all = {'All': np.concatenate([p for p in predictions.values()], axis=0)}
            l_all = {'All': tuple((np.concatenate([l[i] for l in likelihoods.values()], axis=0)
                                   for i in range(ll_length)))}
            l_all_gt = {'All': tuple((np.concatenate([l[i] for l in likelihoods_gt.values()], axis=0)
                                      for i in range(ll_length)))}
            y_all = {'All': np.concatenate([y for y in test_y_values.values()], axis=0)}

            if plots['ll']:
                print('Plotting LL...')

                cmp_plot(f'{experiment_id}: Test_PredictionLL', f'{plot_filepath_base}_ll_test_all', p_all, l_all,
                         y_all, columns=1, rolling=rolling)
                cmp_plot(f'{experiment_id}: Test_PredictionLL', f'{plot_filepath_base}_ll_test_all_r', p_all, l_all,
                         y_all, columns=1, rolling=rolling, sort_by=1)
                cmp_plot(f'{experiment_id}: Test_GTLL', f'{plot_filepath_base}_ll_test_all_gt', p_all, l_all_gt, y_all,
                         columns=1, rolling=rolling)

            if plots['mpe']:
                print('Plotting MPE...')

                l_all_mpe = {'All': tuple((np.concatenate([l[i] for l in likelihoods_mpe.values()], axis=0)
                                           for i in range(ll_length)))}
                ll_plot(f'{experiment_id}: LL PLot', f'{plot_filepath_base}_ll_cmp', p_all, y_all, l_all, l_all_mpe,
                        l_all_gt, rolling=rolling)

            if plots['train']:
                p_all = {'All': np.concatenate([p for p in train_predictions.values()], axis=0)}
                l_all = {'All': tuple((np.concatenate([l[i] for l in train_likelihoods.values()], axis=0)
                                       for i in range(ll_length)))}
                l_all_gt = {
                    'All': tuple((np.concatenate([l[i] for l in likelihoods_gt_train.values()], axis=0)
                                  for i in range(ll_length)))}
                y_all = {'All': np.concatenate([y for y in train_y_values.values()], axis=0)}

                if plots['ll']:
                    print('Plotting Train LL...')

                    cmp_plot(f'{experiment_id}: Train_PredictionLL', f'{plot_filepath_base}_ll_train_all', p_all, l_all,
                             y_all, columns=1, rolling=rolling)
                    cmp_plot(f'{experiment_id}: Test_GTLL', f'{plot_filepath_base}_ll_train_all_gt', p_all, l_all_gt,
                             y_all, columns=1, rolling=rolling)

                if plots['mpe']:
                    print('Plotting Train MPE...')

                    l_all_mpe = {'All': tuple((np.concatenate([l[i] for l in train_likelihoods_mpe.values()],
                                                              axis=0) for i in range(ll_length)))}
                    ll_plot(f'{experiment_id}: LL Plot train', f'{plot_filepath_base}_ll_cmp_train',
                            p_all, y_all, l_all, l_all_mpe, l_all_gt, rolling=rolling)
        except NameError:
            print('Skipped MPE and LL plots')

    plot_single_group_detailed = data_source.plot_single_group_detailed or preprocessing.single_group

    if plots['train']:
        plot_filepath_train = f'{plot_filepath_base}_train'
        plot_experiment(f'{experiment_id}: Train', plot_filepath_train,
                        index_train, train_predictions, train_y_values,
                        plot_single_group_detailed=plot_single_group_detailed)
        print(f'Train plot saved to {plot_filepath_train}')

        if plots['reversed']:
            plot_filepath_train_unnormalized = f'{plot_filepath_base}_train_unnormalized'
            plot_experiment(f'{experiment_id}: Train', plot_filepath_train_unnormalized,
                            index_train, train_predictions_reversed,
                            train_y_values_reversed,
                            plot_single_group_detailed=plot_single_group_detailed)
            print(f'Train plot unnormlized saved to {plot_filepath_train_unnormalized}')

        if plots['mpe']:
            plot_experiment(f'{experiment_id}: Train MPE', plot_filepath_train + '_mpe',
                            index_train, train_predictions_mpe, train_y_values,
                            plot_single_group_detailed=plot_single_group_detailed)
            print(f'Train MPE plot saved to {plot_filepath_train + "_mpe"}')

    plot_filepath_test = f'{plot_filepath_base}_test'
    plot_experiment(f'{experiment_id}: Test', plot_filepath_test, index_test, predictions, test_y_values,
                    plot_single_group_detailed=plot_single_group_detailed)
    print(f'Test plot saved to {plot_filepath_test}')

    if plots['mpe']:
        try:
            plot_experiment(f'{experiment_id}: Test MPE', plot_filepath_test + '_mpe', index_test, predictions_mpe,
                            test_y_values,
                            plot_single_group_detailed=plot_single_group_detailed)
            print(f'Test MPE plot saved to {plot_filepath_test + "_mpe"}')
        except NameError:
            print('Skipped MPE plots')

    if plots['reversed']:
        plot_filepath_test_unnormalized = f'{plot_base_path}{experiment_id}_test_unnormalized'
        plot_experiment(f'{experiment_id}: Test', plot_filepath_test_unnormalized,
                        index_test, predictions_reversed, test_y_values_reversed,
                        plot_single_group_detailed=plot_single_group_detailed)
        print(f'Test plot unnormlized saved to {plot_filepath_test_unnormalized}')

    experiment_filepath = f'{experiments_base_path}{experiment_id}'
    save_experiment(experiment_filepath,
                    (data_source, preprocessing, evaluation_metrics, eval_results,
                     plot_filepath_base, model_filepath))
    print(f'Experiment saved to {experiment_filepath}')

    print(f'--- Experiment finished after {(datetime.now() - start_time).microseconds} microseconds ---')
