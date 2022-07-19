import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# MSE Limit and LL Limit need to be adjusted to the respective model
mse_limit = [-0.5, 8]  # * 0.05
# ll_limit = [-50 * 1000, 2000]
# ll_limit = [-100000 * 1000 * 0.1, 10 * 1000]   #  * 0.015
ll_limit = [-350, 70]
plt.rcParams.update({'font.size': 32, 'figure.figsize': (60, 20), 'ps.useafm': False,
                     'pdf.use14corefonts': True, 'text.usetex': False})
plt.rcParams['text.latex.preamble'] = [r'\usepackage{sansmath}',
                                       r'\sansmath']  # Force sans-serif math mode (for axes labels)
plt.rcParams['font.family'] = 'sans-serif'  # ... for regular text
plt.rcParams['font.sans-serif'] = 'Helvetica, Avant Garde, Computer Modern Sans serif'  # Choose a nice font here


def plot_experiment(title, filepath, index, predictions, ground_truth, columns=4, plot_single_group_detailed=False):

    rows = max(min(next(iter(predictions.values())).shape[0] // columns if plot_single_group_detailed
              else (len(predictions.keys()) // columns) + 1, 6), 1)

    fig, ax = plt.subplots(nrows=rows, ncols=columns, figsize=(60, 40))
    # fig.suptitle(title)

    # Plots a single, random group detailed (rows * columns sequences)
    if plot_single_group_detailed:
        key, prediction = next(iter(predictions.items()))

        sequences = np.random.choice(np.arange(prediction.shape[0]), size=rows * columns)

        print(f'{filepath}: {sequences}')
        for i, i_ in enumerate(sequences):
            ax_ = ax[i // columns, i % columns] if rows > 1 and columns > 1 else (ax[i] if rows * columns > 1 else ax)

            ax_.plot(index[key][0], prediction[i_], 'b', label='Prediction')
            ax_.plot(index[key][0], ground_truth[key][i_], 'g', label='Ground Truth')

            ax_.set_title(key)
            ax_.set_xlabel('time')
            ax_.set_ylabel('amount sold')

            if i == 0:
                ax_.legend()

    # Plots the last sequence from each group
    else:
        sequences = get_sequences(predictions.keys())

        # for i, key in enumerate():
        for i, key in enumerate(sequences):
            try:
                ax_ = ax[i // columns, i % columns] if rows > 1 and columns > 1 else ax[i]
            except IndexError:
                break

            ax_.plot(index[key][-1], predictions[key][-1], label='Prediction')
            ax_.plot(index[key][-1], ground_truth[key][-1], label='Ground Truth')

            ax_.set_title(key)
            ax_.set_xlabel('time')
            ax_.set_ylabel('amount sold')

            if i == 0:
                ax_.legend()

    try:
        if filepath is not None and len(filepath) > 0:
            fig.savefig(filepath)
        else:
            fig.show()
    except OverflowError as e:
        print(e)


def get_rolling(series, rolling=12):
    rolled = pd.Series(series).rolling(window=rolling).mean().iloc[rolling - 1:].values

    return np.concatenate([[rolled[0]] * (rolling - 1), rolled])


def cmp_plot(title, filepath, predictions, likelihoods, targets, columns=3, rolling=12, sort_by=0,
             vlines=None, hlines=None):

    rows = min(5, max(1, len(predictions.keys()) // columns))

    fig, ax = plt.subplots(nrows=rows, ncols=columns, figsize=(18, 9))
    # fig.suptitle(title)
    # fig.tight_layout()

    sequences = get_sequences(predictions.keys())

    # for i, key in enumerate(predictions.keys()):
    for i, key in enumerate(sequences):
        try:
            ax_ = ax[i // columns, i % columns] if rows > 1 and columns > 1 else (ax[i] if rows * columns > 1 else ax)
            # ax_.set_title(key)
        except IndexError:
            break

        mse = ((predictions[key] - targets[key]) ** 2).mean(axis=-1)

        if len(likelihoods[key]) > 2:
            sorted_ = [(e, lc, lj, lm) for e, lc, lj, lm in
                       sorted(zip(mse, *likelihoods[key]), key=lambda x: x[sort_by])]
        else:
            sorted_ = [(e, lc) for e, lc in sorted(zip(mse, *likelihoods[key]), key=lambda x: x[sort_by])]

        index = list(range(len(sorted_)))

        color = 'tab:red'
        ax_.set_xlabel('Sequence \\#')
        ax_.set_ylabel('Prediction MSE')
        ax_.plot(index, get_rolling([s[0] for s in sorted_], rolling=rolling) if sort_by != 0
            else [s[0] for s in sorted_], label='Prediction MSE', color='tab:red', linewidth=2.5)
        ax_.tick_params(axis='y')
        ax_.set_ylim(mse_limit)

        ax_2 = ax_.twinx()
        ax_2.set_ylabel('Conditional Whittle LL')

        if vlines is not None and hlines is not None:
            ax_2.vlines(vlines, ymin=-3000, ymax=hlines, colors='grey', linestyles='dashed')
            ax_2.hlines(hlines, xmin=-1, xmax=vlines, colors='grey', linestyles='dashed')

        if len(likelihoods[key]) > 2:
            ax_2.plot(index, get_rolling([s[2][0] for s in sorted_], rolling=rolling), color='tab:blue', label='joint', linewidth=2.5)
            ax_2.plot(index, get_rolling([s[3][0] for s in sorted_], rolling=rolling), color='orange', label='marginal', linewidth=2.5)

        ax_2.plot(index, get_rolling([s[1][0] for s in sorted_], rolling=rolling) if sort_by != 1
            else [s[1][0] for s in sorted_], color='green', label='CWLL', linewidth=2.5)

        lines, labels = ax_.get_legend_handles_labels()
        lines2, labels2 = ax_2.get_legend_handles_labels()
        ax_2.legend(lines + lines2, labels + labels2, loc='center left'if sort_by == 0 else 'center right')
        ax_2.tick_params(axis='y')

        if ll_limit is not None:
            ax_2.set_ylim(ll_limit)

    try:
        if filepath is not None and len(filepath) > 0:
            fig.savefig(filepath, bbox_inches='tight')
        else:
            fig.show()
    except OverflowError as e:
        print(e)


def ll_plot(title, filepath, predictions, targets, ll, ll_mpe, ll_gt, key='All', rolling=24):
    fig, ax_ = plt.subplots(nrows=1, ncols=1, figsize=(60, 40))
    fig.suptitle(title)

    mse = ((predictions[key] - targets[key]) ** 2).mean(axis=-1)
    sorted_ = [(e, l1, l2, l3) for e, l1, l2, l3 in
               sorted(zip(mse, ll[key][0], ll_gt[key][0], ll_mpe[key][0]), key=lambda x: x[0])]

    index = list(range(len(sorted_)))
    color = 'tab:red'
    ax_.set_xlabel('Sequence \\#')
    ax_.set_ylabel('Prediction MSE', color='tab:red')
    ax_.plot(index, [s[0] for s in sorted_], color='tab:red')
    ax_.tick_params(axis='y', labelcolor=color)

    ax_.legend()
    ax_.set_ylim(mse_limit)

    ax_2 = ax_.twinx()
    ax_2.set_ylabel('LL', color='tab:blue')
    ax_2.plot(index, get_rolling([s[1][0] for s in sorted_], rolling=rolling), color='tab:blue', label='Prediction C-LL')
    ax_2.plot(index, get_rolling([s[2][0] for s in sorted_], rolling=rolling), color='yellow', label='GT C-LL')
    ax_2.plot(index, get_rolling([s[3][0] for s in sorted_], rolling=rolling), color='green', label='MPE C-LL')

    ax_2.legend()
    ax_2.tick_params(axis='y', labelcolor=color)

    if ll_limit is not None:
        ax_2.set_ylim(ll_limit)

    try:
        if filepath is not None and len(filepath) > 0:
            fig.savefig(filepath)
        else:
            fig.show()
    except OverflowError as e:
        print(e)


def get_sequences(keys):
    # For the Retail Data, plot specific products
    if False:
        return []  # Removed for confidentiality reasons
    # Otherwise, Predict groups by their order
    else:
        return keys
