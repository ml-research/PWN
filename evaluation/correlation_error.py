from .evaluation_measure import EvaluationMeasure

import numpy as np


class CorrelationError(EvaluationMeasure):
    """
    Correlation Error (CE)
    """

    def calculate(self, prediction, y, ll=None):
        """
        Returns the CE for the given input
        """

        if ll is None:
            return np.array([1] * len(np.concatenate(list(prediction.values()), axis=0)))

        # Error is calculated across all groups
        prediction_ = np.concatenate(list(prediction.values()), axis=0)
        y_ = np.concatenate(list(y.values()), axis=0)
        ll_ = np.concatenate(list([l[0] for l in ll.values()]), axis=0).reshape((-1,))

        mse = ((prediction_ - y_) ** 2).mean(axis=-1)
        s_pred = np.sqrt((mse - np.nanmin(mse)) / (np.nanmax(mse) - np.nanmin(mse)))

        if np.any(np.isnan(ll_)):
            print('Warning! Detected NaN in LL!')

            ll_s = ll_
            s_ll = np.nan_to_num(
                np.sqrt(np.clip((ll_ - np.nanmax(ll_s)) / (np.nanmin(ll_s) - np.nanmax(ll_s)), 0, 1)), nan=1.)
        else:
            s_ll = np.sqrt((ll_ - np.nanmax(ll_)) / (np.nanmin(ll_) - np.nanmax(ll_)))

        return (s_pred - s_ll) ** 2


class CorrelationErrorSMape(EvaluationMeasure):
    """
    CorrelationError (CE), but based on SMape instead of MSE
    """

    def calculate(self, prediction, y, ll=None):
        if ll is None:
            return np.array([1] * len(np.concatenate(list(prediction.values()), axis=0)))

        # Error is calculated across all groups
        prediction_ = np.concatenate(list(prediction.values()), axis=0)
        y_ = np.concatenate(list(y.values()), axis=0)
        ll_ = np.concatenate(list([l[0] for l in ll.values()]), axis=0).reshape((-1,))

        mse = 2 * (np.abs((prediction_ - y_)) / (np.abs(prediction_) + np.abs(y_))).mean(axis=-1)
        s_pred = np.sqrt((mse - np.nanmin(mse)) / (np.nanmax(mse) - np.nanmin(mse)))

        if np.any(np.isnan(ll_)):
            print('Warning! Detected NaN in LL!')

            ll_s = ll_
            s_ll = np.nan_to_num(
                np.sqrt(np.clip((ll_ - np.nanmax(ll_s)) / (np.nanmin(ll_s) - np.nanmax(ll_s)), 0, 1)), nan=1.)
        else:
            s_ll = np.sqrt((ll_ - np.nanmax(ll_)) / (np.nanmin(ll_) - np.nanmax(ll_)))

        return (s_pred - s_ll) ** 2


class CorrelationErrorOrder(EvaluationMeasure):
    """
    CorrelationErrorOrder (CE)
    """

    def calculate(self, prediction, y, ll=None):
        if ll is None:
            return np.array([1] * len(np.concatenate(list(prediction.values()), axis=0)))

        # Error is calculated across all groups
        prediction_ = np.concatenate(list(prediction.values()), axis=0)
        y_ = np.concatenate(list(y.values()), axis=0)
        ll_ = np.concatenate(list([l[0] for l in ll.values()]), axis=0).reshape((-1,))

        mse = ((prediction_ - y_) ** 2).mean(axis=-1)

        mse_i = np.argsort(mse)
        ll_i = np.argsort(-ll_)

        combined = ll_i[mse_i]
        perfect_order = list(range(ll_i.shape[0]))
        perfect_order.reverse()

        return (((combined - perfect_order) / len(perfect_order)) ** 2).mean()
