from .evaluation_measure import EvaluationMeasure

import numpy as np


class SMAPE(EvaluationMeasure):

    def calculate(self, prediction, y,  ll=None):
        return {key: 2 * (np.abs((p - y[key])) / (np.abs(p) + np.abs(y[key]))).mean()
                for key, p in prediction.items()}
