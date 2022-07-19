from .evaluation_measure import EvaluationMeasure

import numpy as np


class MAE(EvaluationMeasure):
    """
    Mean Absolute Error (MAE)
    """

    def calculate(self, prediction, y,  ll=None):
        """
        Returns the MAE for the given input: 1/n * sum | prediction_i - y_i |
        """

        return {key: (np.abs((p - y[key]))).mean() for key, p in prediction.items()}
