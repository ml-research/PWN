from .evaluation_measure import EvaluationMeasure

import numpy as np


class RMSE(EvaluationMeasure):
    """
    Root Mean Squared Error (RMSE)
    """

    def calculate(self, prediction, y, ll=None):
        """
        Returns the RMSE for the given input: root (1/n * sum (prediction_i - y_i)^2)
        """

        return {key: np.sqrt(((p - y[key]) ** 2).mean(dtype=float)) for key, p in prediction.items()}
