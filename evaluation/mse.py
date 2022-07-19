from .evaluation_measure import EvaluationMeasure


class MSE(EvaluationMeasure):
    """
    Mean Squared Error (MSE)
    """

    def calculate(self, prediction, y,  ll=None):
        """
        Returns the MSE for the given input: 1/n * sum (prediction_i - y_i)^2
        """

        return {key: ((p - y[key]) ** 2).mean() for key, p in prediction.items()}
