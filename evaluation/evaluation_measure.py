from abc import ABC, abstractmethod


class EvaluationMeasure(ABC):
    """
    Abstract super class for all evaluation measures
    """

    @abstractmethod
    def calculate(self, prediction, y, ll=None):
        """
        Calculates the respective evaluation measure for the given predictions and gold standard (y)
        """

        pass
