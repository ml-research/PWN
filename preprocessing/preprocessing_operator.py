from abc import ABC, abstractmethod


class PreprocessingOperator(ABC):
    """
    Abstract super class for all preprocessing operators
    """

    @abstractmethod
    def apply(self, data):
        """
        Applies the preprocessing operator to the given data_source
        """

        pass

    def reverse(self, values):
        """
        Applies the reverse normalization to some given values
        """

        pass
