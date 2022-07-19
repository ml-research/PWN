from abc import ABC, abstractmethod


class DataSource(ABC):
    """
    Abstract super class for all queries
    """

    def __init__(self, is_remote_source, plot_single_group_detailed=False):
        self.is_remote_source = is_remote_source
        self.plot_single_group_detailed = plot_single_group_detailed

    @property
    @abstractmethod
    def data(self):
        """
        Returns the query in string representation
        """

        pass
