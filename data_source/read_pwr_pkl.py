import pickle
import datetime

from data_source.data_source import DataSource


class ReadPowerPKL(DataSource):

    def __init__(self, max_amt_groups=None):
        super(ReadPowerPKL, self).__init__(is_remote_source=False, plot_single_group_detailed=True)

        self._data = None
        self.company_mapping = {}
        self.max_amt_groups = max_amt_groups

    @property
    def data(self):
        # Lazy data loading
        if self._data is None:
            with open('res/power_data/full_data.pkl', 'rb') as f:
                data = pickle.load(f)

                self._data = []
                current_group = 0
                for key, years in data.items():
                    for year, vals in years.items():
                        current_datetime = datetime.datetime(int(year), 1, 1)

                        if current_group not in self.company_mapping.keys():
                            self.company_mapping[current_group] = key

                        for val_day in vals:
                            for val in val_day:
                                # No copy of current_datetime needed, datetime objects are immutable anyway
                                self._data.append({'column_names': ['day', 'year', 'company', 'value'],
                                                   'column_values': (current_datetime, current_datetime.year,
                                                                     current_group, val[1])})

                                current_datetime += datetime.timedelta(minutes=15)

                    current_group += 1

                    if self.max_amt_groups is not None and current_group >= self.max_amt_groups:
                        break

        return self._data

    def get_identifier(self):
        # __class__.__name__
        # Or rather as attribute?
        return __class__.__name__
