import pickle
import datetime

from data_source.data_source import DataSource


class ReadM4(DataSource):

    def __init__(self, key='Daily'):
        super(ReadM4, self).__init__(is_remote_source=False, plot_single_group_detailed=True)

        self._data = None
        self.key = key
        self.company_mapping = {}

    @property
    def data(self):
        # Lazy data loading
        if self._data is None:

            self._data = []
            current_group = 0
            last_indices = {}
            with open(f'res/M4/{self.key}-train.csv', 'r') as f:

                for row in f:
                    row = row.strip()

                    try:
                        if row.startswith('V'):
                            continue
                    except:
                        continue

                    last_i = 0
                    for i, v in enumerate(row.replace('"', '').split(',')):
                        try:
                            val = float(v)
                        except ValueError:
                            continue

                        self._data.append({'column_names': ['index', 'is_test', 'company', 'value'],
                                           'column_values': (i, False, current_group, val)})
                        last_i = i

                    last_indices[current_group] = last_i
                    current_group += 1

            current_group = 0
            with open(f'res/M4/{self.key}-test.csv', 'rb') as f:

                for row in f:
                    row = row.strip()

                    try:
                        if row.startswith('V'):
                            continue
                    except:
                        continue

                    for i, v in enumerate(row.split(',')):
                        self._data.append({'column_names': ['index', 'is_test', 'company', 'value'],
                                           'column_values': (last_indices[current_group] + i, True,
                                                             current_group, float(v))})

                    current_group += 1

        return self._data

    def get_identifier(self):
        # __class__.__name__
        # Or rather as attribute?
        return __class__.__name__ + '_' + self.key
