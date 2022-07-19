import datetime

import numpy as np

from data_source.data_source import DataSource


class Mackey(DataSource):

    def __init__(self, size=100, tmax=512, delta_t=0.1, delay=17, seed=1):
        super(Mackey, self).__init__(is_remote_source=False, plot_single_group_detailed=False)

        self.size = size
        self.tmax = tmax
        self.delta_t = delta_t
        self.delay = delay
        self.seed = seed

        self._data = None

    @property
    def data(self):
        # Lazy data loading
        if self._data is None:
            series = self.generate_sequences()

            self._data = []
            for i, seq in enumerate(series):
                current_datetime = datetime.datetime(int(2018), 1, 1)

                for val in seq:
                    # No copy of current_datetime needed, datetime objects are immutable anyway
                    self._data.append({'column_names': ['day', 'series', 'value'],
                                       'column_values': (current_datetime, i, val)})

                    current_datetime += datetime.timedelta(minutes=15)

        return self._data

    def generate_sequences(self):
        steps = int(self.tmax / self.delta_t) + 100

        # multi-dimensional data.
        def mackey(x, tau, gamma=0.1, beta=0.2, n=10):
            return beta * x[:, -tau] / (1 + np.float_power(x[:, -tau], n)) - gamma * x[:, -1]

        tau = int(self.delay * (1 / self.delta_t))
        x0 = np.ones([tau])
        x0 = np.stack(self.size * [x0], axis=0)

        print('Mackey initial state is random.')
        np.random.seed(self.seed)
        x0 += np.random.uniform(-0.1, 0.1, size=x0.shape)

        x = x0
        for _ in range(steps):
            res = np.expand_dims(x[:, -1] + self.delta_t * mackey(x, tau), -1)
            x = np.concatenate([x, res], -1)
        discard = 100 + tau

        return x[:, discard:]
