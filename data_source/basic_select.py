from data_source.data_source import DataSource

import numbers
import string


class BasicSelect(DataSource):

    def __init__(self, tables, join_keys=[], where_conditions=[], columns=[], misc='', plot_single_group_detailed=False):
        super(BasicSelect, self).__init__(is_remote_source=True, plot_single_group_detailed=plot_single_group_detailed)

        self.tables = tables
        self.join_keys = join_keys
        self.where_conditions = where_conditions
        self.columns = columns
        self.misc = misc

    @property
    def data(self):
        assert len(self.join_keys) == len(self.tables) - 1

        query = 'SELECT '
        if len(self.columns) < 1:
            query += '*'
        else:
            for i, columns in enumerate(self.columns):
                for column in columns:
                    if type(column) == tuple:
                        query += f'{column[1]}({string.ascii_uppercase[i]}.{column[0]}),'
                    else:
                        query += f'{string.ascii_uppercase[i]}.{column},'

            # Remove last comma
            query = query[:-1]

        query += f' FROM {self.tables[0]} AS A'

        for i, join_keys in enumerate(self.join_keys):
            query += f' JOIN {self.tables[i + 1]} AS {string.ascii_uppercase[i + 1]} ON'

            for j, (join_key_1, join_key_2) in enumerate(join_keys):
                query += f' {string.ascii_uppercase[i]}.{join_key_1}={string.ascii_uppercase[i + 1]}.{join_key_2}'

                if j != len(join_keys) - 1:
                    query += ' AND'

        if len(self.where_conditions) > 0:
            query += ' WHERE '

            for i, (column, value) in enumerate(self.where_conditions):
                query += f'{column}=' + (f'{value}' if isinstance(value, numbers.Number) else f'\'{value}\'')

                if i != len(self.where_conditions) - 1:
                    query += ' AND '

        return query + self.misc

    def get_identifier(self):
        # __class__.__name__
        # Or rather as attribute?
        return __class__.__name__
