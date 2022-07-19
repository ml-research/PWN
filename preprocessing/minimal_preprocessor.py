from .preprocessing_operator import PreprocessingOperator

import numpy as np

import datetime


class MinimalPreprocessor(PreprocessingOperator):
    earliest_datetime = datetime.datetime(2015, 1, 1, 0, 0, 0)
    earliest_date = datetime.date(2015, 1, 1)

    def __init__(self, index_columns, target_column, group_by_attribute, emb_attributes, split_data=True,
                 prediction_timespan=14, context_timespan=90, timespan_step=1, min_group_size=400, retail=False,
                 remove_sundays=False):
        """
        Indices must be absolute, not relative
        """

        self.index_columns = index_columns
        self.target_column = target_column
        self.group_by_attribute = group_by_attribute
        self.emb_attributes = emb_attributes
        self.split_data = split_data
        self.prediction_timespan = prediction_timespan
        self.context_timespan = context_timespan
        self.timespan_step = timespan_step
        self.min_group_size = min_group_size
        self.retail = retail
        self.remove_sundays = remove_sundays

    @property
    def sequence_length(self):
        return self.context_timespan + self.prediction_timespan

    def apply(self, data):
        """
        Applies no normalization, but splits the given data_source into index, features and values
        """

        final_features = {}
        final_targets = {}
        column_names = {}

        feature_indices = []
        for index, d_raw in enumerate(data):
            try:
                # noinspection PyStatementEffect
                d_raw['column_names'], d_raw['column_values']

                d = d_raw
            except KeyError:
                d = {'column_names': d_raw.column_names, 'column_values': d_raw.column_values}

            final_index_column = 0
            final_group_column = len(self.index_columns)
            final_target_column = len(d['column_names']) - 1

            # Build column names
            if index == 0:
                column_names['index'] = [(d['column_names'][i], final_index_column) for i in self.index_columns]
                column_names['group_by'] = (d['column_names'][self.group_by_attribute], final_group_column)
                column_names['target'] = (d['column_names'][self.target_column], final_target_column)
                column_names['features'] = []

                column_index = 2
                for i, name in enumerate(d['column_names']):
                    if i not in [*self.index_columns, self.group_by_attribute, self.target_column]:
                        feature_indices.append(i)
                        column_names['features'].append((name, column_index))
                        column_index += 1

            # Build features in correct order (index_0,..., index_n, group_by_attribute, features, target_column)
            if type(d['column_values'][0]) == datetime.datetime:
                index_col = [(d['column_values'][i] - MinimalPreprocessor.earliest_datetime).total_seconds() / 60 / 15
                             for i in self.index_columns]
            elif type(d['column_values'][0]) == datetime.date:
                index_col = [(d['column_values'][i] - MinimalPreprocessor.earliest_date).days()
                             for i in self.index_columns]
            else:
                index_col = [d['column_values'][i] for i in self.index_columns]

            d_ = [*index_col, d['column_values'][self.group_by_attribute]] + \
                 [d['column_values'][i] for i in feature_indices] + [d['column_values'][self.target_column]]
            d_ = [MinimalPreprocessor.correct_datatype(v) for v in d_]

            # Add elements to the respective group
            try:
                final_features[d_[final_group_column]].append(d_)
            # If the group doesnt exist yet, create it
            except KeyError:
                final_features[d_[final_group_column]] = [d_]

        key_mapping = {key: i for i, key in enumerate(final_features.keys())}
        # pickle.dump(key_mapping, open('retail_pred_keys.pkl', 'wb'))

        # Sort all lists by index, prune groups below min_group_size and split to sequences if split_data is set
        amt_padded = 0
        prune_count = 0
        total_groups_present = len(final_features.keys())
        for key in list(final_features.keys()):  # Get keys beforehand as list, since the key dictonary will change
            if len(final_features[key]) < self.min_group_size:
                del final_features[key]
                prune_count += 1
            else:
                for i in range(len(final_features[key])):
                    final_features[key][i][final_group_column] = key_mapping[final_features[key][i][final_group_column]]

                if self.retail:
                    features_sorted = self.retail_preprocess(final_features, key)
                else:
                    features_sorted = np.array(sorted(final_features[key], key=lambda entry: entry[0]))

                if self.split_data:
                    targets = []
                    feature_sequences = []
                    sequences = list(range(0, len(features_sorted) - self.sequence_length -
                                           self.prediction_timespan + 1, self.timespan_step))

                    if len(sequences) > 2:
                        for i in sequences:
                            feature_sequences.append(features_sorted[i:i + self.context_timespan])
                            targets.append(features_sorted[i + self.context_timespan: i + self.sequence_length])

                        # Add last sequence, i.e. the test set
                        feature_sequences.append(features_sorted[-self.sequence_length:-self.prediction_timespan])
                        targets.append(features_sorted[-self.prediction_timespan:])
                    elif len(features_sorted) >= self.sequence_length + self.prediction_timespan:
                        # Add first sequence
                        feature_sequences.append(features_sorted[:self.context_timespan])
                        targets.append(features_sorted[self.context_timespan: self.sequence_length])

                        # Add last sequence, i.e. the test set
                        feature_sequences.append(features_sorted[-self.sequence_length:-self.prediction_timespan])
                        targets.append(features_sorted[-self.prediction_timespan:])

                    elif len(features_sorted) >= self.prediction_timespan * 2:
                        amt_padded += 1

                        # Add first sequence with padding
                        f = features_sorted[:-self.prediction_timespan * 2]
                        feature_sequences.append(np.pad(f, ((self.context_timespan - len(f), 0), (0, 0))))
                        targets.append(features_sorted[-self.prediction_timespan * 2:-self.prediction_timespan])

                        # Add last sequence, i.e. the test set, with potential padding
                        f = features_sorted[-self.sequence_length:-self.prediction_timespan]
                        feature_sequences.append(np.pad(f, ((self.context_timespan - len(f), 0), (0, 0))))
                        targets.append(features_sorted[-self.prediction_timespan:])

                else:
                    targets = [features_sorted]
                    feature_sequences = [features_sorted]

                if len(feature_sequences) > 1 or len(targets) > 1:
                    final_targets[key] = np.array(targets)
                    final_features[key] = np.array(feature_sequences)
                else:
                    try:
                        del final_features[key]
                    except KeyError:
                        pass

                    print('WARNING! DROPPED SEQUENCE!')
                    prune_count += 1

        print(f'Warning! Index was not checked for breaks in continuity! '
              f'Rather, datasets with less than {self.min_group_size} entries have been pruned. '
              f'{prune_count} / {total_groups_present} groups have been pruned in the process.')
        print(f'Padded {amt_padded} / {sum([len(v) for v in final_features.values()])} sequences')

        embedding_sizes = [4*366*96, 10, 0]

        self.key_mapping = key_mapping

        return final_features, final_targets, column_names, embedding_sizes

    def reverse(self, values):
        """
        Applies the reverse normalization to some given values
        """

        # We did no normalization, so no operations needs to be performed here
        return values

    @staticmethod
    def correct_datatype(value):
        if hasattr(value, 'real'):
            return float(value.real)
        else:
            return int(value) if isinstance(value, str) else value

    def retail_preprocess(self, final_features, key, weekly=False):
        if weekly:
            features_sorted_ = np.array(sorted(final_features[key], key=lambda entry: float(
                f'{int(entry[0])}.{("0" if entry[1] < 10 else "") + str(int(entry[1]))}')))
            min_week = 1.  # min(all_weeks)
            max_week = 54.  # max(all_weeks)
            max_week_2020 = 49

            cur_week = 48.
            cur_year = 2018.
            features_sorted = []
            for week in features_sorted_:
                if week[0] == cur_year:
                    for i in range(int(cur_week), int(week[1])):
                        features_sorted.append(np.array([cur_year, i,
                                                         *features_sorted_[0, len(self.index_columns):-1], 0.]))
                        cur_week += 1

                else:
                    for i in range(int(cur_week), int(max_week + 1)):
                        features_sorted.append(np.array([cur_year, i,
                                                         *features_sorted_[0, len(self.index_columns):-1], 0.]))
                        cur_week += 1

                    cur_year += 1
                    cur_week = min_week

                    for i in range(int(cur_week), int(week[1])):
                        features_sorted.append(np.array([cur_year, i,
                                                         *features_sorted_[0, len(self.index_columns):-1], 0.]))
                        cur_week += 1

                features_sorted.append(week)
                cur_week += 1

            while cur_year < 2020:
                for i in range(int(cur_week), int(max_week + 1)):
                    features_sorted.append(np.array([cur_year, i,
                                                     *features_sorted_[0, len(self.index_columns):-1], 0.]))
                    cur_week += 1

                cur_year += 1
                cur_week = min_week

            for k in range(int(cur_week), max_week_2020 + 1):
                features_sorted.append(np.array([cur_year, cur_week,
                                                 *features_sorted_[0, len(self.index_columns):-1], 0.]))

        else:
            import datetime

            features_sorted_ = np.array(sorted(final_features[key], key=lambda entry: entry[0]))

            # 765 days in total
            features_sorted = []
            start_date = datetime.date(2018, 11, 26)
            end_date = datetime.date(2020, 12, 5)
            cur_date = start_date
            for d in features_sorted_:
                while cur_date != d[0]:
                    if cur_date.isoweekday() == 7:
                        cur_date += datetime.timedelta(days=1)
                        continue

                    features_sorted.append(np.array([(cur_date - start_date).days,
                                                     *features_sorted_[0, len(self.index_columns):-1], 0.]))
                    cur_date += datetime.timedelta(days=1)

                features_sorted.append(np.array([(d[0] - start_date).days,
                                                 *features_sorted_[0, len(self.index_columns):-1], d[-1]]))
                cur_date += datetime.timedelta(days=1)

            while cur_date != end_date + datetime.timedelta(days=1):
                if cur_date.isoweekday() == 7:
                    cur_date += datetime.timedelta(days=1)
                    continue

                features_sorted.append(np.array([(cur_date - start_date).days,
                                                 *features_sorted_[0, len(self.index_columns):-1], 0.]))
                cur_date += datetime.timedelta(days=1)

        return features_sorted
