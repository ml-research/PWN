from data_source import BasicSelect, Mackey, ReadPowerPKL, ReadM4
from .minimal_preprocessor import MinimalPreprocessor
from util.dataset import split_dataset

import numpy as np


class ZScoreNormalization(MinimalPreprocessor):

    def __init__(self, index_columns, target_column, group_by_attribute, emb_attributes, split_data=True,
                 prediction_timespan=14, context_timespan=90, timespan_step=1, min_group_size=600, retail=False,
                 remove_sundays=False, single_group=False, multivariate=False):
        super().__init__(index_columns, target_column, group_by_attribute, emb_attributes, split_data,
                         prediction_timespan, context_timespan, timespan_step, min_group_size, retail, remove_sundays)

        self.mean = {}
        self.std = {}

        self.single_group = single_group
        self.multivariate = multivariate

    def apply(self, data, data_source=None, manual_split=True):
        """
        Applies the z-score normalization (0 mean, std of 1, performed per group) to the given data_source
            and splits it into index, features and values
        """

        features, targets, column_names, embedding_sizes = super().apply(data)

        if self.single_group:
            last_sequence_labels = np.concatenate([[i == len(f) - 1 for i, x in enumerate(f)]
                                                   for f in features.values()], axis=0)

            features = {'All': np.concatenate([x for x in features.values()], axis=0)}
            targets = {'All': np.concatenate([x for x in targets.values()], axis=0)}
        else:
            last_sequence_labels = None

        if self.multivariate:
            raise NotImplementedError()

        if manual_split and isinstance(data_source, ReadPowerPKL):
            train_x, train_y, test_x, test_y = {}, {}, {}, {}

            for key in features.keys():
                train_x[key], train_y[key], test_x[key], test_y[key] = [], [], [], []

                for f, t in zip(features[key], targets[key]):
                    company = data_source.company_mapping[f[-1, 1]]
                    year_start = f[0, 2]
                    year_end = f[-1, 2]

                    if ('germany_TenneT' in company and year_start == 2015 and year_end == 2015) or \
                        ('belgium' in company and year_start == 2016 and year_end == 2016) or \
                            ('austria' in company and year_start == 2017 and year_end == 2017) or \
                            ('germany_Amprion' in company and year_start == 2018 and year_end == 2018):
                        test_x[key].append(f)
                        test_y[key].append(t)
                    else:
                        train_x[key].append(f)
                        train_y[key].append(t)

                train_x[key], train_y[key], test_x[key], test_y[key] = \
                    np.array(train_x[key]), np.array(train_y[key]), np.array(test_x[key]), np.array(test_y[key])

        elif manual_split and isinstance(data_source, ReadM4):
            if last_sequence_labels is not None:
                key = next(iter(features.keys()))
                train_x = {key: np.array([f for i, f in enumerate(features[key])
                                          if not last_sequence_labels[i]])}
                train_y = {key: np.array([t for i, t in enumerate(targets[key])
                                          if not last_sequence_labels[i]])}

                test_x = {key: np.array([f for i, f in enumerate(features[key])
                                         if last_sequence_labels[i]])}
                test_y = {key: np.array([t for i, t in enumerate(targets[key])
                                         if last_sequence_labels[i]])}
            else:
                train_x = {key: f[:-1] for key, f in features.items()}
                train_y = {key: t[:-1] for key, t in targets.items()}

                test_x = {key: f[-1:] for key, f in features.items()}
                test_y = {key: t[-1:] for key, t in targets.items()}

        elif manual_split and isinstance(data_source, BasicSelect):
            # Removed due to NDA
            pass

        elif isinstance(data_source, BasicSelect) and not self.single_group:
            train_x, train_y, test_x, test_y = split_dataset(features, targets,
                                                             1. - (13 - 1) / len(next(iter(features.values()))))
        else:
            train_x, train_y, test_x, test_y = split_dataset(features, targets, 0.8)

        # Perform normalization per group - normalization is only performed on target attribute!
        del_count = 0
        for key in features.keys():
            try:
                combined_values = np.concatenate([train_x[key][..., -1], train_y[key][..., -1]], axis=-1)
            except:
                del_count += 1
                continue

            self.mean[key] = combined_values.mean()
            train_x[key][:, :, -1] -= self.mean[key]
            train_y[key][:, :, -1] -= self.mean[key]

            self.std[key] = combined_values.std()
            train_x[key][:, :, -1] /= self.std[key]
            train_y[key][:, :, -1] /= self.std[key]

            try:
                test_x[key][:, :, -1] -= self.mean[key]
                test_y[key][:, :, -1] -= self.mean[key]
                test_x[key][:, :, -1] /= self.std[key]
                test_y[key][:, :, -1] /= self.std[key]
            except IndexError:
                pass

        print(f'Deleted {del_count} groups, {len(features.values())} groups left')

        return train_x, train_y, test_x, test_y, column_names, embedding_sizes, last_sequence_labels

    def reverse(self, values):
        """
        Applies the reverse normalization to some given values
        """

        values_ = {}
        for key in values.keys():
            values_[key] = values[key] * self.std[key] + self.mean[key]

        return values_
