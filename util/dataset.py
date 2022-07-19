import random


def split_dataset(features, targets, split_portion):
    train_x, test_x = split_data(features, split_portion)
    train_y, test_y = split_data(targets, split_portion)

    return train_x, train_y, test_x, test_y


def split_data(values, split_portion):
    train = {}
    test = {}

    for key in values.keys():
        # If we have only one sequence, the data_source was not split, so select test data_source randomly from complete sequence
        if len(values[key]) == 1 and False:
            all_indices = list(range(len(values[key][0])))

            indices_train = sorted(random.sample(all_indices, k=int(split_portion*len(values[key][0]))))
            indices_test = [i for i in all_indices if i not in indices_train]

            train[key] = values[key][:, indices_train]
            test[key] = values[key][:, indices_test]

        # Split by sub-sequences
        else:
            split_point = int(split_portion * len(values[key])) - 1

            train[key] = values[key][:split_point]
            test[key] = values[key][split_point:]

    return train, test


def split_to_x_y(data):
    x = {}
    y = {}

    for key in data.keys():
        x[key] = data[key][:, :-1]
        y[key] = data[key][:, -1:]

    return x, y
