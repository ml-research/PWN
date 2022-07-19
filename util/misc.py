import numpy as np


def generate_dummy_data(count, length, context_ratio=0.5, split=0.8):
    t = np.arange(length) * 0.1

    xy_split = int(length * context_ratio)
    train_x, train_y, test_x, test_y = [], [], [], []
    for i in range(count):
        d = np.sin(t + np.random.uniform(0, 2 * np.pi)) * np.random.uniform(0.5, 6) + np.random.normal(0, 0.1, length)
        d = np.expand_dims(d, -1)

        if i < count * split:
            train_x.append(d[:xy_split])
            train_y.append(d[xy_split:])

        else:
            test_x.append(d[:xy_split])
            test_y.append(d[xy_split:])

    return {'All': np.array(train_x)}, {'All': np.array(train_y)}, {'All': np.array(test_x)}, {'All': np.array(test_y)}

