import os
import pandas as pd
import numpy as np


def read_single_dataset(file):
    frame = pd.read_csv(file, index_col='date', parse_dates=True, infer_datetime_format=True)
    frame.drop(['realtime_start', 'realtime_end'], axis=1, inplace=True)

    return frame


def read_all_datasets(path='../res/kaggle/'):
    all_frames = [read_single_dataset(path + f) for f in os.listdir(path) if f.endswith('.csv')]
    values = np.concatenate([f.values for f in all_frames], axis=1)

    return all_frames[0].index.values, values
