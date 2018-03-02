"""
utils_data.py

@author: Development Seed

Collection of tools to help with loading and dealing with data
"""
import numpy as np


def get_concatenated_data(data_set_fnames, shuffle=True, seed=None):
    """Helper to take a set of data.npz filenames (from label-maker) and concatenate them"""

    x_trains, x_tests = [], []
    y_trains, y_tests = [], []

    # Load each dataset and append
    for data_set_fname in data_set_fnames:
        pilot_set = np.load(data_set_fname)

        x_trains.append(pilot_set['x_train'].astype(np.float32))
        y_trains.append(pilot_set['y_train'].astype(np.float32))
        x_tests.append(pilot_set['x_test'].astype(np.float32))
        y_tests.append(pilot_set['y_test'].astype(np.float32))

    # Concatenate datasources into single arrays
    x_train = np.concatenate(x_trains)
    y_train = np.concatenate(y_trains)
    x_test = np.concatenate(x_tests)
    y_test = np.concatenate(y_tests)

    # Generate sequence of random ints
    shuffle_inds_train = np.arange(len(x_train))
    shuffle_inds_test = np.arange(len(x_test))
    if shuffle:
        np.random.seed(seed)

        # Use the same shuffling seq
        np.random.shuffle(shuffle_inds_train)
        np.random.shuffle(shuffle_inds_test)

    return dict(x_train=x_train[shuffle_inds_train, ...],
                y_train=y_train[shuffle_inds_train, ...],
                x_test=x_test[shuffle_inds_test, ...],
                y_test=y_test[shuffle_inds_test, ...])
