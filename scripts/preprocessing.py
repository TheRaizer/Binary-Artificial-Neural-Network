import numpy as np


def standardize_input_data(X):
    # standardize the data set using Gaussian distribution
    X_standardized = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)

    return X_standardized


def normalize_input_data(X):
    # normalize the data set between 0 and 1
    X_normalized = (X - np.min(X, axis=1, keepdims=True)) / (np.max(X, axis=1, keepdims=True) - np.min(X, axis=1, keepdims=True))

    return X_normalized
