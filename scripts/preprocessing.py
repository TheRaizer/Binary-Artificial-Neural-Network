import numpy as np


def standardize_input_data(X):
    X_standardized = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)

    return X_standardized