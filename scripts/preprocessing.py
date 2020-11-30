import numpy as np


def standardize_input_data(X):
    X_standardized = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)

    return X_standardized


def normalize_input_data(X):
    X_normalized = (X - np.mean(X, axis=1, keepdims=True)) / np.var(X, axis=1, keepdims=True)

    return X_normalized
