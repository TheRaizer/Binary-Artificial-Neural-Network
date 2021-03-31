import numpy as np


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))

    return A


def ReLu(Z):
    A = np.maximum(0, Z)

    return A


def ReLu_derivative(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0

    return dZ


def sigmoid_derivative(dA, Z):
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    return dZ
