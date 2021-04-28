import numpy as np


def sigmoid(Z):
    """ Calculates the sigmoid activation function on Z

    Preconditions:
    Z: non-empty numpy ndarray containing linear transformations of the previous layer's activations or input layer.

    Parameters:
    Z: The numpy array that will be used to calculate the activations of a layer

    Postconditions:
    A: non-empty numpy ndarray that contains the non-linear transformations of Z
    """

    A = 1 / (1 + np.exp(-Z))

    return A


def ReLu(Z):
    """ Calculates the Rectified linear unit (ReLu) activation function on Z

    Preconditions:
    Z: non-empty numpy ndarray containing linear transformations of the previous layer's activations or input layer.

    Parameters:
    Z: The numpy array that will be used to calculate the activations of a layer

    Postconditions:
    A: non-empty numpy ndarray that contains the non-linear transformations of Z
    """
    # With relu if the value of Z < 0 it is 0 otherwise it is passed through a 
    # linear function of slope 1 which is why its derivative is either 0 or 1.
    A = np.maximum(0, Z)

    return A


def ReLu_derivative(dA, Z):
    """ Calculates/Produces the derivative of the cost function with respect to Z when Z was passed through.

    Preconditions:
    dA: non-empty numpy narray with the same dimensions as Z
    Z: non-empty numpy ndarray containing linear transformations of the previous layers activations or input layer.

    Parameters:
    dA: Numpy array containing the derivative of the cost function with respect to the activations (dC/dA)
    Z: The numpy array that will be used to calculate the activations of a layer

    Postconditions:
    dZ: non-empty numpy ndarray that contains the derivative of the cost function with respect to Z (dC/dZ)
    """

    # We can make a copy of dA because wherever Z > 0 dA/dZ = 1
    # and since dC/dZ = (dC/dA)x(dA/dZ) we can just initialize it to be dA and change it when Z <= 0
    dZ = np.array(dA, copy=True)

    # where ever Z <= 0 the derivative is changed to 0
    dZ[Z <= 0] = 0

    return dZ


def sigmoid_derivative(dA, Z):
    """ Calculates/Produces the derivative of the cost function with respect to the Z when Z was passed through the sigmoid.

    Preconditions:
    dA: non-empty numpy narray with the same dimensions as Z
    Z: non-empty numpy ndarray containing linear transformations of the previous layers activations or input layer.

    Parameters:
    dA: Numpy array containing the derivatives of the cost function with respect to the activations (dC/dA)
    Z: The numpy array that will be used to calculate the activations of a layer

    Postconditions:
    dZ: non-empty numpy ndarray that contains the derivatives of the cost function with respect to Z (dC/dZ)
    """

    s = 1 / (1 + np.exp(-Z))

    # dC represents the derivative of the cost function
    # multiplies dC/dA by the dA/dZ to obtain the dC/dZ
    dZ = dA * s * (1 - s)

    return dZ
