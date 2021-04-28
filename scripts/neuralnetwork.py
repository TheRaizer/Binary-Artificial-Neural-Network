import numpy as np
import os
os.environ['MPLCONFIGDIR'] = '/tmp'
import matplotlib.pyplot as plt
import scripts.activations as atv
import scripts.batches as batches

SIGMOID = "sigmoid"
RELU = "reLu"
PRINT_ITERATION = 100

DERIV_BIAS = "db"
DERIV_WEIGHTS = "dW"
MOMENT_WEIGHTS = "vdW"
MOMENT_BIAS = "vdb"
RMS_WEIGHTS = "sdW"
RMS_BIAS = "sdb"
WEIGHTS = "W"
BIAS = "b"

INCALCULABLE = "incalculable"


def initialize_parameters(dims):
    """ Initialize the weights and bias'
    
    Preconditions:
    dims: list of int length >= 2

    Parameters:
    dims: list of integers that each define the number of neurons in their respective layer

    Postconditions:
    theta: dictionary containing numpy array's with the weights and biases for the network
    adams: dictionary containing numpy array's with the parameters for adams optimization
    """

    theta = {}
    adams = {}
    for l in range(1, len(dims)):
        # initialize random values for the weights of each layer
        theta[WEIGHTS + str(l)] = np.random.randn(dims[l], dims[l - 1]) * np.sqrt(2 / dims[l - 1])

        # intialize zeros for the biases of each layer
        theta[BIAS + str(l)] = np.zeros((dims[l], 1))

        # zeros for each adams parameters as they will be bias corrected later
        adams[MOMENT_WEIGHTS + str(l)] = np.zeros((dims[l], dims[l - 1]))
        adams[RMS_WEIGHTS + str(l)] = np.zeros((dims[l], dims[l - 1]))
        adams[MOMENT_BIAS + str(l)] = np.zeros((dims[l], 1))
        adams[RMS_BIAS + str(l)] = np.zeros((dims[l], 1))


    return theta, adams


def linear_forward(A_prev, W, b):
    """ Calculates and produces the linear function on the weights, bias' and previous layers activations.

    Preconditions:
    A_prev: numpy ndarray whose # rows = W # columns and # columns = # of data samples
    W: numpy ndarray whose # columns = A_prev # rows
    b: numpy ndarray whose rows = W rows and has 1 column 

    Parameters:
    A_prev: The previous layer's activation's, each column is for a different data sample
    W: The weights of a layer in the network
    b: The biases of a layer in the network

    Postconditions:
    Z: numpy ndarray containing linear transformations of the previous layers activations or input layer.
    linear_cache: Tuple containing A_prev, W, and b (used for back prop)
    """

    # The linear calculation is the weights dot producted with the previous activations plus the biases.
    Z = W @ A_prev + b
    linear_cache = (A_prev, W, b)

    return Z, linear_cache


def forward_activations(A_prev, W, b, activation):
    """ Executes a given activation function on the weight's and bias' layer and returns back the activations to be used for the next layer

    Preconditions:
    A_prev: numpy ndarray whose # rows = W # columns and # columns = # of data samples
    W: numpy ndarray whose # columns = A_prev # rows
    b: numpy ndarray whose rows = W rows and has 1 column 
    activation: string "reLu" or "sigmoid"

    Parameters:
    A_prev: The previous layer's activation's, each column is for a different data sample
    W: The weights of the current layer in the network
    b: The biases of the current layer in the network
    activation: The type of activation function to run

    Postconditions:
    cache: tuple of the linear cache as well as the linear computations(Z) of a layer
    A: the activations of a layer
    """

    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == SIGMOID:
        A = atv.sigmoid(Z)
    elif activation == RELU:
        A = atv.ReLu(Z)

    cache = (Z, linear_cache)
    return cache, A


def forward_propagation(dims, X, theta):
    """ Passes through each layer in the network and calculates the linear function and activations.
    Stores the linear function and activations in the caches.

    Preconditions:
    dims: list of int length >= 2
    X: numpy array whose rows = dims[0], columns = # of data samples
    theta: dict

    Parameters:
    dims: The dimensions of the neural network model
    X: The input into the neural network
    theta: The learned paramters of a neural network model

    Postconditions:
    AL: numpy array containing the final activations for each sample (the raw prediction)
    caches: list of each cache produced from function forward_activations()
    """

    caches = []
    # the first activations are represented as the input layer
    A_prev = X

    for l in range(1, len(dims) - 1):
        cache, A = forward_activations(A_prev, theta[WEIGHTS + str(l)], theta[BIAS + str(l)], RELU)
        A_prev = A
        caches.append(cache)

    # Sigmoid activation clamps values between 0 and 1 making it suited for the use of the last layer/output layer
    cache, AL = forward_activations(A_prev, theta[WEIGHTS + str(l + 1)], theta[BIAS + str(l + 1)], SIGMOID)
    caches.append(cache)

    return AL, caches


def cross_entropy_binary(Y, AL, theta, dims, lambd=0):
    """ Computes the cross entropy cost and applies regularization

    Preconditions:
    Y: numpy ndarray with the same dimensions as AL
    AL: numpy ndarray with the same dimensions as Y
    theta: dict
    lambd: float >= 0

    Parameters:
    Y: binary labels
    AL: raw predictions of the neural network
    theta: learned parameters of the model
    lambd: regularization hyper-parameter for negating overfitting

    Postconditions:
    total_cost: float total cost of the current iteration
    dAL: numpy array containing the derivative of the final activation for each data sample
    (used to calculate remaining derivatives of the final layer)
    """

    num_samples = Y.shape[1]

    # ignore any divide warnings numpy raises
    with np.errstate(divide='ignore', invalid='ignore'):
        # calculate the cross entropy cost
        cross_entropy_cost = -1 / num_samples * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))

        # if there was a division by zero
        if not np.isfinite(cross_entropy_cost):
            print("Cost has become incalculable")
            return INCALCULABLE, None

    # calculate the regularization cost
    regularized_cost = get_regularized_cost(dims, theta, num_samples, lambd)

    # combine the two costs
    total_cost = regularized_cost + cross_entropy_cost

    # calculate derivative of the cost function with respect to the predictions
    dAL = np.divide(1 - Y, 1 - AL) - np.divide(Y, AL)

    return np.squeeze(total_cost), dAL


def get_regularized_cost(dims, theta, num_samples, lambd):
    """ Calculates L2 Regularization cost using hyper parameter lambd

    Preconditions:
    dims: list of int length >= 2
    theta: dict
    num_samples: int

    Parameters:
    dims: dimensions of the neural network model
    theta: learned parameters
    num_samples: # of total data samples
    lambd: regularization hyper-parameter to negate overfitting

    Postconditions:
    regularized_cost: float regularized cost to be added onto the unregularized cost
    """

    regularized_cost = 0

    for layer in range(1, len(dims) - 1):
        regularized_cost += np.sum(np.square(theta[WEIGHTS + str(layer)]))

    regularized_cost *= (1 / num_samples) * (lambd / 2)

    return regularized_cost


def linear_backward(dZ, linear_cache, lambd=0):
    """ Calculate the derivatives of the cost function with respect to the weights, bias'
    and the previous layers activation's

    Preconditions:
    dZ: non-empty numpy ndarray
    linear_cache: tuple of 3 items
    lambd: float >= 0

    Parameters:
    dZ: non-empty numpy ndarray that contains the derivatives of the cost function with respect to Z (dC/dZ)
    linear_cache: Tuple containing A_prev, W, and b
    lambd: regularization hyper-parameter to negate overfitting

    Postconditions:
    dW: numpy array derivatives of the cost function with respect to the weights
    db: numpy array derivatives of the cost function with respect to the bias'
    dA_prev: numpy array derivatives of the cost function with respect to the previous layers activation's
    """

    # unpack the linear_cache tuple
    A_prev, W, b = linear_cache

    # dC is the derivative of the cost function
    # each derivative utilizes dZ which is dC/dZ due to the chain rule.

    # get the number of samples by getting the number of columns in the activations
    num_samples = A_prev.shape[1]

    # calculate dC/dW using dC/dZ dot producted with dZ/dW which is equivalent to A_prev.T
    # the lambd / num_samples * W is the derivative of the regularized cost which also utilizes W.
    dW = (1. / num_samples) * dZ @ A_prev.T + lambd / num_samples * W

    # calculate dC/db by summing dC/dZ into the shape of b. dZ/db is 1 so no need to account for it.
    db = (1. / num_samples) * np.sum(dZ, axis=1, keepdims=True)

    # calculate dC/dA_prev using dZ/dA which is equivalent to the transpose of W, dot producted with dC/dZ. 
    dA_prev = W.T @ dZ

    return dW, db, dA_prev


def activation_backward(dA, cache, activation, lambd=0):
    """ Calculates the derivative of the cost function with respect to Z using the given
    prime activation function

    Preconditions:
    dA: non-empty numpy ndarray
    cache: tuple of 2 items
    activation: string "sigmoid" or "relu"
    lambd: float >= 0

    Parameters:
    dA: derivatives of the cost function with respect to this layers activation's
    cache: tuple of the linear cache as well as the linear computations for this layer
    activation: type of activation function to run
    lambd: regularization hyper-parameter to negate overfitting

    Postconditions:
    Runs linear_backward() function which returns:
        dW: numpy array derivatives of the cost function with respect to the weights
        db: numpy array derivatives of the cost function with respect to the bias'
        dA_prev: numpy array derivatives of the cost function with respect to the previous layer's activation's
    """

    Z, linear_cache = cache
    if activation == SIGMOID:
        dZ = atv.sigmoid_derivative(dA, Z)
    elif activation == RELU:
        dZ = atv.ReLu_derivative(dA, Z)

    return linear_backward(dZ, linear_cache, lambd)


def back_propagation(dAL, caches, lambd=0):
    """ Passes through each layer other than the input layer starting from the output layer
    and calculates each of the derivatives, storing them in the dictionary 'grads'

    Preconditions:
    dAL: non-empty numpy ndarray
    caches: list of tuples
    lambd: float >= 0

    Parameters:
    dAL: derivatives of the cost function with respect to the last activation for each
    data sample
    caches: list of each cache produced from function forward_activations()
    
    Postconditions:
    grads: dictionary containing the derivatives of the cost function with respect to the weights and bias'
    """

    grads = {}
    layer_count = len(caches)
    current_cache = caches[layer_count - 1]

    # calculate the last layers derivatives
    # use SIGMOID because that was the last activation function we used during forward propagation
    dW, db, dA_prev = activation_backward(dAL, current_cache, SIGMOID, lambd)
    grads[DERIV_WEIGHTS + str(layer_count)] = dW
    grads[DERIV_BIAS + str(layer_count)] = db

    # calculate the remaining derivatives starting from the second last layer
    for l in reversed(range(1, layer_count)):
        current_cache = caches[l - 1]

        # use RELU because that was the activation function we used for the remaining layers 
        dW, db, dA_prev = activation_backward(dA_prev, current_cache, RELU, lambd)
        grads[DERIV_WEIGHTS + str(l)] = dW
        grads[DERIV_BIAS + str(l)] = db

    return grads


def update_parameters(dims, grads, adams, theta, alpha, epoch, beta_1=0.9, beta_2=0.999, eps=1e-8):
    """ Use adams optimization to update the weights and biases at each layer. Produces the
    learned adams parameters which will be used on the next iteration. Adams uses exponentially
    weighted averages to average out positive and negative gradients, producing a more straightforward direction
    to the global optimum. It uses previous values of the Adam parameters to learn new ones.

    Preconditions:
    dims: list of int length >= 2
    grads: dict
    adams: dict
    theta: dict
    alpha: float > 0
    epoch: int > 0
    beta_1: float >= 0
    beta_2: float >= 0
    eps: 0 <= float <= 0.01

    Parameters:
    dims: dimensions of the neural network
    grads: derivatives of the weights and biases
    adams: parameters used for adams optimization
    theta: learned parameters of the neural network model      
    alpha: learning rate
    epoch: epoch count
    beta_1: hyper-parameter that controls how much the previous gradients should affect new ones.
    beta_2: hyper parameter that controls how much current gradients should affect new ones.
    eps: a value of epsilon to avoid division by 0

    Postconditions:
    adams: dictionary of updated parameters for next iteration of Adams optimization
    """

    for l in range(1, len(dims)):
        # calculate gradient descent with momentum
        adams[MOMENT_WEIGHTS + str(l)] = beta_1 * adams[MOMENT_WEIGHTS + str(l)] + (1 - beta_1) * grads[DERIV_WEIGHTS + str(l)]
        adams[MOMENT_BIAS + str(l)] = beta_1 * adams[MOMENT_BIAS + str(l)] + (1 - beta_1) * grads[DERIV_BIAS + str(l)]

        # calculate root means squared
        adams[RMS_WEIGHTS + str(l)] = beta_2 * adams[RMS_WEIGHTS + str(l)] + (1 - beta_2) * np.square(grads[DERIV_WEIGHTS + str(l)])
        adams[RMS_BIAS + str(l)] = beta_2 * adams[RMS_BIAS + str(l)] + (1 - beta_2) * np.square(grads[DERIV_BIAS + str(l)])

        # use bias correction for each
        vdW_corrected = np.divide(adams[MOMENT_WEIGHTS + str(l)], 1 - (beta_1 ** epoch))
        vdb_corrected = np.divide(adams[MOMENT_BIAS + str(l)], 1 - (beta_1 ** epoch))
        sdW_corrected = np.divide(adams[RMS_WEIGHTS + str(l)], 1 - (beta_2 ** epoch))
        sdb_corrected = np.divide(adams[RMS_BIAS + str(l)], 1 - (beta_2 ** epoch))

        # calculate the new parameters using the Adams parameters
        theta[WEIGHTS + str(l)] -= alpha * np.divide(vdW_corrected, np.sqrt(sdW_corrected) + eps)
        theta[BIAS + str(l)] -= alpha * np.divide(vdb_corrected, np.sqrt(sdb_corrected) + eps)

    return adams


def predict_binary(X, Y, dims, theta):
    """ Predicts on a binary data set

    Preconditions:
    X: non-empty numpy ndarray
    Y: non-empty numpy ndarray
    dims: list of int length >= 2
    theta: dict

    Parameters:
    X: the input into the neural network
    Y: the binary labels of the neural network
    dims: dimensions of the neural network model
    theta: learned parameters of the neural network model
    
    Postconditions:
    p: numpy array of predictions rounded to 1 or 0
    """

    num_samples = X.shape[1]
    p = np.zeros((1, num_samples))

    # propagate forward in the model and obtain a prediction AL
    AL, caches = forward_propagation(dims, X, theta)

    for i in range(0, AL.shape[1]):
        # if the prediction is greater than 0.5 it is considered a 1
        if AL[0, i] > 0.5:
            p[0, i] = 1
        # otherwise its considered a 0
        else:
            p[0, i] = 0

    # compare the true labels and the predictions to obtain accuracy
    print("Accuracy: " + str((np.sum((p == Y) / num_samples)) * 100) + '%')

    return p


def training_model(dims, alpha, X, Y, num_iterations, theta, adams=None, lambd=0, is_mini_batch=False, batch_count=32, decay_rate=0):
    """ The base training model that either runs a batch model or mini batch model

    Preconditions:
    dims: list of int length >= 2
    alpha: float > 0
    X: non-empty numpy ndarray
    Y: non-empty numpy ndarray
    num_iterations: int > 0
    theta: dict
    adams: dict
    lambd: float >= 0
    is_mini_batch: bool
    batch_count: int > 0
    decay_rate: float >= 0

    Parameters:
    dims: The dimensions of the neural network
    alpha: The learning rate of the model
    X: the input into the neural network
    Y: the binary labels of the neural network
    num_iterations: The number of times to pass through the entire data set (epochs)
    theta: parameters to learn in the model
    adams: parameters to learn during Adams optimization
    lambd: The regularization parameter for negating overfitting
    is_mini_batch: Whether the model will be using mini-batches
    batch_count: The number of training samples in each batch
    decay_rate: The rate at which to reduce the learning rate each epoch

    Postconditions:
    theta: The learned parameters from the model
    """
    if is_mini_batch:
        mini_batches = batches.generate_batches(batch_count, X, Y)
        costs = mini_batch_model(dims, alpha, mini_batches, num_iterations, decay_rate, theta, adams, lambd)
    else:
        costs = batch_model(dims, alpha, X, Y, num_iterations, decay_rate, theta, adams, lambd)

    message = "Close the plot to continue. If [Enter] is pressed before the plot closes you will be skipping the next step."
    print(message.upper())

    # plot the costs
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iteration (per hundreds)')
    plt.title("Learning rate =" + str(alpha))
    plt.show()

    return theta


def mini_batch_model(dims, alpha, mini_batches, num_iterations, decay_rate, theta, adams=None, lambd=0):
    """ The mini batch model will train over mini batches

    Preconditions:
    dims: list of int length >= 2
    alpha: float > 0
    mini_batches: list of tuple
    num_iterations: int > 0
    decay_rate: float >= 0
    theta: dict
    adams: dict
    lambd: float >= 0

    Parameters:
    dims: The dimensions of the neural network
    alpha: The learning rate of the model
    mini_batches: the mini batches containing the input data and labels
    num_iterations: The number of times to pass through the entire data set (epochs)
    decay_rate: The rate at which to reduce the learning rate each epoch
    theta: parameters to learn in the model
    adams: parameters to learn during Adams optimization
    lambd: The regularization parameter for negating overfitting

    Postconditions:
    costs: list of each float cost
    """

    costs = []

    base_alpha = alpha
    for i in range(num_iterations):
        # increase epoch count
        epoch = i + 1
        # decay the learning rate
        alpha = learning_rate_decay(decay_rate, epoch, base_alpha)

        # loop through mini batches
        for b in range(len(mini_batches)):
            # extract X and Y from mini batches
            X, Y = mini_batches[b]

            # propagate forward
            AL, caches = forward_propagation(dims, X, theta)

            # calculate cost
            cost, dAL = cross_entropy_binary(Y, AL, theta, dims, lambd)

            # if the cost was incalculable stop training
            if cost == INCALCULABLE:
                return costs

            # get derivatives
            grads = back_propagation(dAL, caches, lambd)

            # update parameters using derivatives
            adams = update_parameters(dims, grads, adams, theta, alpha, epoch)

        if i % PRINT_ITERATION == 0:
            # %i represents an int which will be the first value in the tuple
            # %f represents a float which will be the second value in the tuple
            print("Cost after iteration %i: %f" % (i, cost))
            costs.append(cost)

    # return the costs so they can be plotted
    return costs


def batch_model(dims, alpha, X, Y, num_iterations, decay_rate, theta, adams=None, lambd=0):
    """Batch model that will train over the entire batch of training data

    Preconditions:
    dims: list of int length >= 2
    alpha: float > 0
    X: non-empty numpy ndarray
    Y: non-empty numpy ndarray
    num_iterations: int > 0
    decay_rate: float >= 0
    theta: dict
    adams: dict
    lambd: float >= 0

    Parameters:
    dims: The dimensions of the neural network
    alpha: The learning rate of the model
    X: the input into the neural network
    Y: the binary labels of the neural network
    num_iterations: The number of times to pass through the entire data set (epochs)
    decay_rate: The rate at which to reduce the learning rate each epoch
    theta: parameters to learn in the model
    adams: parameters to learn during Adams optimization
    lambd: The regularization parameter for negating overfitting

    Postconditions:
    costs: list of each float cost
    """
    
    costs = []
    base_alpha = alpha
    for i in range(num_iterations):
        # increase epoch count
        epoch = i + 1

        # decay learning rate
        alpha = learning_rate_decay(decay_rate, epoch, base_alpha)

        # propagate forward
        AL, caches = forward_propagation(dims, X, theta)
        
        # calculate the cost
        cost, dAL = cross_entropy_binary(Y, AL, theta, dims, lambd)

        # if the cost was incalculable stop training
        if cost == INCALCULABLE:
            return costs

        # get derivatives
        grads = back_propagation(dAL, caches, lambd)

        # update parameters using derivatives
        adams = update_parameters(dims, grads, adams, theta, alpha, epoch)

        if i % PRINT_ITERATION == 0:
            # %i represents an int which will be the first value in the tuple
            # %f represents a float which will be the second value in the tuple
            print("Cost after iteration %i: %f" % (i, cost))
            costs.append(cost)
            
    # return the costs so they can be plotted
    return costs


def learning_rate_decay(decay_rate, epoch, base_alpha):
    """ decays a given learning rate

    Preconditions:
    decay_rate: float >= 0
    epoch: int > 0
    base_alpha: float >= 0

    Parameters:
    decay_rate: The rate at which to reduce the learning rate each epoch
    epoch: # of times the network has passed through the data set
    base_alpha: starting learning rate value

    Postconditions:
    alpha: float the decayed alpha
    """

    if decay_rate == 0:
        return base_alpha
    alpha = (1 / (1 + decay_rate * epoch)) * base_alpha

    return alpha
