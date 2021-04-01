import numpy as np
import matplotlib.pyplot as plt
import scripts.activations as atv
import scripts.batches as batches


def initialize_parameters(dims):
    """ Initialize the weight's and bias'

    :param dims: list of integers that each define the number of neurons in their respective layer

    :return:
    theta: dictionary containing numpy array's with the weights and biases for the network
    adams: dictionary containing numpy array's with the parameters for adams optimization
    """

    theta = {}
    adams = {}
    for l in range(1, len(dims)):
        theta['W' + str(l)] = np.random.randn(dims[l], dims[l - 1]) * np.sqrt(2 / dims[l - 1])
        adams['vdW' + str(l)] = np.zeros((dims[l], dims[l - 1]))
        adams['sdW' + str(l)] = np.zeros((dims[l], dims[l - 1]))

        theta['b' + str(l)] = np.zeros((dims[l], 1))
        adams['vdb' + str(l)] = np.zeros((dims[l], 1))
        adams['sdb' + str(l)] = np.zeros((dims[l], 1))

    return theta, adams


def linear_forward(A_prev, W, b):
    """ Calculates and produces the linear function on the weight's and bias' layer.
    Produces a linear cache as well.

    :param A_prev: numpy array of the previous layer's activation's, each column is for a different data sample
    :param W: numpy array containing the weights of the current layer in the network
    :param b: numpy array containing the biases of the current layer in the network

    :return:
    Z: numpy array containing the linear calculations at the weight's/bias' layer
    linear_cache: Tuple containing A_prev, W, and b (used for back prop)
    """

    Z = W @ A_prev + b
    linear_cache = (A_prev, W, b)

    return Z, linear_cache


def forward_activations(A_prev, W, b, activation):
    """ Executes a given activation on the weight's and bias' layer and gives back the activation
    to be used for the next layer

    :param A_prev: numpy array of the previous layer's activation's, each column is for a different sample
    :param W: numpy array containing the weights of the current layer in the network
    :param b: numpy array containing the biases of the current layer in the network
    :param activation: string representing the type of activation function to use

    :return:
    cache: tuple of the linear cache as well as the linear computations for a layer
    A: the activation for a layer
    """

    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == "sigmoid":
        A = atv.sigmoid(Z)
    elif activation == "ReLu":
        A = atv.ReLu(Z)
    cache = (Z, linear_cache)
    return cache, A


def forward_propagation(dims, X, theta):
    """ moves through each layer in the network and calculates the linear function and activations.
    Stores the linear function and activations in the caches.

    :param dims: list of integers that each define the number of neurons in their respective layer
    :param X: numpy array of input data. rows = dims[0], columns = the number of data samples
    :param theta: dictionary of parameters holding the weights and bias'

    :return:
    AL: numpy array containing the final activations for each sample (the raw prediction)
    caches: list of each cache produced from function forward_activations
    """

    caches = []
    A_prev = X

    for l in range(1, len(dims) - 1):
        cache, A = forward_activations(A_prev, theta['W' + str(l)], theta['b' + str(l)], 'ReLu')
        A_prev = A
        caches.append(cache)

    cache, AL = forward_activations(A_prev, theta['W' + str(l + 1)], theta['b' + str(l + 1)], 'sigmoid')
    caches.append(cache)

    return AL, caches


def cross_entropy_binary(Y, AL, theta, dims, lambd=0):
    """ computes the cross entropy cost and applies regularization

    :param Y: numpy array of true labels for each sample
    :param AL: numpy array containing the final activations for each data sample (the raw prediction)
    :param theta: dictionary of parameters holding the weights and bias'
    :param dims: list of integers that each define the number of neurons in their respective layer
    :param lambd: float regularization hyper parameter that stops exploding gradients

    :return:
    total_cost: float total cost of the current iteration
    dAL: numpy array containing the derivative of the final activation for each data sample '
    (used to calculate remaining derivatives of the final layer)
    """

    # This cost function takes the log of the probability that the network is correct in order to get the loss
    m = Y.shape[1]
    cross_entropy_cost = -1 / m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    regularized_cost = get_regularized_cost(dims, theta, m, lambd)
    total_cost = regularized_cost + cross_entropy_cost

    dAL = np.divide(1 - Y, 1 - AL) - np.divide(Y, AL)

    return np.squeeze(total_cost), dAL


def cross_entropy(Y, AL, theta, dims, lambd=0):
    """ Calculates the multiclass cross entropy cost and applies regularization

    :param Y: numpy array of true labels for each sample
    :param AL: numpy array containing the final activations for each data sample (the raw prediction)
    :param theta: dictionary of parameters holding the weights and bias'
    :param dims: list of integers that each define the number of neurons in their respective layer
    :param lambd: regularization hyper parameter that stops exploding gradients

    :return:
    total_cost: the total cost of the current iteration
    dAL: numpy array containing the derivative of the final activation for each data sample '
    (used to calculate remaining derivatives of the final layer)
    """

    # this cost function is not for multi class cross entropy therefore is also wrong
    m = Y.shape[1]

    cross_entropy_cost = -1 / m * np.sum(Y * np.log(AL))
    regularized_cost = get_regularized_cost(dims, theta, m, lambd)

    total_cost = cross_entropy_cost + regularized_cost
    dAL = Y - AL

    return np.squeeze(total_cost), dAL


def get_regularized_cost(dims, theta, m, lambd):
    """ Calculates L2 Regularization cost using hyper parameter lambd

    :param dims: list of integers that each define the number of neurons in their respective layer
    :param theta: dictionary of parameters holding the weights and bias'
    :param m: int number of samples to average over
    :param lambd: float regularization hyper parameter that stops exploding gradients

    :return: regularized_cost: float regularized cost to be added onto the unregularized cost
    """

    regularized_cost = 0

    for layer in range(1, len(dims) - 1):
        regularized_cost += np.sum(np.square(theta['W' + str(layer)]))

    regularized_cost *= 1 / m * lambd / 2

    return regularized_cost


def linear_backward(dZ, linear_cache, lambd=0):
    """ Calculate the derivatives of the cost function with respect to the weights, bias'
    and the previous layer's activation's

    :param dZ: numpy array containing the derivatives of the cost function with respect to the linear computation Z for
    each data sample
    :param linear_cache: Tuple containing A_prev, W, and b
    :param lambd: float regularization hyper parameter

    :return:
    dW: numpy array derivatives of the cost function with respect to the weights
    db: numpy array derivatives of the cost function with respect to the bias'
    dA_prev: numpy array derivatives of the cost function with respect to the previous layer's activation's
    """

    A_prev, W, b = linear_cache
    m = A_prev.shape[1]

    dW = (1. / m) * dZ @ A_prev.T + lambd / m * W
    db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = W.T @ dZ

    return dW, db, dA_prev


def activation_backward(dA, cache, activation, lambd=0):
    """ Calculates the derivative of the cost function with respect to Z using the given
    prime activation function

    :param dA: numpy array derivatives of the cost function with respect to the current layers activation's
    :param cache: tuple of the linear cache as well as the linear computations for a layer
    :param activation: string representing the type of activation function to use
    :param lambd: float regularization hyper parameter

    :return:
    dW: numpy array derivatives of the cost function with respect to the weights
    db: numpy array derivatives of the cost function with respect to the bias'
    dA_prev: numpy array derivatives of the cost function with respect to the previous layer's activation's
    """

    Z, linear_cache = cache
    if activation == "sigmoid":
        dZ = atv.sigmoid_derivative(dA, Z)
    elif activation == "ReLu":
        dZ = atv.ReLu_derivative(dA, Z)

    return linear_backward(dZ, linear_cache, lambd)


def back_propagation(dAL, caches, lambd=0):
    """ Goes through each layer other than the input layer starting from the output layer
    and calculates each of the derivatives, storing them in the dictionary 'grads'

    :param dAL: numpy array of derivatives of the cost function with respect to the last activation for each
    data sample
    :param caches: list of each cache produced from function forward_activations
    :param lambd: float regularization hyper parameter

    :return: grads: dictionary containing the derivatives of the cost function with respect to the weights and bias'
    """

    grads = {}
    layer_count = len(caches)
    current_cache = caches[layer_count - 1]

    dW, db, dA_prev = activation_backward(dAL, current_cache, "sigmoid", lambd)

    grads['dW' + str(layer_count)] = dW
    grads['db' + str(layer_count)] = db

    for l in reversed(range(1, layer_count)):
        current_cache = caches[l - 1]
        dW, db, dA_prev = activation_backward(dA_prev, current_cache, "ReLu", lambd)
        grads['dW' + str(l)] = dW
        grads['db' + str(l)] = db

    return grads


def update_parameters(dims, grads, adams, theta, alpha, t, beta_1=0.9, beta_2=0.999, eps=1e-8):
    """ Use adams optimization to update the weights and biases at each layer. Produces the
    learned adams parameters which will be used on the next iteration. Adams uses exponentially
    weighted averages to average out positive and negative gradients, producing a more straightforward direction
    to the global optimum. It uses previous values of the Adam parameters to adapt and learn new ones.

    :param dims: list of integers that each define the number of neurons in their respective layer
    :param grads: dictionary containing the derivatives of the cost function with respect to the weights and bias'
    :param adams: dictionary containing numpy array's with the parameters for adams optimization
    :param theta: dictionary containing numpy array's with the weights and biases for the network
    :param alpha: float learning rate that increases or decreases the step length for each iteration of Adams
    :param t: int the current iteration
    :param beta_1: float hyper parameter used for the EWA of the gradient descent with momentum part of Adams
    :param beta_2: float hyper parameter used for the EWA of RMSprop part of Adams
    :param eps: float value to remove errors when dividing during the RMSprop part of Adams

    :return: adams: dictionary of updated parameters for next iteration of Adams
    """

    for l in range(1, len(dims)):
        adams['vdW' + str(l)] = beta_1 * adams['vdW' + str(l)] + (1 - beta_1) * grads['dW' + str(l)]
        adams['vdb' + str(l)] = beta_1 * adams['vdb' + str(l)] + (1 - beta_1) * grads['db' + str(l)]
        adams['sdW' + str(l)] = beta_2 * adams['sdW' + str(l)] + (1 - beta_2) * np.square(grads['dW' + str(l)])
        adams['sdb' + str(l)] = beta_2 * adams['sdb' + str(l)] + (1 - beta_2) * np.square(grads['db' + str(l)])

        vdW_corrected = np.divide(adams['vdW' + str(l)], 1 - (beta_1 ** t))
        vdb_corrected = np.divide(adams['vdb' + str(l)], 1 - (beta_1 ** t))
        sdW_corrected = np.divide(adams['sdW' + str(l)], 1 - (beta_2 ** t))
        sdb_corrected = np.divide(adams['sdb' + str(l)], 1 - (beta_2 ** t))

        theta['W' + str(l)] -= alpha * np.divide(vdW_corrected, np.sqrt(sdW_corrected) + eps)
        theta['b' + str(l)] -= alpha * np.divide(vdb_corrected, np.sqrt(sdb_corrected) + eps)

    return adams


def predict_binary(X, Y, dims, theta):
    """ Predicts on a binary data set

    :param X: numpy array of input data. rows = dims[0], columns = the number of data samples
    :param Y: numpy array of true labels for each sample
    :param dims: list of integers that each define the number of neurons in their respective layer
    :param theta: dictionary containing numpy array's with the weights and biases for the network

    :return: p: numpy array of predictions rounded to 1 or 0
    """

    m = X.shape[1]
    p = np.zeros((1, m))

    AL, caches = forward_propagation(dims, X, theta)

    for i in range(0, AL.shape[1]):
        if AL[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    # print(f"predictions: " + str(p))
    # print(f"true labels: " + str(Y))
    print(f"Accuracy: " + str((np.sum((p == Y) / m)) * 100) + '%')

    return p


def training_model(dims, alpha, X, Y, num_iterations, theta, adams=None, lambd=0, is_mini_batch=False, batch_count=32, decay_rate=0):
    """ The base training model that either runs a batch model or mini batch model

    :param dims: list of integers that each define the number of neurons in their respective layer
    :param alpha: float learning rate that increases or decreases the step length for each iteration of Adams
    :param X: numpy array of input data. rows = dims[0], columns = the number of data samples
    :param Y: numpy array of true labels for each sample
    :param num_iterations: int number of iterations to train on
    :param theta: dictionary containing numpy array's with the weights and biases for the network
    :param adams: dictionary containing numpy array's with the parameters for adams optimization
    :param lambd: float regularization hyper parameter
    :param is_mini_batch: boolean whether to use mini batches or not
    :param batch_count: int the size for each batch
    :param decay_rate: float the rate to decay the learning rate

    :return: theta: dictionary containing numpy array's with the updated weights and biases for the network
    """
    if is_mini_batch:
        mini_batches = batches.generate_batches(batch_count, X, Y)
        costs = mini_batch_model(dims, alpha, mini_batches, num_iterations, decay_rate, theta, adams, lambd)
    else:
        costs = batch_model(dims, alpha, X, Y, num_iterations, decay_rate, theta, adams, lambd)

    # bold this sentence
    print("\033[1m" + "Close the plot to continue." + "\033[0m")
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iteration (per hundreds)')
    plt.title("Learning rate =" + str(alpha))
    plt.show()
    return theta


def mini_batch_model(dims, alpha, mini_batches, num_iterations, decay_rate, theta, adams=None, lambd=0):
    """ The mini batch model will train over mini batches

    :param dims: list of integers that each define the number of neurons in their respective layer
    :param alpha: float learning rate that increases or decreases the step length for each iteration of Adams
    :param mini_batches: list of mini batches that were created
    :param num_iterations: int number of iterations to train on
    :param decay_rate: float the rate to decay the learning rate
    :param theta: dictionary containing numpy array's with the weights and biases for the network
    :param adams: dictionary containing numpy array's with the parameters for adams optimization
    :param lambd: float regularization hyper parameter


    :return: cost: float final cost after training
    """

    costs = []

    base_alpha = alpha
    for i in range(num_iterations):
        # increase iteration count
        t = i + 1
        # decay the learning rate
        alpha = learning_rate_decay(decay_rate, t, base_alpha)

        # loop through mini batches
        for b in range(len(mini_batches)):
            # extract X and Y from mini batches
            X, Y = mini_batches[b]

            # propagate forward
            AL, caches = forward_propagation(dims, X, theta)

            # classify
            cost, dAL = cross_entropy_binary(Y, AL, theta, dims, lambd)
            # get derivatives
            grads = back_propagation(dAL, caches, lambd)

            # update parameters using derivatives
            adams = update_parameters(dims, grads, adams, theta, alpha, t)

        if i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if i % 100 == 0:
            costs.append(cost)

    return costs


def batch_model(dims, alpha, X, Y, num_iterations, decay_rate, theta, adams=None, lambd=0):
    """Batch model that will train over the entire batch of training data

     :param dims: list of integers that each define the number of neurons in their respective layer
    :param alpha: float learning rate that increases or decreases the step length for each iteration of Adams
    :param X: numpy array of input data. rows = dims[0], columns = the number of data samples
    :param Y: numpy array of true labels for each sample
    :param num_iterations: int number of iterations to train on
    :param decay_rate: float the rate to decay the learning rate
    :param theta: dictionary containing numpy array's with the weights and biases for the network
    :param adams: dictionary containing numpy array's with the parameters for adams optimization
    :param lambd: float regularization hyper parameter

    :return: cost: float final cost after training
    """

    costs = []
    base_alpha = alpha
    for i in range(num_iterations):
        t = i + 1
        alpha = learning_rate_decay(decay_rate, t, base_alpha)
        AL, caches = forward_propagation(dims, X, theta)

        cost, dAL = cross_entropy_binary(Y, AL, theta, dims, lambd)

        grads = back_propagation(dAL, caches, lambd)
        adams = update_parameters(dims, grads, adams, theta, alpha, t)

        if i % 100 == 0:
            # %i represents an int which will be the first value in the tuple
            # %f represents a float which will be the second value in the tuple
            print("Cost after iteration %i: %f" % (i, cost))
        if i % 100 == 0:
            costs.append(cost)

    return costs


def learning_rate_decay(decay_rate, epoch, base_alpha):
    """ decays a given learning rate

    :param decay_rate: float rate at which to decay
    :param epoch: int epoch number
    :param base_alpha: float starting learning rate value

    :return: alpha: float the decayed alpha
    """

    if decay_rate == 0:
        return base_alpha
    alpha = (1 / (1 + decay_rate * epoch)) * base_alpha

    return alpha
