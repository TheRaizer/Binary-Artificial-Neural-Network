import numpy as np
import matplotlib.pyplot as plt
import scripts.activations as atv
import scripts.batches as batches


def initialize_parameters(dims):
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
    Z = W @ A_prev + b
    linear_cache = (A_prev, W, b)

    return Z, linear_cache


def forward_activations(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == "sigmoid":
        A = atv.sigmoid(Z)
    elif activation == "ReLu":
        A = atv.ReLu(Z)
    elif activation == "softmax":
        A = atv.softmax(Z)
    cache = (Z, linear_cache)
    return cache, A


def forward_propagation(dims, X, theta):
    caches = []
    A_prev = X

    for l in range(1, len(dims) - 1):
        cache, A = forward_activations(A_prev, theta['W' + str(l)], theta['b' + str(l)], 'ReLu')
        A_prev = A
        caches.append(cache)

    cache, AL = forward_activations(A_prev, theta['W' + str(l + 1)], theta['b' + str(l + 1)], 'softmax')
    # cache, AL = forward_activations(A_prev, theta['W' + str(l + 1)], theta['b' + str(l + 1)], 'sigmoid')
    caches.append(cache)

    return AL, caches


def compute_cost(Y, AL, theta, dims, lambd=0):
    m = Y.shape[1]
    cross_entropy_cost = -1 / m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    regularized_cost = get_regularized_cost(dims, theta, m, lambd)
    total_cost = regularized_cost + cross_entropy_cost

    dAL = np.divide(1 - Y, 1 - AL) - np.divide(Y, AL)

    return np.squeeze(total_cost), dAL


def softmax_cost(Y, AL, theta, dims, lambd=0):
    m = Y.shape[1]
    cost = -1 / m * np.sum(Y * np.log(AL))
    regularized_cost = get_regularized_cost(dims, theta, m, lambd)

    total_cost = cost + regularized_cost
    dAL = np.divide(Y, AL)

    return np.squeeze(total_cost), dAL


def get_regularized_cost(dims, theta, m, lambd):
    regularized_cost = 0

    for layer in range(1, len(dims) - 1):
        regularized_cost += np.sum(np.square(theta['W' + str(layer)]))

    regularized_cost *= 1 / m * lambd / 2

    return regularized_cost


def linear_backward(dZ, linear_cache, lambd=0):
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]

    dW = (1. / m) * dZ @ A_prev.T + lambd / m * W
    db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = W.T @ dZ

    return dW, db, dA_prev


def activation_backward(dA, cache, activation, lambd=0, AL=None, Y=None):
    Z, linear_cache = cache
    if activation == "sigmoid":
        dZ = atv.sigmoid_backward(dA, Z)
    elif activation == "ReLu":
        dZ = atv.ReLu_backward(dA, Z)
    elif activation == "softmax":
        dZ = atv.softmax_backward(AL, Y)

    return linear_backward(dZ, linear_cache, lambd)


def back_propagation(dAL, caches, lambd=0, AL=None, Y=None):
    grads = {}
    layer_count = len(caches)
    current_cache = caches[layer_count - 1]

    dW, db, dA_prev = activation_backward(dAL, current_cache, "softmax", lambd, AL, Y)
    # dW, db, dA_prev = activation_backward(dAL, current_cache, "sigmoid", lambd, AL, Y)
    grads['dW' + str(layer_count)] = dW
    grads['db' + str(layer_count)] = db

    for l in reversed(range(1, layer_count)):
        current_cache = caches[l - 1]
        dW, db, dA_prev = activation_backward(dA_prev, current_cache, "ReLu", lambd)
        grads['dW' + str(l)] = dW
        grads['db' + str(l)] = db

    return grads


def update_parameters(dims, grads, adams, theta, alpha, t, beta_1=0.9, beta_2=0.999, eps=1e-8):
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


def predict(X, Y, dims, theta):
    m = X.shape[1]
    p = np.zeros((1, m))

    AL, caches = forward_propagation(dims, X, theta)

    true_labels = np.argmax(Y, axis=0)
    predicted = np.argmax(AL, axis=0)

    for i in range(0, AL.shape[1]):
        if true_labels[i] == predicted[i]:
            p[0, i] = 1
        else:
            p[0, i] = 0

    # print(f"predictions: " + str(p))
    # print(f"true labels: " + str(Y))
    print(f"Accuracy: " + str((np.sum((p == Y) / m)) * 100) + '%')

    return predicted, true_labels


def training_model(dims, alpha, X, Y, num_iterations, theta, adams=None, lambd=0, mini_batch=False):
    if mini_batch:
        mini_batches = batches.generate_batches(16, X, Y)
        costs = mini_batch_model(dims, alpha, mini_batches, num_iterations, theta, adams, lambd)
    else:
        costs = batch_model(dims, alpha, X, Y, num_iterations, theta, adams, lambd)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iteration (per hundreds)')
    plt.title("Learning rate =" + str(alpha))
    plt.show()
    return theta


def mini_batch_model(dims, alpha, mini_batches, num_iterations, theta, adams=None, lambd=0):
    costs = []

    for i in range(num_iterations):
        t = i + 1
        for b in range(len(mini_batches)):
            X, Y = mini_batches[b]
            AL, caches = forward_propagation(dims, X, theta)
            cost, dAL = softmax_cost(Y, AL, theta, dims, lambd)
            # cost, dAL = compute_cost(Y_batch, AL, theta, dims, lambd)
            grads = back_propagation(dAL, caches, lambd, AL, Y)
            adams = update_parameters(dims, grads, adams, theta, alpha, t)

            if i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
            if i % 100 == 0:
                costs.append(cost)

    return costs


def batch_model(dims, alpha, X, Y, num_iterations, theta, adams=None, lambd=0):
    costs = []
    for i in range(num_iterations):
        t = i + 1
        AL, caches = forward_propagation(dims, X, theta)
        cost, dAL = softmax_cost(Y, AL, theta, dims, lambd)
        # cost, dAL = compute_cost(Y_batch, AL, theta, dims, lambd)
        grads = back_propagation(dAL, caches, lambd, AL, Y)
        adams = update_parameters(dims, grads, adams, theta, alpha, t)

        if i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if i % 100 == 0:
            costs.append(cost)

    return costs
