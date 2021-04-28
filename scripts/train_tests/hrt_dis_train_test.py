import numpy as np
import scripts.neuralnetwork as nn
import scripts.saveloaddata as svld
from scripts.Data import Data


TRAINING_SET_PATH = "heart_disease/heart_train.csv"
TEST_SET_PATH = "heart_disease/heart_test.csv"
DIMS_PATH = 'dims/hrt_dims.pkl'
THETA_PATH = 'thetas/hrt_theta_binary.pkl'


def train_heart_binary(dims, alpha=0.00045, iterations=1000, is_mini_batch=True, batch_count=16, lambd=0.00001, decay_rate=0):
    """ Trains the model on the heart disease binary data set.

    Preconditions:
    dims: list of int length >= 2
    alpha: float > 0
    iterations: int > 0
    is_mini_batch: bool
    batch_count: int > 0
    lambd: float >= 0
    decay_rate: >= 0

    Parameters:
    dims: The dimensions of the neural network
    alpha: The learning rate of the model
    iterations: The number of times to pass through the entire data set (epochs)
    is_mini_batch: Whether the model will be using mini-batches
    batch_count: The number of training samples in each batch
    lambd: The regularization parameter for negating overfitting
    decay_rate: The rate at which to reduce the learning rate each iteration

    Postconditions:
    theta: The learned parameters from the model
    """

    # initializes the parameter dicts for the network and adams optimization
    theta, adams = nn.initialize_parameters(dims)

    # load the data sets
    X_train, df = svld.load_csv_sets(TRAINING_SET_PATH, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), ",", 1)

    # get the labels using the loaded dataframe
    Y_train = np.array(df.target).reshape(1, len(df))

    # instantiate a Data class
    data = Data(X_train, Y_train)
    data.convert_labels_to_binary()

    # transpose the matrices in order to shuffle
    data.transpose()
    data.shuffle()

    # transpose them again in order to feed them into the neural network
    data.transpose()

    # preproccess the data
    data.standardize_input_data()
    data.normalize_input_data()

    # execute the training model
    theta = nn.training_model(dims, alpha, data.X, data.Y, iterations, theta, adams, lambd, is_mini_batch, batch_count, decay_rate)

    # test how it did
    print("\nAfter evaluation on test set the model had:")
    test_heart_binary(dims, theta)

    # ask if the user wishes to save the parameters
    svld.check_theta_save(dims, theta, THETA_PATH, DIMS_PATH)

    return theta


def test_heart_binary(dims=None, theta=None):
    """ Runs a neural network model on the heart disease test set

    Preconditions:
    dims: list of int length >= 2
    theta: dict

    Parameters:
    dims: The dimensions of the neural network model
    theta: The learned paramters of a neural network model

    Postconditions:
    Uses a model that has the given parameters and predicts on a test set.
    """

    # if no theta is given load it
    if theta is None:
        theta = svld.load_pkl(THETA_PATH)

    # if no dimensions are given load it
    if dims is None:
        dims = svld.load_pkl(DIMS_PATH)

    # load the .csv file for the test set
    X_test, df = svld.load_csv_sets(TEST_SET_PATH, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), ",", 1)

    # get the test labels from the pandas dataframe
    Y_test = np.array(df.target).reshape(1, len(df))

    # instantiate a Data class using the data
    data = Data(X_test, Y_test)
    data.convert_labels_to_binary()

    # transpose the matrices in order to shuffle
    data.transpose()
    data.shuffle()

    # transpose them again in order to feed them into the neural network
    data.transpose()

    # preproccess the data
    data.standardize_input_data()
    data.normalize_input_data()

    # predict on the test set
    nn.predict_binary(data.X, data.Y, dims, theta)
