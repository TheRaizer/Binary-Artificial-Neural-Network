import scripts.neuralnetwork as nn
import scripts.saveloaddata as svld
import numpy as np
from scripts.Data import Data


def train_iris_binary(dims, alpha=0.1, iterations=5000, is_mini_batch=False, batch_count=16, lambd=0, decay_rate=0):
    theta, adams = nn.initialize_parameters(dims)
    X_train, df = svld.load_csv_sets("iris_data/IrisTrainingBinary.csv", (1, 2, 3, 4), ",", 1)
    Y_train = np.array(df.Species).reshape(1, len(df))

    data = Data(X_train, Y_train)
    data.convert_labels_to_binary()

    data.transpose()
    data.shuffle()

    data.transpose()

    data.standardize_input_data()
    data.normalize_input_data()

    theta = nn.training_model(dims, alpha, data.X, data.Y, iterations, theta, adams, lambd, is_mini_batch, batch_count, decay_rate)

    print("\nAfter evaluation on test set the model had:")
    test_iris_binary(dims, theta)
    svld.check_theta_save(dims, theta, 'thetas/iris_theta_binary.pkl', 'dims/iris_dims.pkl')

    return theta


def test_iris_binary(dims=None, theta=None):
    if theta is None:
        theta = svld.load_pkl('thetas/iris_theta_binary.pkl')
    if dims is None:
        dims = svld.load_pkl('dims/iris_dims.pkl')
    X_test, df = svld.load_csv_sets("iris_data/IrisTestBinary.csv", (1, 2, 3, 4), ",", 1)
    Y_test = np.array(df.Species).reshape(1, len(df))

    data = Data(X_test, Y_test)
    data.convert_labels_to_binary()

    data.transpose()
    data.shuffle()

    data.transpose()
    data.standardize_input_data()
    data.normalize_input_data()

    nn.predict_binary(data.X, data.Y, dims, theta)
