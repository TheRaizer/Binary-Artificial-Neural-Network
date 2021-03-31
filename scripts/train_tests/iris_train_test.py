import scripts.neuralnetwork as nn
import scripts.saveloaddata as svld
import numpy as np
from scripts.Data import Data

dims = [4, 3, 2, 1]


def train_iris_binary(dimensions=dims, alpha=0.1, iterations=5000, is_mini_batch=False, batch_count=16, lambd=0, decay_rate=0):
    theta, adams = nn.initialize_parameters(dimensions)
    X_train, df = svld.load_csv_sets("iris_data/IrisTrainingBinary.csv", "rb", (1, 2, 3, 4), ",", 1)
    Y_train = np.array(df.Species).reshape(1, len(df))

    data = Data(X_train, Y_train)
    data.convert_labels_to_binary()

    data.transpose()
    data.shuffle()

    data.transpose()

    data.standardize_input_data()
    data.normalize_input_data()

    theta = nn.training_model(dimensions, alpha, data.X, data.Y, iterations, theta, adams, lambd, is_mini_batch, batch_count, decay_rate)

    svld.check_theta_save(theta, 'thetas/iris_theta_binary.pkl', 'wb')

    return theta


def test_iris_binary():
    theta = svld.load_theta('thetas/iris_theta_binary.pkl', 'rb')
    X_test, df = svld.load_csv_sets("iris_data/IrisTestBinary.csv", "rb", (1, 2, 3, 4), ",", 1)
    Y_test = np.array(df.Species).reshape(1, len(df))

    data = Data(X_test, Y_test)
    data.convert_labels_to_binary()

    data.transpose()
    data.shuffle()

    data.transpose()
    data.standardize_input_data()
    data.normalize_input_data()

    nn.predict_binary(data.X, data.Y, dims, theta)
