import numpy as np
import scripts.neuralnetwork as nn
import scripts.saveloaddata as svld
from scripts.Data import Data

dims = [13, 9, 6, 3, 2, 1]


def train_heart_binary(dimensions=dims, alpha=0.00045, iterations=1000, is_mini_batch=True, batch_count=16, lambd=0.00001, decay_rate=0):
    theta, adams = nn.initialize_parameters(dimensions)
    X_train, df = svld.load_csv_sets("heart_disease/heart_train.csv", "rb", (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), ",", 1)

    Y_train = np.array(df.target).reshape(1, len(df))

    data = Data(X_train, Y_train)
    data.convert_labels_to_binary()

    data.transpose()
    data.shuffle()

    data.transpose()
    data.standardize_input_data()
    data.normalize_input_data()

    theta = nn.training_model(dimensions, alpha, data.X, data.Y, iterations, theta, adams, lambd, is_mini_batch, batch_count, decay_rate)

    svld.check_theta_save(theta, 'thetas/hrt_theta_binary.pkl', 'wb')

    return theta


def test_heart_binary():
    theta = svld.load_theta('thetas/hrt_theta_binary.pkl', 'rb')
    X_test, df = svld.load_csv_sets("heart_disease/heart_test.csv", "rb", (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), ",", 1)
    Y_test = np.array(df.target).reshape(1, len(df))

    data = Data(X_test, Y_test)
    data.convert_labels_to_binary()

    data.transpose()
    data.shuffle()

    data.transpose()
    data.standardize_input_data()
    data.normalize_input_data()

    nn.predict_binary(data.X, data.Y, dims, theta)
