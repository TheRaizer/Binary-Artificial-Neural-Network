import numpy as np
import scripts.neuralnetwork as nn
import scripts.saveloaddata as svld
import scripts.preprocessing as pp

dims = [13, 9, 6, 3, 2, 1]


def train_heart_binary():
    theta, adams = nn.initialize_parameters(dims)
    X_train, df = svld.load_csv_sets("heart_disease/heart_train.csv", "rb", (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), ",", 1)
    Y = np.array(df.target).reshape(1, len(df))
    Y_train = svld.covert_labels_to_int_binary(Y)
    X_train, Y_train = svld.unison_shuffled_copies(X_train.T, Y_train.T)

    X_train = X_train.T
    Y_train = Y_train.T

    X_train = pp.standardize_input_data(X_train)
    X_train = pp.normalize_input_data(X_train)

    theta = nn.training_model(dims, 0.00045, X_train, Y_train, 1000, theta, adams, mini_batch=True, batch_count=16, lambd=0.00001)

    svld.check_theta_save(theta, 'thetas/hrt_theta_binary.pkl', 'wb')

    return theta


def test_heart_binary():
    theta = svld.load_theta('thetas/hrt_theta_binary.pkl', 'rb')
    X_test, df = svld.load_csv_sets("heart_disease/heart_test.csv", "rb", (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), ",", 1)
    Y = np.array(df.target).reshape(1, len(df))
    Y_test = svld.covert_labels_to_int_binary(Y)
    X_test, Y_test = svld.unison_shuffled_copies(X_test.T, Y_test.T)

    X_test = X_test.T
    Y_test = Y_test.T

    X_test = pp.standardize_input_data(X_test)
    X_test = pp.normalize_input_data(X_test)

    predictions = nn.predict_binary(X_test, Y_test, dims, theta)
    print(predictions)
    print(Y_test)