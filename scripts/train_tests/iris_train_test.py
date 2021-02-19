from scripts import neuralnetwork as nn, saveloaddata as svld
import numpy as np


def train_iris_binary():
    dims = [4, 3, 2, 1]

    theta, adams = nn.initialize_parameters(dims)
    X_train, df = svld.load_csv_sets("iris_data/IrisTrainingBinary.csv", "rb", (1, 2, 3, 4), ",", 1)
    Y = np.array(df.Species).reshape(1, len(df))
    Y_train = svld.covert_labels_to_int_binary(Y)
    X_train, Y_train = svld.unison_shuffled_copies(X_train.T, Y_train.T)

    X_train = X_train.T
    Y_train = Y_train.T

    theta = nn.training_model(dims, 0.1, X_train, Y_train, 10_000, theta, adams, mini_batch=False, classification='binary')

    svld.check_theta_save(theta, 'thetas/iris_theta_binary.pkl', 'wb')

    return theta


def test_iris_binary():
    dims = [4, 3, 2, 1]
    theta = svld.load_theta('thetas/iris_theta_binary.pkl', 'rb')
    X_test, df = svld.load_csv_sets("iris_data/IrisTestBinary.csv", "rb", (1, 2, 3, 4), ",", 1)
    Y = np.array(df.Species).reshape(1, len(df))
    Y_test = svld.covert_labels_to_int_binary(Y)
    X_test, Y_test = svld.unison_shuffled_copies(X_test.T, Y_test.T)

    X_test = X_test.T
    Y_test = Y_test.T

    nn.predict_binary(X_test, Y_test, dims, theta)


def train_iris():
    dims = [4, 3, 3, 3]

    theta, adams = nn.initialize_parameters(dims)
    X_train, df = svld.load_csv_sets("iris_data/IrisTraining.csv", "rb", (1, 2, 3, 4), ",", 1)
    Y = np.array(df.Species).reshape(1, len(df))
    Y_train, label_list = svld.covert_labels_to_int(Y, dims)
    X_train, Y_train = svld.unison_shuffled_copies(X_train.T, Y_train.T)

    X_train = X_train.T
    Y_train = Y_train.T

    theta = nn.training_model(dims, 0.0075, X_train, Y_train, 2500, theta, adams, classification='multiclass', mini_batch=True, batch_count=16)
    svld.check_theta_save(theta, 'thetas/iris_theta.pkl', 'wb')

    return theta


def test_iris():
    dims = [4, 3, 3, 3]

    theta = svld.load_theta('thetas/iris_theta.pkl', 'rb')
    X_test, df = svld.load_csv_sets("iris_data/IrisTest.csv", "rb", (1, 2, 3, 4), ",", 1)

    Y = np.array(df.Species).reshape(1, len(df))
    Y_test, label_list = svld.covert_labels_to_int(Y, dims)
    X_test, Y_test = svld.unison_shuffled_copies(X_test.T, Y_test.T)

    X_test = X_test.T
    Y_test = Y_test.T
    predicted, true_labels = nn.predict(X_test, Y_test, dims, theta)

    raw_predicted_labels = svld.convert_int_to_labels(predicted, label_list)
    raw_true_labels = svld.convert_int_to_labels(true_labels, label_list)

    print('Predicted labels')
    print(predicted)
    print(' '.join(str(label) for label in raw_predicted_labels))
    print('True labels')
    print(true_labels)
    print(' '.join(str(label) for label in raw_true_labels))
