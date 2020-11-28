import pickle
import pandas as pd
import numpy as np


def check_theta_save(theta, file, mode):
    saveDec = input("type 'save' or 'yes' to save: ")
    if saveDec == 'yes' or saveDec == 'save':
        a_file = open(file, mode)
        pickle.dump(theta, a_file)
        a_file.close()


def load_theta(file, mode):
    a_file = open(file, mode)
    theta = pickle.load(a_file)
    a_file.close()

    return theta


def load_csv_sets(file, mode, usecols, delimiter, skiprows):
    X = np.loadtxt(open(file, mode), delimiter=delimiter, usecols=usecols, skiprows=skiprows).T
    df = pd.read_csv(file, header=0)

    return X, df


def covert_labels_to_int_binary(Y):
    Y_set = np.zeros((Y.shape[0], Y.shape[1]))
    label_list = []

    for row in range(Y.shape[0]):
        for column in range(Y.shape[1]):
            if label_list.count(Y[row, column]) == 0:
                label_list.append(Y[row, column])
            Y_set[row, column] = label_list.index(Y[row, column])

    return Y_set


def covert_labels_to_int(Y, dims):
    Y_set = np.zeros((len(dims) - 1, Y.shape[1]))
    label_list = []

    for row in range(len(dims) - 1):
        for column in range(Y.shape[1]):
            label = Y[0, column]
            if label_list.count(Y[0, column]) == 0:
                label_list.append(Y[row, column])
            Y_set[label_list.index(label), column] = 1

    return Y_set, label_list


def convert_int_to_labels(argmax_int_array, label_list):
    raw_labels = list()

    for i in range(len(argmax_int_array)):
        raw_labels.append(label_list[argmax_int_array[i]])

    return raw_labels


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))

    return a[p], b[p]
