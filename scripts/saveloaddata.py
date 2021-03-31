import pickle
import pandas as pd
import numpy as np


def check_theta_save(dims, theta, theta_file, dims_file):
    saveDec = input("Type 'save' or 'yes' to save the trained parameters: ")
    if saveDec == 'yes' or saveDec == 'save':
        theta_file = open(theta_file, "wb")
        dims_file = open(dims_file, "wb")

        pickle.dump(theta, theta_file)
        pickle.dump(dims, dims_file)

        dims_file.close()
        theta_file.close()


def load_pkl(file):
    a_file = open(file, 'rb')
    obj = pickle.load(a_file)
    a_file.close()

    return obj


def load_csv_sets(file, usecols, delimiter, skiprows):
    X = np.loadtxt(open(file, 'rb'), delimiter=delimiter, usecols=usecols, skiprows=skiprows).T
    df = pd.read_csv(file, header=0)

    return X, df

