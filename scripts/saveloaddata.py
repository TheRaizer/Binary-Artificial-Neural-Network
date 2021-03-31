import pickle
import pandas as pd
import numpy as np


def check_theta_save(theta, file, mode):
    saveDec = input("Type 'save' or 'yes' to save the trained parameters: ")
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

