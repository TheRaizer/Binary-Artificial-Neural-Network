import pickle
import pandas as pd
import numpy as np


def check_theta_save(dims, theta, theta_file, dims_file):
    """ Asks the user and may save the learned parameters and dimensions

    Preconditions:
    dims: list of int length >= 2
    theta: dict
    theta_file: string
    dims_file: string

    Parameters:
    dims: the dimensions of the neural network model
    theta: learned parameters of the model
    theta_file: file path to save the parameters
    dims_file: file path to save the dimensions

    Postconditions:
    May or may not save the learned parameters as well as the dimensions used to obtain said parameters.
    """


    saveDecision = input("Type 'save' or 'yes' to save the trained parameters, otherwise type anything: ")

    if saveDecision == 'yes' or saveDecision == 'save':
        # open the files for binary writing
        theta_file = open(theta_file, "wb")
        dims_file = open(dims_file, "wb")

        # use the pickle import to save the dict and list
        pickle.dump(theta, theta_file)
        pickle.dump(dims, dims_file)

        # close the files
        dims_file.close()
        theta_file.close()


def load_pkl(file):
    """ Loads a .pkl file

    Preconditions:
    file: string

    Parameters:
    file: path to the file that will be loaded

    Postconditions:
    obj: some object that was loaded from the file
    """
    a_file = open(file, 'rb')
    obj = pickle.load(a_file)
    a_file.close()

    return obj


def load_csv_sets(file, usecols, delimiter, skiprows):
    """ Load a .csv file

    Preconditions:
    file: string
    usecols: collection of int or int
    delimiter: char
    skiprows: collection of int or int

    Parameters:
    file: path to the .csv file to be loaded
    usecols: the columns to use
    delimiter: the char that splits each piece of data in the file
    skiprows: the rows to skip

    Postconditions:
    X: input data extracted from the .csv
    df: the data frame extracted from the .csv
    """

    # get the input data from the .csv and transpose it
    X = np.loadtxt(open(file, 'rb'), delimiter=delimiter, usecols=usecols, skiprows=skiprows).T

    # read the .csv to a pandas dataframe
    df = pd.read_csv(file, header=0)

    return X, df

