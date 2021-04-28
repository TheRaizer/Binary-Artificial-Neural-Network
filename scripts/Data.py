import numpy as np


class Data:
    """ Holds the input and labels of a set of data

    Methods:
    __init__: initializes input data and labels into self.X and self.Y respectively
    __str__: produces non-empty string containing the shapes of self.X and self.Y
    standardize_input_data: standardizes and mutates the input data self.X
    normalize_input_data: normalizes and mutates the input data self.X
    shuffle: shuffles both self.X and self.Y
    transpose: transposes and mutates the matrices/vectors self.X and self.Y
    convert_labels_to_binary: converts self.Y from an array of string labels to an array of 0's and 1's

    Attributes:
    X: input data, non-empty numpy ndarray
    Y: labels, non-empty numpy array with either row or column length of 1
    """

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def standardize_input_data(self):
        """Standardize the data set to be represented as z-scores.

        Postconditions:
        Every value in self.X is represented as a z-score on a standard distribution.
        """

        X_standardized = (self.X - np.mean(self.X, axis=1, keepdims=True)) / np.std(self.X, axis=1, keepdims=True)

        self.X = X_standardized

    def normalize_input_data(self):
        """Normalize the data set between 0 and 1

        Postconditions:
        Normalizes each value in self.X to be within the range of 0 and 1.
        """

        X_normalized = (self.X - np.min(self.X, axis=1, keepdims=True)) / (
                np.max(self.X, axis=1, keepdims=True) - np.min(self.X, axis=1, keepdims=True))

        self.X = X_normalized

    def shuffle(self):
        """ Shuffles the rows of X and Y keeping each row of X matched with its corresponding row in Y.

        Postconditions:
        Reassigns self.X and self.Y to be their shuffled equivalent.
        """
        assert len(self.X) == len(self.Y)

        # generate list of random ints between the permuted range of 0 to len(self.X)
        permutation = np.random.permutation(len(self.X))

        # shuffle X and Y using the same permutation
        self.X = self.X[permutation]
        self.Y = self.Y[permutation]

    def transpose(self):
        """ Transposes both the X and Y matrices/vectors.

        Postconditions:
        Reassigns self.X and self.Y to be their transposed equivalents.
        """
        self.X = self.X.T
        self.Y = self.Y.T

    def convert_labels_to_binary(self):
        """ Convert string labels to binary 0 or 1 (self.Y must contain exactly 2 unique labels)
    
        Postconditions:
        Reassigns self.Y to be its equivalent, but with the strings replaced with either 0 or 1.
        """

        # the shape of the converted set should be a single row with the same number of columns as self.Y
        Y_set_binary = np.zeros((1, self.Y.shape[1]))

        # the labels that were found
        label_list = []

        for column in range(self.Y.shape[1]):
            # if the label_list does not contain the string label
            if label_list.count(self.Y[0, column]) == 0:
                # add the label to the list
                label_list.append(self.Y[0, column])
            # assign the column in the binary set to be the index of the label
            Y_set_binary[0, column] = label_list.index(self.Y[0, column])

        self.Y = Y_set_binary

    def __str__(self):
        return "X shape:", self.X.shape, "Y shape:", self.Y.shape
