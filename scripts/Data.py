import numpy as np


class Data:
	def __init__(self, X, Y):
		self.X = X
		self.Y = Y

	def standardize_input_data(self):
		"""Standardize the data set to be represented as z-scores.

		"""

		X_standardized = (self.X - np.mean(self.X, axis=1, keepdims=True)) / np.std(self.X, axis=1, keepdims=True)

		self.X = X_standardized

	def normalize_input_data(self):
		"""Normalizes each value in self.X to be within the range of 0 and 1.

		"""

		X_normalized = (self.X - np.min(self.X, axis=1, keepdims=True)) / (
				np.max(self.X, axis=1, keepdims=True) - np.min(self.X, axis=1, keepdims=True))

		self.X = X_normalized

	def shuffle(self):
		""" Shuffles the rows of X and Y keeping each row of X matched with its corresponding row in Y.

		"""
		assert len(self.X) == len(self.Y)

		# generate list of random ints between the permuted range of 0 to len(self.X)
		permutation = np.random.permutation(len(self.X))

		# shuffle X and Y using the same permutation
		self.X = self.X[permutation]
		self.Y = self.Y[permutation]

	def transpose(self):
		self.X = self.X.T
		self.Y = self.Y.T

	def convert_labels_to_binary(self):
		Y_set_binary = np.zeros((1, self.Y.shape[1]))
		label_list = []

		for column in range(self.Y.shape[1]):
			if label_list.count(self.Y[0, column]) == 0:
				label_list.append(self.Y[0, column])
			Y_set_binary[0, column] = label_list.index(self.Y[0, column])

		self.Y = Y_set_binary

	def __str__(self):
		return "X shape:", self.X.shape, "Y shape:", self.Y.shape
