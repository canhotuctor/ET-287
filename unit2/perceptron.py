import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, weights: np.array) -> None:
        """
        Initialize the Perceptron object.

        Args:
            weights (np.array): The initial weights for the perceptron.
        """
        self.weights = weights

    def __init__(self, n_weights: int) -> None:
        """
        Initialize the Perceptron object.

        Args:
            weights (np.array): The initial weights for the perceptron.
        """
        self.weights = np.zeros(n_weights)

    def run_epoch(self, x: np.ndarray, activation, learning_rate: float, desired_output: float):
        """
        Run a single epoch of the perceptron training.

        Args:
            p (Perceptron): The perceptron object.
            x (np.array): The input data.
            activation (function): The activation function.
            learning_rate (float): The learning rate.
            desired_output (float): The desired output.

        Returns:
            None
        """
        u = np.multiply(x, self.weights) # element-wise multiplication
        y = activation(u) # activation function
        e = desired_output - y # error
        self.weights = self.weights + learning_rate*e*x # update weights

    def train(self, data, outputs, activation, learning_rate, max_epochs):
        """
        Train the perceptron.

        Args:
            data (np.array): The input data.
            outputs (np.array): The desired outputs.
            p (Perceptron): The perceptron object.
            activation (function): The activation function.
            learning_rate (float): The learning rate.
            max_epochs (int): The maximum number of epochs.

        Returns:
            None
        """
        for i in range(max_epochs):
            for j in range(data.shape[0]):
                self.run_epoch(data[j], activation, learning_rate, outputs[j])


