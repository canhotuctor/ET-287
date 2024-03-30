import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, n_weights: int, activation = lambda x: 1 if x > 0 else -1) -> None:
        """
        Initialize the Perceptron object.

        Args:
            n_weights (np.array): The number of features being considered on this perceptron.
            activation (lambda): The activation function to be used. Defaults to the step function.
        """
        self.weights = np.random.rand(n_weights)
        self.bias = np.random.rand()
        self.activation = activation

    def run_epoch(self, x: np.array, learning_rate: float, desired_output: float):
        """
        Run a single epoch of the perceptron training.

        Args:
            x (np.array): The input data.
            learning_rate (float): The learning rate.
            desired_output (float): The desired output.

        Returns:
            None
        """
        y = self.eval(x) # feed-forward
        e = desired_output - y # error calculation
        self.weights = self.weights + learning_rate*e*x # update weights
        self.bias = self.bias + learning_rate*e # update bias

    def train(self, data: np.ndarray, outputs: np.array, learning_rate = 0.1, max_epochs = 100):
        """
        Train the perceptron.

        Args:
            data (np.ndarray): The input data.
            outputs (np.array): The desired outputs.
            learning_rate (float): The learning rate.
            max_epochs (int): The maximum number of epochs.

        Returns:
            None
        """
        for i in range(max_epochs):
            for j in range(data.shape[0]):
                self.run_epoch(data[j], learning_rate, outputs[j])
    
    def eval(self, inputs):
        u = np.multiply(self.weights, inputs).sum() + self.bias
        return self.activation(u)
    
    def success_rate(self, data, outputs):
        """
        Calculate the success rate of the perceptron.
        
        Args:
            data (np.ndarray): The input data.
            outputs (np.array): The desired outputs.
            
        Returns:
            float: The success rate of the perceptron.
        """
        correct = 0
        for i in range(data.shape[0]):
            if self.eval(data[i]) == outputs[i]:
                correct += 1
        return correct/data.shape[0]
        
        pass
