import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, weights) -> None:
        """
        Initialize the Perceptron object.

        Args:
            weights (np.array): The initial weights for the perceptron.
        """
        self.weights = weights

def run_epoch(p: Perceptron, x: np.ndarray, activation, learning_rate: float, desired_output: float):
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
    u = np.multiply(x, p.weights) # element-wise multiplication
    y = activation(u) # activation function
    e = desired_output - y # error
    p.weights = p.weights + learning_rate*e*x # update weights

def train(data, outputs, p: Perceptron, activation, learning_rate, max_epochs):
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
            run_epoch(p, data[j], activation, learning_rate, outputs[j])

# Setting up the AND problem weights and 'training data' example to validate the perceptron

p = Perceptron(np.random.rand(3))
# p = Perceptron(np.array([1., 1., 1.]))

Data = np.array([[1, 0, 0, 0],
                 [1, 0, 1, 0],
                 [1, 1, 0, 0], 
                 [1, 1, 1, 1]])

train(Data[:, 0:3], Data[:, 3], p, lambda x: 1 if np.sum(x) > 0 else 0, 0.5, 12)

print(p.weights)


# displaying the 
for i in Data:
    plt.scatter(i[1], i[2], color='red' if i[3] == 1 else 'blue')

# defining the points to plot the line given the weights
# w0 + w1*x + w2*y = 0
# y = (-w0 - w1*x)/w2

point1 = [0, -p.weights[0]/p.weights[2]]
point2 = [-p.weights[0]/p.weights[1], 0]
x_values = [point1[0], point2[0]]
y_values = [point1[1], point2[1]]
plt.plot(x_values, y_values, linestyle="--")
plt.grid()
plt.show()
