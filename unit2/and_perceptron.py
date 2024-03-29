from perceptron import Perceptron
import numpy as np
import matplotlib.pyplot as plt

# Setting up the AND problem weights and 'training data' example to validate the perceptron

p = Perceptron(np.random.rand(3))
# p = Perceptron(np.array([1., 1., 1.]))

Data = np.array([[1, 0, 0, 0],
                 [1, 0, 1, 0],
                 [1, 1, 0, 0], 
                 [1, 1, 1, 1]])

p.train(Data[:, 0:3], Data[:, 3], lambda x: 1 if np.sum(x) > 0 else 0, 0.5, 12)

print(p.weights)

# displaying the data points 
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
