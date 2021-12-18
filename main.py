import numpy as np
import math


class Neuron(object):
    def __init__(self):
        self.weights = np.array([1.0, 2.0])
        self.bias = 0.0

    def forward(self, inputs):
        a_cell_sum = np.sum(inputs * self.weights) + self.bias
        result = 1.0 / (1.0 + math.exp(-a_cell_sum))
        return result


neuron = Neuron()

output = neuron.forward(np.array([1, 1]))
print(output)
