__author__ = 'dvgodoy'

from activations import Activation
from losses import Loss
import numpy as np


class Neuron(object):
    def __init__(self):
        self.bias = np.array([0], ndmin=2)
        self.db = np.array([], ndmin=2)
        self.weights = np.array([], ndmin=2)
        self.dw = np.array([], ndmin=2)
        self.activations = np.array([], ndmin=2)
        self.input_neurons = []
        self.output_neurons = []
        self.m = 0

    def update_parameters(self, learning_rate):
        self.weights = self.weights - learning_rate * self.dw
        self.bias = self.bias - learning_rate * self.db

    def add_output(self, neuron):
        self.output_neurons.append(neuron)

    def add_input(self, neuron):
        self.weights = np.append(self.weights, np.random.randn(1, 1) * 0.01).reshape(1, -1)
        self.input_neurons.append(neuron)

    def connect(self, neurons):
        assert neurons is not None
        if not isinstance(neurons, list):
            neurons = [neurons]
        assert isinstance(neurons[0], Neuron)

        for neuron in neurons:
            self.add_output(neuron)
            neuron.add_input(self)


class Input(Neuron):
    def __init__(self):
        super(Input, self).__init__()
        self.X = np.array([], ndmin=2)

    def examples(self, X):
        '''

        :param X: numpy array (1, m) containing m examples for a single feature
        :return: None
        '''
        assert isinstance(X, np.ndarray)
        assert X.shape[0] == 1

        self.m = X.shape[1]
        self.activations = X


class Hidden(Neuron):
    def __init__(self, activation_function):
        super(Hidden, self).__init__()
        assert isinstance(activation_function, Activation)
        self.activation_function = activation_function
        self.inputs = np.array([], ndmin=2)

    def forward_propagation(self):
        self.inputs = np.concatenate(map(lambda neuron: neuron.activations, self.input_neurons))
        assert self.inputs.shape[1] > 0
        assert self.weights.shape[1] == self.inputs.shape[0]
        assert self.bias.shape == (1, 1)

        self.z = np.dot(self.weights, self.inputs) + self.bias
        self.activations = self.activation_function.evaluate(self.z)

    def back_propagation(self):
        da = np.array(map(lambda neuron: neuron.da, self.output_neurons))
        self.dz = da * self.activation_function.gradient(self.z)
        self.dw = np.mean(self.dz * self.inputs, axis=1).reshape(1, -1)
        self.db = np.mean(self.dz, keepdims=True)
        self.da = np.dot(self.weights.T, self.dz)


class Output(Hidden):
    def __init__(self, activation_function, loss_function):
        super(Output, self).__init__(activation_function)
        assert isinstance(loss_function, Loss)
        self.loss_function = loss_function
        self.y = np.array([], ndmin=2)

    def responses(self, y):
        assert isinstance(y, np.ndarray)
        assert y.shape[0] == 1

        self.y = y

    def compute_cost(self):
        assert self.activations.shape == self.y.shape

        return np.mean(self.loss_function.compute(self.activations, self.y))

    def back_propagation(self):
        self.dz = self.loss_function.gradient(self.activations, self.y)
        self.dw = np.mean(self.dz * self.inputs, axis=1).reshape(1, -1)
        self.db = np.mean(self.dz, keepdims=True)
        self.da = np.dot(self.weights.T, self.dz)