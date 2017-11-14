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
        self.gradients = np.array([], ndmin=2)
        self.inputs = np.array([], ndmin=2)

        self.m = 0

    def update_parameters(self, learning_rate):
        self.weights = self.weights - learning_rate * self.dw
        self.bias = self.bias - learning_rate * self.db

    def add_output(self, neuron):
        self.output_neurons.append(neuron)
        self.gradients = np.zeros((len(self.output_neurons), 0))

    def add_input(self, neuron):
        self.weights = np.append(self.weights, np.random.randn(1, 1) * 0.01).reshape(1, -1)
        self.input_neurons.append(neuron)
        self.inputs = np.zeros((len(self.input_neurons), 0))

    def connect(self, neurons):
        assert neurons is not None
        if not isinstance(neurons, list):
            neurons = [neurons]
        assert isinstance(neurons[0], Neuron)

        for neuron in neurons:
            self.add_output(neuron)
            neuron.add_input(self)

    def gradient_from(self, da, neuron):
        i = next((i for i, v in enumerate(self.output_neurons) if v == neuron), None)
        if self.gradients.shape[1] != da.shape[1]:
            self.gradients = np.zeros((len(self.output_neurons), da.shape[1]))
        self.gradients[i, np.newaxis, :] = da

    def input_from(self, activations, neuron):
        i = next((i for i, v in enumerate(self.input_neurons) if v == neuron), None)
        if self.inputs.shape[1] != activations.shape[1]:
            self.inputs = np.zeros((len(self.input_neurons), activations.shape[1]))
        self.inputs[i, np.newaxis, :] = activations


class Input(Neuron):
    def __init__(self):
        super(Input, self).__init__()
        self.activations = np.array([], ndmin=2)

    #def __repr__(self):
    #    return 'Input Neuron {}'.format(self.activations.shape)

    def examples(self, X):
        '''

        :param X: numpy array (1, m) containing m examples for a single feature
        :return: None
        '''
        assert isinstance(X, np.ndarray)
        assert X.shape[0] == 1

        self.m = X.shape[1]
        self.activations = X
        for neuron in self.output_neurons:
            neuron.input_from(self.activations, self)


class Hidden(Neuron):
    def __init__(self, activation_function):
        super(Hidden, self).__init__()
        assert isinstance(activation_function, Activation)
        self.activation_function = activation_function
        self.inputs = np.array([], ndmin=2)

    #def __repr__(self):
    #    return 'Hidden Neuron {}'.format(self.inputs.shape)

    def forward_propagation(self):
        assert self.inputs.shape[1] > 0
        assert self.weights.shape[1] == self.inputs.shape[0]
        assert self.bias.shape == (1, 1)

        self.z = np.dot(self.weights, self.inputs) + self.bias
        self.activations = self.activation_function.evaluate(self.z)
        for neuron in self.output_neurons:
            neuron.input_from(self.activations, self)

    def back_propagation(self):
        self.dz = np.sum(self.gradients, axis=0, keepdims=True) * self.activation_function.gradient(self.z)
        self.dw = np.mean(self.dz * self.inputs, axis=1).reshape(1, -1)
        self.db = np.mean(self.dz, keepdims=True)
        da = np.dot(self.weights.T, self.dz)
        for i, neuron in enumerate(self.input_neurons):
            gradient = da[i, np.newaxis, :]
            neuron.gradient_from(gradient, self)


class Output(Hidden):
    def __init__(self, activation_function, loss_function):
        super(Output, self).__init__(activation_function)
        assert isinstance(loss_function, Loss)
        self.loss_function = loss_function
        self.y = np.array([], ndmin=2)

    #def __repr__(self):
    #    return 'Output Neuron {}'.format(self.y.shape)

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
        da = np.dot(self.weights.T, self.dz)
        for i, neuron in enumerate(self.input_neurons):
            gradient = da[i, np.newaxis, :]
            neuron.gradient_from(gradient, self)