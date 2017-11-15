__author__ = 'dvgodoy'

from activations import Activation
from losses import Loss
import numpy as np
import abc


class Neuron(object):
    def __init__(self):
        self.bias = np.zeros((1, 1))
        self.db = np.zeros((1, 0))
        self.weights = np.zeros((1, 0))
        self.dw = np.zeros((1, 0))
        self.activations = np.zeros((1, 0))

        self.input_neurons = []
        self.input_hashes = np.array([])
        self.input_activated = np.array([])

        self.output_neurons = []
        self.output_hashes = np.array([])
        self.output_feedback = np.array([])

        self.gradients = np.zeros((1, 0))
        self.inputs = np.zeros((1, 0))

        self.m = 0
        self.layer = 0
        self.input_path_lengths = np.zeros(0)
        self.output_path_lengths = np.zeros(0)

    def update_parameters(self, learning_rate):
        self.weights = self.weights - learning_rate * self.dw
        self.bias = self.bias - learning_rate * self.db

    def add_output(self, neuron):
        self.output_neurons.append(neuron)
        self.output_hashes = np.append(self.output_hashes, [[neuron.__hash__()]])
        self.gradients = np.zeros((len(self.output_neurons), 0))
        self.output_path_lengths = np.zeros(len(self.output_neurons))
        self.output_feedback = np.zeros(len(self.output_neurons))

    def add_input(self, neuron):
        self.weights = np.append(self.weights, np.random.randn(1, 1) * 0.01).reshape(1, -1)
        self.input_neurons.append(neuron)
        self.input_hashes = np.append(self.input_hashes, [[neuron.__hash__()]])
        self.inputs = np.zeros((len(self.input_neurons), 0))
        self.input_path_lengths = np.zeros(len(self.input_neurons))
        self.input_activated = np.zeros(len(self.input_neurons))

    def connect(self, neurons):
        assert neurons is not None
        if not isinstance(neurons, list):
            neurons = [neurons]
        assert isinstance(neurons[0], Neuron)

        for neuron in neurons:
            # prevents connecting twice to same neuron
            if self.__neuron_index(self.output_hashes, neuron) < 0:
                # adds destination neuron to outputlist
                self.add_output(neuron)
                # at the destination neuron, adds source neuron as input
                neuron.add_input(self)

    def _initialize_forward(self, input_neuron=None, path_length=0):
        if path_length > self.layer:
            self.layer = path_length

        i = self.__neuron_index(self.input_hashes, input_neuron.__hash__())
        if i >= 0:
            self.input_path_lengths[i] = path_length

        for neuron in self.output_neurons:
            neuron._initialize_forward(self, path_length + 1)

    def _initialize_backward(self, output_neuron=None, path_length=0):
        i = self.__neuron_index(self.output_hashes, output_neuron.__hash__())
        if i >= 0:
            self.output_path_lengths[i] = path_length

        for neuron in self.input_neurons:
            neuron._initialize_backward(self, path_length + 1)

    def __neuron_index(self, neurons_hashes, neuron):
        try:
            idx = np.where(neurons_hashes == neuron.__hash__())[0][0]
        except IndexError:
            idx = -1
        return idx

    @abc.abstractmethod
    def forward_propagation(self):
        pass

    @abc.abstractmethod
    def back_propagation(self, learning_rate):
        pass

    def gradient_from(self, da, neuron, learning_rate):
        i = self.__neuron_index(self.output_hashes, neuron)
        if i >= 0:
            self.output_feedback[i] = 1.
            if self.gradients.shape[1] != da.shape[1]:
                self.gradients = np.zeros((len(self.output_neurons), da.shape[1]))
            self.gradients[i, np.newaxis, :] = da

            if np.all(self.output_feedback):
                self.output_feedback -= 1.
                self.back_propagation(learning_rate)

    def input_from(self, activations, neuron):
        i = self.__neuron_index(self.input_hashes, neuron)
        if i >= 0:
            self.input_activated[i] = 1.
            if self.inputs.shape[1] != activations.shape[1]:
                self.inputs = np.zeros((len(self.input_neurons), activations.shape[1]))
            self.inputs[i, np.newaxis, :] = activations

            if np.all(self.input_activated):
                self.input_activated -= 1.
                self.forward_propagation()


class Input(Neuron):
    def __init__(self):
        super(Input, self).__init__()
        self.activations = np.zeros((1, 0))

    def __str__(self):
        return 'Input Neuron {}'.format(self.activations.shape)

    def examples(self, X):
        '''
        Feeds m examples of a single feature to the neuron.

        :param X: numpy array (1, m) containing m examples for a single feature
        :return: None
        '''
        assert isinstance(X, np.ndarray)
        assert X.shape[0] == 1

        self.m = X.shape[1]
        self.activations = X
        self.forward_propagation()

    def forward_propagation(self):
        for neuron in self.output_neurons:
            neuron.input_from(self.activations, self)

    def back_propagation(self, learning_rate):
        pass

    def initialize(self):
        self._initialize_forward()

class Hidden(Neuron):
    def __init__(self, activation_function):
        super(Hidden, self).__init__()
        assert isinstance(activation_function, Activation)

        self.activation_function = activation_function
        self.inputs = np.zeros((1, 0))

    def __str__(self):
        return 'Hidden Neuron {}'.format(self.inputs.shape)

    def forward_propagation(self):
        assert self.inputs.shape[1] > 0
        assert self.weights.shape[1] == self.inputs.shape[0]
        assert self.bias.shape == (1, 1)

        self.z = np.dot(self.weights, self.inputs) + self.bias
        self.activations = self.activation_function.evaluate(self.z)
        for neuron in self.output_neurons:
            neuron.input_from(self.activations, self)

    def _compute_dz(self):
        return np.sum(self.gradients, axis=0, keepdims=True) * self.activation_function.gradient(self.z)

    def back_propagation(self, learning_rate):
        self.dz = self._compute_dz()
        self.dw = np.mean(self.dz * self.inputs, axis=1).reshape(1, -1)
        self.db = np.mean(self.dz, keepdims=True)
        da = np.dot(self.weights.T, self.dz)
        self.update_parameters(learning_rate)
        for i, neuron in enumerate(self.input_neurons):
            gradient = da[i, np.newaxis, :]
            neuron.gradient_from(gradient, self, learning_rate)


class Output(Hidden):
    def __init__(self, activation_function, loss_function, learning_rate):
        super(Output, self).__init__(activation_function)
        assert isinstance(loss_function, Loss)
        self.loss_function = loss_function
        if not callable(learning_rate):
            constant = learning_rate
            learning_rate = lambda epochs: constant
        self.learning_rate = learning_rate
        self.y = np.zeros((1, 0))
        self.epochs = 0
        self.cost = np.inf

    def __str__(self):
        return 'Output Neuron {}'.format(self.y.shape)

    def responses(self, y):
        assert isinstance(y, np.ndarray)
        assert y.shape[0] == 1
        self.y = y

    def forward_propagation(self):
        super(Output, self).forward_propagation()
        self.cost = self.compute_cost()
        self.epochs += 1
        self.back_propagation(self.learning_rate(self.epochs))

    def compute_cost(self):
        assert self.activations.shape == self.y.shape
        return np.mean(self.loss_function.compute(self.activations, self.y))

    def _compute_dz(self):
        return self.loss_function.gradient(self.activations, self.y)

    def initialize(self):
        self._initialize_backward()