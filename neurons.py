__author__ = 'dvgodoy'

from activations import Activation
from losses import Loss
import numpy as np
import abc
import uuid

class Node(object):
    def __init__(self):
        self.id = uuid.uuid4().hex

        self._input_nodes = []
        self._input_ids = np.array([])
        self._input_path_lengths = np.zeros(0)

        self._output_nodes = []
        self._output_ids = np.array([])
        self._output_path_lengths = np.zeros(0)

        self._layer = 0

    @property
    def node_id(self):
        return self.id

    @property
    def layer(self):
        return self._layer

    def _node_index(self, node_ids, node):
        try:
            idx = np.where(node_ids == node.node_id)[0][0]
        except (IndexError, AttributeError):
            idx = -1
        return idx

    def add_output(self, node):
        self._output_nodes.append(node)
        self._output_ids = np.append(self._output_ids, [[node.node_id]])
        self._output_path_lengths = np.zeros(len(self._output_nodes))

    def add_input(self, node):
        self._input_nodes.append(node)
        self._input_ids = np.append(self._input_ids, [[node.node_id]])
        self._input_path_lengths = np.zeros(len(self._input_nodes))

    def _initialize_forward(self, node=None, path_length=0):
        if path_length > self.layer:
            self._layer = path_length

        i = self._node_index(self._input_ids, node)
        if i >= 0:
            self._input_path_lengths[i] = path_length

        for node in self._output_nodes:
            node._initialize_forward(self, path_length + 1)

    def _initialize_backward(self, node=None, path_length=0):
        i = self._node_index(self._output_ids, node)
        if i >= 0:
            self._output_path_lengths[i] = path_length

        for node in self._input_nodes:
            node._initialize_backward(self, path_length + 1)


class Neuron(Node):
    def __init__(self):
        super(Neuron, self).__init__()
        self._charged = False
        self._updated = False
        self._learning_rate = 0.3

        self._bias = np.zeros((1, 1))
        self._weights = np.zeros((1, 0))
        self._activations = np.zeros((1, 0))

        self._db = np.zeros((1, 0))
        self._dw = np.zeros((1, 0))
        self._dz = np.zeros((1, 0))
        self._da = np.zeros((1, 0))

        self._inputs = np.zeros((1, 0))
        self._input_activated = np.array([])

        self._gradients = np.zeros((1, 0))
        self._output_feedback = np.array([])

        self._auto_prop = False

        self._m = 0

    @property
    def weights(self):
        return self._weights

    @property
    def bias(self):
        return self._bias

    @property
    def activations(self):
        return self._activations

    @property
    def charged(self):
        return self._charged

    def update_parameters(self, learning_rate):
        self._weights = self._weights - learning_rate * self._dw
        self._bias = self._bias - learning_rate * self._db

    def add_output(self, neuron):
        super(Neuron, self).add_output(neuron)
        self._gradients = np.zeros((len(self._output_nodes), 0))
        self._output_feedback = np.zeros(len(self._output_nodes))

    def add_input(self, neuron):
        super(Neuron, self).add_input(neuron)
        n_weights = self._weights.shape[1] + 1
        self._weights = (np.random.randn(1, n_weights) * np.sqrt(2. / n_weights)).reshape(1, -1)
        self._inputs = np.zeros((len(self._input_nodes), 0))
        self._input_activated = np.zeros(len(self._input_nodes))

    def connect(self, neurons):
        assert neurons is not None
        if not isinstance(neurons, list):
            neurons = [neurons]
        assert isinstance(neurons[0], Neuron)

        for neuron in neurons:
            # prevents connecting twice to same neuron
            if self._node_index(self._output_ids, neuron) < 0:
                # adds destination neuron to outputlist
                self.add_output(neuron)
                # at the destination neuron, adds source neuron as input
                neuron.add_input(self)

    @abc.abstractmethod
    def forward_propagation(self, mu=0.0, var=1.0):
        pass

    @abc.abstractmethod
    def back_propagation(self, learning_rate):
        pass

    def gradient_from(self, da, neuron, learning_rate):
        i = self._node_index(self._output_ids, neuron)
        if i >= 0:
            self._learning_rate = learning_rate
            self._output_feedback[i] = 1.
            if self._gradients.shape[1] != da.shape[1]:
                self._gradients = np.zeros((len(self._output_nodes), da.shape[1]))
            self._gradients[i, np.newaxis, :] = da

            if np.all(self._output_feedback):
                self._updated = True
                if self._auto_prop:
                    self.feedback()

    def feedback(self):
        if self._updated:
            self._output_feedback -= 1.
            self.back_propagation(self._learning_rate)

    def input_from(self, activations, neuron):
        i = self._node_index(self._input_ids, neuron)
        if i >= 0:
            self._input_activated[i] = 1.
            if self._inputs.shape[1] != activations.shape[1]:
                self._inputs = np.zeros((len(self._input_nodes), activations.shape[1]))
            self._inputs[i, np.newaxis, :] = activations

            if np.all(self._input_activated):
                assert self._inputs.shape[1] > 0
                assert self._weights.shape[1] == self._inputs.shape[0]
                assert self._bias.shape == (1, 1)
                self._charged = True
                self._z = np.dot(self._weights, self._inputs) + self._bias
                if self._auto_prop:
                    self.fire()

    def fire(self, mu=0.0, var=1.0):
        if self._charged:
            self._input_activated -= 1.
            self.forward_propagation(mu, var)


class Input(Neuron):
    def __init__(self, **kwargs):
        super(Input, self).__init__()
        self._activations = np.zeros((1, 0))

    def __str__(self):
        return 'Input Neuron {}'.format(self._activations.shape)

    def examples(self, X):
        assert isinstance(X, np.ndarray)
        assert X.shape[0] == 1

        self._m = X.shape[1]
        self._z = X
        self._activations = X
        self.forward_propagation()

    def forward_propagation(self, mu=0.0, var=1.0):
        for neuron in self._output_nodes:
            neuron.input_from(self._activations, self)

    def back_propagation(self, learning_rate):
        pass

    def initialize(self):
        self._initialize_forward()


class Hidden(Neuron):
    def __init__(self, activation_function, **kwargs):
        super(Hidden, self).__init__()
        assert isinstance(activation_function, Activation)

        self._activation_function = activation_function
        self._inputs = np.zeros((1, 0))

    def __str__(self):
        return 'Hidden Neuron {}'.format(self._inputs.shape)

    def forward_propagation(self, mu=0.0, var=1.0):
        # self._z = np.dot(self._weights, self._inputs) + self._bias
        self._z = (self._z - mu) / np.sqrt(var + 1e-7)
        self._activations = self._activation_function.evaluate(self._z)
        for neuron in self._output_nodes:
            neuron.input_from(self._activations, self)

    def _compute_dz(self):
        return np.sum(self._gradients, axis=0, keepdims=True) * self._activation_function.gradient(self._z)

    def back_propagation(self, learning_rate):
        self._dz = self._compute_dz()
        self._dw = np.mean(self._dz * self._inputs, axis=1).reshape(1, -1)
        self._db = np.mean(self._dz, keepdims=True)
        self._da = np.dot(self._weights.T, self._dz)
        self.update_parameters(learning_rate)
        for i, neuron in enumerate(self._input_nodes):
            gradient = self._da[i, np.newaxis, :]
            neuron.gradient_from(gradient, self, learning_rate)


class Output(Hidden):
    def __init__(self, activation_function, loss_function, learning_rate=0.3):
        super(Output, self).__init__(activation_function)
        assert isinstance(loss_function, Loss)
        self._loss_function = loss_function
        if not callable(learning_rate):
            constant = learning_rate
            learning_rate = lambda epochs: constant
        self._learning_rate = learning_rate
        self._y = np.zeros((1, 0))
        self._output_feedback = np.zeros(1)
        self._epochs = 0
        self._cost = np.inf

    def __str__(self):
        return 'Output Neuron {}'.format(self._y.shape)

    @property
    def cost(self):
        return self._cost

    def responses(self, y):
        assert isinstance(y, np.ndarray)
        assert y.shape[0] == 1
        self._y = y

    def forward_propagation(self, mu=0.0, var=1.0):
        super(Output, self).forward_propagation()
        self._cost = self.compute_cost()
        self._epochs += 1
        self._updated = True

    def back_propagation(self, learning_rate=None):
        learning_rate = self._learning_rate(self._epochs)
        super(Output, self).back_propagation(learning_rate)

    def compute_cost(self):
        assert self._activations.shape == self._y.shape
        self._output_feedback += 1.
        return np.mean(self._loss_function.compute(self._activations, self._y))

    def _compute_dz(self):
        return self._loss_function.gradient(self._activations, self._y)

    def initialize(self):
        self._initialize_backward()