__author__ = 'dvgodoy'

from neurons import Input, Hidden, Output
from losses import LogLoss
from scipy.io import loadmat
import os

from activations import SigmoidActivation, ReluActivation, TanhActivation
import numpy as np
from matplotlib import pyplot as plt


class Layer(object):
    classes = ['Input', 'Hidden', 'Output']

    def __init__(self, n_units, layer_type, **kwargs):
        assert layer_type in self.classes
        self._charged = False
        self.n_units = n_units
        self.layer_type = layer_type
        neuron = globals()[layer_type]
        self.units = [neuron(**kwargs) for _ in range(n_units)]

    def connect(self, layer):
        assert isinstance(layer, Layer)
        for source in self.units:
            for destination in layer.units:
                source.connect(destination)

    def fire(self):
        mu, var = self.batch_norm()
        for unit in self.units:
            unit.fire(mu, var)

    def feedback(self):
        for unit in self.units:
            unit.feedback()

    def examples(self, X):
        assert X.shape[0] == self.n_units
        for i, unit in enumerate(self.units):
            unit.examples(X[i, np.newaxis, :])

    def responses(self, Y):
        assert Y.shape[0] == self.n_units
        for i, unit in enumerate(self.units):
            unit.responses(Y[i, np.newaxis, :])

    def batch_norm(self):
        mu, var = 0.0, 1.0
        if np.all([unit.charged for unit in self.units]):
            z = np.array([unit._z for unit in self.units])
            mu = z.mean(axis = 0)
            var = z.var(axis = 0)
        return mu, var

    @property
    def activations(self):
        activations = np.array([unit.activations for unit in self.units])
        return activations.reshape(activations.shape[0], -1)

    @property
    def weights(self):
        weights = np.array([unit.weights for unit in self.units])
        return weights.reshape(weights.shape[0], -1)

    @property
    def biases(self):
        biases = np.array([unit.biases for unit in self.units])
        return biases.reshape(biases.shape[0], -1)

    @property
    def cost(self):
        return np.array([unit.cost for unit in self.units])


if __name__ == '__main__':
    np.random.seed(13)

    name = 'var_u'
    data = loadmat(os.path.join('../data', name + '.mat'))

    sigmoid = SigmoidActivation()
    relu = ReluActivation()
    tanh = TanhActivation()

    l_i = Layer(12, 'Input')
    l_h1 = Layer(10, 'Hidden', activation_function=relu)
    l_h2 = Layer(7, 'Hidden', activation_function=relu)
    l_h3 = Layer(5, 'Hidden', activation_function=relu)
    l_h4 = Layer(4, 'Hidden', activation_function=relu)
    l_h5 = Layer(3, 'Hidden', activation_function=relu)
    l_o = Layer(1, 'Output', activation_function=sigmoid, loss_function=LogLoss(), learning_rate=0.0004)
    layers = [l_i, l_h1, l_h2, l_h3, l_h4, l_h5, l_o]

    l_i.connect(l_h1)
    l_h1.connect(l_h2)
    l_h2.connect(l_h3)
    l_h3.connect(l_h4)
    l_h4.connect(l_h5)
    l_h5.connect(l_o)

    epochs = 10000
    h1_weight_history = []
    #h2_weight_history = []
    #h3_weight_history = []
    cost_history = []
    examples = data['F'].reshape(12, -1)
    responses = data['y']
    batch_size = 4096
    for epoch in range(epochs):
        if not (epoch % 100):
            print(epoch)
        for n_batch in range(1):
            examples = data['F'].reshape(12, -1)[:, (n_batch * batch_size):(n_batch * batch_size + batch_size)]
            responses = data['y'][:, (n_batch * batch_size):(n_batch * batch_size + batch_size)]
            l_i.examples(examples)
            l_o.responses(responses)
            for layer in layers:
                layer.fire()
            cost_history.append(l_o.cost)
            h1_weight_history.append(l_h1.weights)
            #h2_weight_history.append(l_h2.weights)
            #h3_weight_history.append(l_h3.weights)
            for layer in layers[::-1]:
                layer.feedback()
        #print(l_o.activations)
        #print(l_o.cost)
    print(l_o.activations)
    predictions = l_o.activations > 0.5
    print((predictions == responses).mean())
    #h1_mean = [w.mean() for w in h1_weight_history]
    #h2_mean = [w.mean() for w in h2_weight_history]
    #h3_mean = [w.mean() for w in h3_weight_history]
    #h1_std = [w.std() for w in h1_weight_history]
    #h2_std = [w.std() for w in h2_weight_history]
    #h3_std = [w.std() for w in h3_weight_history]
    #plt.plot(h1_mean, 'k')
    #plt.plot(h2_mean, 'r')
    #plt.plot(h3_mean, 'g')
    #plt.plot(h1_std, 'k--')
    #plt.plot(h2_std, 'r--')
    #plt.plot(h3_std, 'g--')
    plt.plot(cost_history)
    plt.show()