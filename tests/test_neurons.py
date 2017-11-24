__author__ = 'dvgodoy'

from neurons import Input, Hidden, Output
from losses import LogLoss
from activations import SigmoidActivation, ReluActivation
import numpy as np

class Layer(object):
    classes = ['Input', 'Hidden', 'Output']

    def __init__(self, n_units, layer_type, **kwargs):
        assert layer_type in self.classes
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
        for unit in self.units:
            unit.fire()

    def feedback(self):
        for unit in self.units:
            unit.feedback()

    def examples(self, X):
        assert X.shape[0] == self.n_units
        for i, unit in enumerate(self.units):
            unit.examples(X[i, :])

    def responses(self, Y):
        assert Y.shape[0] == self.n_units
        for i, unit in enumerate(self.units):
            unit.responses(Y[i, :])

    @property
    def activations(self):
        return np.array([unit.activations for unit in self.units])

    @property
    def weights(self):
        return np.array([unit.weights for unit in self.units])

    @property
    def biases(self):
        return np.array([unit.biases for unit in self.units])

    @property
    def cost(self):
        return np.array([unit.cost for unit in self.units])

if __name__ == '__main__':
    np.random.seed(13)

    sigmoid = SigmoidActivation()
    relu = ReluActivation()

    l_i = Layer(2, 'Input')
    l_h1 = Layer(2, 'Hidden', activation_function=relu)
    l_h2 = Layer(2, 'Hidden', activation_function=relu)
    l_o = Layer(1, 'Output', activation_function=sigmoid, loss_function=LogLoss(), learning_rate=0.3)
    layers = [l_i, l_h1, l_h2, l_o]

    l_i.connect(l_h1)
    l_h1.connect(l_h2)
    l_h2.connect(l_o)

    x1 = np.array([1,2,3], ndmin=2)
    x1 = x1 - np.mean(x1)
    x2 = np.array([2,0,7], ndmin=2)
    x2 = x2 - np.mean(x2)
    y = np.array([1,1,0], ndmin=2)

    """
    i1 = Input()
    i2 = Input()

    h11 = Hidden(relu)
    h12 = Hidden(relu)

    h21 = Hidden(relu)
    h22 = Hidden(relu)

    learning_rate = 0.3
    o = Output(sigmoid, LogLoss(), learning_rate)

    i1.connect(h11)
    i1.connect(h12)
    i2.connect(h11)
    i2.connect(h12)
    #i2.connect(h22)

    h11.connect(h21)
    h11.connect(h22)
    h12.connect(h21)
    h12.connect(h22)

    h21.connect(o)
    h22.connect(o)

    i1.initialize()
    i2.initialize()
    o.initialize()

    neurons = sorted([h11, o, i1, i2, h12, h21, h22], key=lambda o: o.layer)
    """

    epochs = 1000
    for _ in range(epochs):
        l_i.examples(np.array([x1, x2]))
        l_o.responses(np.array([y]))
        for layer in layers:
            layer.fire()
        print(l_h1.weights)
        #print(l_o.cost)
        for layer in layers[::-1]:
            layer.feedback()
    print(l_o.activations)
    print(l_o.activations.shape)
    print(l_h1.weights.shape)

    #for _ in range(epochs):
        #o.responses(y)
        #i1.examples(x1)
        #i2.examples(x2)
        #for neuron in neurons:
        #    neuron.fire()
        #print(o.cost)
        #for neuron in neurons[::-1]:
        #    neuron.feedback()
    #print(o.activations)
