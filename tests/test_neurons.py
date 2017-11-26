__author__ = 'dvgodoy'

from neurons import Input, Hidden, Output
from losses import LogLoss
from activations import SigmoidActivation, ReluActivation
import numpy as np

if __name__ == '__main__':
    np.random.seed(13)

    sigmoid = SigmoidActivation()
    relu = ReluActivation()

    x1 = np.array([1,2,3], ndmin=2)
    x1 = x1 - np.mean(x1)
    x2 = np.array([2,0,7], ndmin=2)
    x2 = x2 - np.mean(x2)
    y = np.array([1,1,0], ndmin=2)

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

    epochs = 1000
    for _ in range(epochs):
        o.responses(y)
        i1.examples(x1)
        i2.examples(x2)
        for neuron in neurons:
            neuron.fire()
        print(o.cost)
        for neuron in neurons[::-1]:
            neuron.feedback()
    print(o.activations)
