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
    #x2 = np.array([5,0,7], ndmin=2)
    #x2 = np.array([7,0,7], ndmin=2)
    x2 = x2 - np.mean(x2)
    y = np.array([1,1,0], ndmin=2)

    i1 = Input()
    i2 = Input()

    h11 = Hidden(relu)
    h12 = Hidden(relu)

    h21 = Hidden(relu)
    h22 = Hidden(relu)

    o = Output(sigmoid, LogLoss())

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

    i1.examples(x1)
    i2.examples(x2)
    o.responses(y)

    learning_rate = 0.3
    epochs = 1000

    history = []
    for _ in range(epochs):
        # Forward propagation steps
        ## 1st layer
        h11.forward_propagation()
        h12.forward_propagation()
        ## 2nd layer
        h21.forward_propagation()
        h22.forward_propagation()
        ## output layer
        o.forward_propagation()

        cost = o.compute_cost()
        history.append(cost)

        # Backpropagation steps
        ## output layer
        o.back_propagation()
        o.update_parameters(learning_rate)

        ## 2nd layer
        h21.back_propagation()
        h21.update_parameters(learning_rate)
        h22.back_propagation()
        h22.update_parameters(learning_rate)

        ## 1st layer
        h11.back_propagation()
        h11.update_parameters(learning_rate)
        h12.back_propagation()
        h12.update_parameters(learning_rate)

    print(np.array(history))
    print(h11.weights)
    print(h12.weights)
    print(h21.weights)
    print(h22.weights)
    print(o.weights)
    print(h11.activations)
    print(h12.activations)
    print(h21.activations)
    print(h22.activations)
    print(o.activations)