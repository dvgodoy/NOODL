import abc
import numpy as np

__author__ = 'dvgodoy'


class Activation(object):
    __metaclass__ = abc.ABCMeta

    def __repr__(self):
        return 'Activation Function'

    @abc.abstractmethod
    def evaluate(self, z):
        pass

    @abc.abstractmethod
    def derivative(self):
        def func(z):
            return 0
        return func

    def gradient(self, z):
        return self.derivative()(z)


class LinearActivation(Activation):
    def __repr__(self):
        return 'Linear Activation Function'

    def evaluate(self, z):
        return z

    def derivative(self):
        def func(z):
            return 1
        return func


class SigmoidActivation(Activation):
    def __repr__(self):
        return 'Sigmoid Activation Function'

    def evaluate(self, z):
        return 1. / (1 + np.exp(-z))

    def derivative(self):
        def func(z):
            a = self.evaluate(z)
            return a * (1 - a)
        return func


class TanhActivation(Activation):
    def __repr__(self):
        return 'Tanh Activation Function'

    def evaluate(self, z):
        e1 = np.exp(z)
        e2 = np.exp(-z)
        return (e1 - e2) / (e1 + e2)

    def derivative(self):
        def func(z):
            return 1 - self.evaluate() ** 2
        return func


class ReluActivation(Activation):
    def __repr__(self):
        return 'ReLU Activation Function'

    def evaluate(self, z):
        return np.maximum(0, z)

    def derivative(self):
        def func(z):
            return z >= 0
        return func


class LeakyReluActivation(Activation):
    def __repr__(self):
        return 'Leaky ReLU Activation Function'

    def evaluate(self, z):
        return np.maximum(0.01 * z, z)

    def derivative(self):
        def func(z):
            return (z >= 0) * 0.99 + 0.01
        return func
