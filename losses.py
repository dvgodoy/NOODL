import abc
import numpy as np

__author__ = 'dvgodoy'


class Loss(object):
    __metaclass__ = abc.ABCMeta

    def __str__(self):
        return 'Loss Function'

    @abc.abstractmethod
    def compute(self, y_hat, y):
        pass

    @abc.abstractmethod
    def derivative(self):
        def func(y_hat, y):
            return 0
        return func

    def gradient(self, y_hat, y):
        return self.derivative()(y_hat, y)


class LogLoss(Loss):
    def __str__(self):
        return 'Logistic Loss Function'

    def compute(self, y_hat, y):
        return - (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    def derivative(self):
        def func(y_hat, y):
            return y_hat - y
        return func
