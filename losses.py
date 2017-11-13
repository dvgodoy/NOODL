import abc
import numpy as np

__author__ = 'dvgodoy'


class Loss(object):
    __metaclass__ = abc.ABCMeta

    def __repr__(self):
        return 'Loss Function'

    @abc.abstractmethod
    def compute(self, y_hat, y):
        pass

    @abc.abstractmethod
    def derivative(self):
        def func(y_hat, y):
            return 1
        return func

    def gradient(self, y_hat, y):
        return self.derivative()(y_hat, y)


class LogisticLoss(Loss):
    def __repr__(self):
        return 'Logistic Loss Function'

    def compute(self, y_hat, y):
        return y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)

    def derivative(self):
        def func(y_hat, y):
            return y_hat - y
        return func
