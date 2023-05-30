import numpy as np
from abc import ABCMeta, abstractmethod
from .functional import Functional


class Cost(metaclass=ABCMeta):

    @abstractmethod
    def fn(self, a, y, from_logits=False):
        pass

    @abstractmethod
    def init_delta(self, z, a, y):
        pass


class QuadraticCost(Cost):

    def __init__(self, activation='softmax'):
        self.activation = None
        self.activation_prime = None
        if activation is not None:
            self.activation = getattr(Functional, activation, None)
            self.activation_prime = getattr(Functional, f'{activation}_prime', None)

    def fn(self, a, y, from_logits=False):
        if from_logits and self.activation is not None:
            a = self.activation(a)
        return 0.5 * np.sum((a - y) ** 2)

    def init_delta(self, z, a, y):
        """

        Args:
            z: 2-D array
            a: 2-D array
            y: 2-D array

        Returns:

        """
        d = a - y
        if self.activation_prime is not None:
            sd = self.activation_prime(z)
            if len(sd.shape) == 3:
                res = np.array([i_d @ i_sd for i_d, i_sd in zip(d, sd)])
            else:
                res = d * sd
        else:
            res = d
        return res


class CrossEntropyCost(Cost):

    def __init__(self, activation='softmax'):
        if activation is not None:
            self.activation = getattr(Functional, activation, None)
            self.activation_prime = getattr(Functional, f'{activation}_prime', None)

    def fn(self, a, y, from_logits=False):
        if from_logits and self.activation is not None:
            a = self.activation(a)
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    def init_delta(self, z, a, y):
        """

        Args:
            z: 2-D array
            a: 2-D array
            y: 2-D array

        Returns:

        """
        d = np.nan_to_num(-y / a + (1 - y) / (1 - a))
        if self.activation_prime is not None:
            sd = self.activation_prime(z)
            if len(sd.shape) == 3:
                res = np.array([i_d @ i_sd for i_d, i_sd in zip(d, sd)])
            else:
                res = d * sd
        else:
            res = d
        return res
