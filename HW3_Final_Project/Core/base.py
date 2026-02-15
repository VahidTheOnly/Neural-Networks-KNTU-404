import numpy as np
from abc import ABC, abstractmethod

class Layer(ABC):
    
    def __init__(self):
        self.trainable = True
        self.name = self.__class__.__name__

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, output_gradient, **kwargs):
        pass

    def get_params(self):
        return np.zeros((0, 1))

    def set_params(self, params):
        pass

    def compute_jacobian(self, chain_grad=None):
        raise NotImplementedError(f"EKF Jacobian not implemented for {self.name}")

    def reset_state(self):
        pass

    def __repr__(self):
        return f"<{self.name}>"


class Loss(ABC):
    @abstractmethod
    def compute(self, y_pred, y_true):
        pass

    @abstractmethod
    def gradient(self, y_pred, y_true):
        pass


class Optimizer(ABC):
    @abstractmethod
    def update(self, model, x, y):
        pass