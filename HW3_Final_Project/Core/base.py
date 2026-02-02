import numpy as np
from abc import ABC, abstractmethod

class Layer(ABC):
    """
    Abstract Base Class for all neural network layers.
    Establishes the contract for Forward, Backward, and Parameter handling.
    """
    
    def __init__(self):
        # Flag to indicate if the layer has trainable parameters
        self.trainable = True
        # Layer name can be useful for visualization and debugging
        self.name = self.__class__.__name__

    @abstractmethod
    def forward(self, x):
        """
        Computes the output of the layer.
        Args:
            x: Input tensor (numpy array).
        Returns:
            Output tensor.
        """
        pass

    @abstractmethod
    def backward(self, output_gradient, **kwargs):
        """
        Computes the gradient w.r.t input and updates parameters if trainable.
        
        Args:
            output_gradient: Gradient of the loss w.r.t the output of this layer.
            kwargs: Dictionary of optimization parameters (e.g., learning rates).
                    Example: {'lr': 0.01, 'lr_centers': 0.001}
        
        Returns:
            input_gradient: Gradient of the loss w.r.t the input of this layer.
        """
        pass

    def get_params(self):
        """
        Returns all trainable parameters as a single flattened vector.
        REQUIRED ONLY FOR EKF (Question 3).
        
        Returns:
            np.array: Column vector (N, 1) or empty array if no params.
        """
        # Default implementation allows layers without EKF support to work in standard mode
        return np.zeros((0, 1))

    def set_params(self, params):
        """
        Sets the layer parameters from a flattened vector.
        REQUIRED ONLY FOR EKF (Question 3).
        
        Args:
            params: np.array of shape (N, 1).
        """
        # Default: Do nothing
        pass

    def compute_jacobian(self, chain_grad=None):
        """
        Computes the Jacobian matrix of the layer outputs w.r.t parameters.
        REQUIRED ONLY FOR EKF (Question 3).
        
        Args:
            chain_grad: Gradient from the next layer (chain rule).
            
        Returns:
            np.array: Jacobian matrix.
        """
        # Raises error to alert user if they try to use EKF on unsupported layers
        raise NotImplementedError(f"EKF Jacobian not implemented for {self.name}")

    def reset_state(self):
        """
        Resets the internal memory state of the layer.
        CRITICAL FOR RECURRENT LAYERS (Question 2).
        For non-recurrent layers, this does nothing.
        """
        pass

    def __repr__(self):
        return f"<{self.name}>"


class Loss(ABC):
    """
    Abstract Base Class for loss functions.
    """
    
    @abstractmethod
    def compute(self, y_pred, y_true):
        """
        Computes the scalar loss value.
        Used for monitoring training progress (Visualization).
        """
        pass

    @abstractmethod
    def gradient(self, y_pred, y_true):
        """
        Computes the gradient of the loss w.r.t the network output (y_pred).
        This starts the Backpropagation process.
        """
        pass


class Optimizer(ABC):
    """
    Abstract Base Class for global optimizers (like EKF).
    Standard SGD is usually implemented directly in Layer.backward or models.py,
    but complex ones like EKF need a class.
    """
    
    @abstractmethod
    def update(self, model, x, y):
        """
        Performs a parameter update step.
        Args:
            model: The entire Neural Network model.
            x: Input data.
            y: Target data.
        """
        pass