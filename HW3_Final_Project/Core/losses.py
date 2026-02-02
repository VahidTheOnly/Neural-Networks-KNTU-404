import numpy as np
from Core.base import Loss

class MSELoss(Loss):
    """
    Mean Squared Error (L2 Loss).
    Standard loss for regression problems.
    Formula: L = 0.5 * mean((y_pred - y_true)^2)
    """
    def compute(self, y_pred, y_true):
        # Ensure shapes match
        y_true = y_true.reshape(y_pred.shape)
        return 0.5 * np.mean((y_pred - y_true) ** 2)

    def gradient(self, y_pred, y_true):
        """
        Gradient of MSE: (y_pred - y_true)
        Note: We usually omit the 1/N term in backprop for stronger gradients 
        in stochastic updates, effectively summing the error.
        """
        y_true = y_true.reshape(y_pred.shape)
        return (y_pred - y_true)


class RoughMSELoss(Loss):
    """
    Rough Mean Squared Error.
    Used when the network output is a concatenation of Lower and Upper bounds.
    
    Logic:
    The loss is the sum of MSE for the Lower bound and MSE for the Upper bound.
    Target (y_true) is usually crisp, so we compare both bounds to the same target.
    
    Formula: L = 0.5 * (y_L - y_true)^2 + 0.5 * (y_U - y_true)^2
    """
    def compute(self, y_pred, y_true):
        """
        y_pred: Shape (2*N, 1). First half is Lower, second half is Upper.
        y_true: Shape (N, 1). Crisp target.
        """
        # Split prediction into Lower and Upper
        split_idx = y_pred.shape[0] // 2
        y_L = y_pred[:split_idx]
        y_U = y_pred[split_idx:]
        
        # Ensure target shape matches split size
        y_true = y_true.reshape(y_L.shape)
        
        loss_L = 0.5 * np.mean((y_L - y_true) ** 2)
        loss_U = 0.5 * np.mean((y_U - y_true) ** 2)
        
        return loss_L + loss_U

    def gradient(self, y_pred, y_true):
        """
        Returns a gradient vector of shape (2*N, 1).
        Top half: Gradient for Lower outputs.
        Bottom half: Gradient for Upper outputs.
        """
        split_idx = y_pred.shape[0] // 2
        y_L = y_pred[:split_idx]
        y_U = y_pred[split_idx:]
        
        y_true = y_true.reshape(y_L.shape)
        
        grad_L = (y_L - y_true)
        grad_U = (y_U - y_true)
        
        return np.concatenate([grad_L, grad_U], axis=0)


class RobustLogCoshLoss(Loss):
    """
    Robust Loss Function (Log-Hyperbolic Cosine).
    Required for Question 3 (Parts c, d, f) - Chaos and Noise Management.
    
    Why this is 'Innovative' and Robust:
    - For small x: log(cosh(x)) is approx x^2 / 2 (Like MSE).
    - For large x: log(cosh(x)) is approx |x| - log(2) (Like L1 / MAE).
    
    Key Property:
    The gradient is tanh(x). It saturates at +1 and -1.
    This means large errors (outliers/noise spikes) caused by the Butterfly Effect
    will NOT produce massive gradients that destabilize the network.
    MSE gradient is linear (unbounded), which is why it fails with heavy noise.
    """
    def compute(self, y_pred, y_true):
        y_true = y_true.reshape(y_pred.shape)
        error = y_pred - y_true
        return np.mean(np.log(np.cosh(error)))

    def gradient(self, y_pred, y_true):
        y_true = y_true.reshape(y_pred.shape)
        error = y_pred - y_true
        # Gradient of log(cosh(x)) is tanh(x)
        return np.tanh(error)


class RoughRobustLoss(Loss):
    """
    Combination of Rough Sets and Robust Loss.
    Specifically for Question 3-d: "Improve... using Rough concepts AND Cost Function innovation".
    
    Applies LogCosh loss to both Lower and Upper bounds.
    """
    def compute(self, y_pred, y_true):
        split_idx = y_pred.shape[0] // 2
        y_L = y_pred[:split_idx]
        y_U = y_pred[split_idx:]
        y_true = y_true.reshape(y_L.shape)
        
        # Robust loss on both bounds
        loss_L = np.mean(np.log(np.cosh(y_L - y_true)))
        loss_U = np.mean(np.log(np.cosh(y_U - y_true)))
        
        return loss_L + loss_U

    def gradient(self, y_pred, y_true):
        split_idx = y_pred.shape[0] // 2
        y_L = y_pred[:split_idx]
        y_U = y_pred[split_idx:]
        y_true = y_true.reshape(y_L.shape)
        
        grad_L = np.tanh(y_L - y_true)
        grad_U = np.tanh(y_U - y_true)
        
        return np.concatenate([grad_L, grad_U], axis=0)