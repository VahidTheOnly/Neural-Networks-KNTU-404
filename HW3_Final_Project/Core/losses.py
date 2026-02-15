import numpy as np
from Core.base import Loss

class MSELoss(Loss):
    
    def compute(self, y_pred, y_true):
        y_true = y_true.reshape(y_pred.shape)
        return 0.5 * np.mean((y_pred - y_true) ** 2)

    def gradient(self, y_pred, y_true):
        y_true = y_true.reshape(y_pred.shape)
        return (y_pred - y_true)


class RoughMSELoss(Loss):
    def compute(self, y_pred, y_true):
        split_idx = y_pred.shape[0] // 2
        y_L = y_pred[:split_idx]
        y_U = y_pred[split_idx:]
        
        y_true = y_true.reshape(y_L.shape)
        
        loss_L = 0.5 * np.mean((y_L - y_true) ** 2)
        loss_U = 0.5 * np.mean((y_U - y_true) ** 2)
        
        return loss_L + loss_U

    def gradient(self, y_pred, y_true):
        split_idx = y_pred.shape[0] // 2
        y_L = y_pred[:split_idx]
        y_U = y_pred[split_idx:]
        
        y_true = y_true.reshape(y_L.shape)
        
        grad_L = (y_L - y_true)
        grad_U = (y_U - y_true)
        
        return np.concatenate([grad_L, grad_U], axis=0)


class RobustLogCoshLoss(Loss):
    def compute(self, y_pred, y_true):
        y_true = y_true.reshape(y_pred.shape)
        error = y_pred - y_true
        return np.mean(np.log(np.cosh(error)))

    def gradient(self, y_pred, y_true):
        y_true = y_true.reshape(y_pred.shape)
        error = y_pred - y_true
        return np.tanh(error)


class RoughRobustLoss(Loss):
    def compute(self, y_pred, y_true):
        split_idx = y_pred.shape[0] // 2
        y_L = y_pred[:split_idx]
        y_U = y_pred[split_idx:]
        y_true = y_true.reshape(y_L.shape)
        
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