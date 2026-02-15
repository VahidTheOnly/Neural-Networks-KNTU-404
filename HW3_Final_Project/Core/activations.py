import numpy as np
from Core.base import Layer

class Linear(Layer):
    def forward(self, x):
        return x
    
    def backward(self, output_gradient, **kwargs):
        return output_gradient

class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, output_gradient, **kwargs):
        return output_gradient * (self.out * (1 - self.out))

class TrigonometricActivation(Layer):
    def __init__(self, input_dim, mode='sin_cos'):
        super().__init__()
        self.input_dim = input_dim
        self.mode = mode
        
        self.alpha = np.random.uniform(0.5, 1.5, (input_dim, 1))
        self.theta = np.random.uniform(-0.1, 0.1, (input_dim, 1))
        
        self.x = None
        self.net_trig = None 

    def forward(self, x):
        self.x = x.reshape(-1, 1)
        self.net_trig = self.alpha * self.x + self.theta
        
        if self.mode == 'sin_cos':
            return np.sin(self.net_trig) + np.cos(self.net_trig)
        else:
            return np.sin(self.net_trig)

    def backward(self, output_gradient, **kwargs):
        lr = kwargs.get('lr_trig', kwargs.get('lr', 0.01))
        
        if self.mode == 'sin_cos':
            derivative_term = np.cos(self.net_trig) - np.sin(self.net_trig)
        else:
            derivative_term = np.cos(self.net_trig)
        
        grad_input = output_gradient * self.alpha * derivative_term
        grad_alpha = output_gradient * self.x * derivative_term
        grad_theta = output_gradient * derivative_term
        
        if self.trainable:
            self.alpha -= lr * grad_alpha
            self.theta -= lr * grad_theta
            
        return grad_input

    def get_params(self):
        return np.concatenate([self.alpha.flatten(), self.theta.flatten()]).reshape(-1, 1)

    def set_params(self, params):
        size = self.alpha.size
        self.alpha = params[:size].reshape(self.alpha.shape)
        self.theta = params[size:].reshape(self.theta.shape)

    def compute_jacobian(self, chain_grad=None):
        if self.mode == 'sin_cos':
            derivative_term = np.cos(self.net_trig) - np.sin(self.net_trig)
        else:
            derivative_term = np.cos(self.net_trig)

        if chain_grad is not None:
            grad_upstream = chain_grad.reshape(derivative_term.shape)
        else:
            grad_upstream = 1.0

        j_alpha = (self.x * derivative_term) * grad_upstream
        j_theta = derivative_term * grad_upstream
        
        return np.hstack([j_alpha.flatten(), j_theta.flatten()]).reshape(1, -1)
    
    def get_input_derivative_for_ekf(self):
        if self.mode == 'sin_cos':
            derivative_term = np.cos(self.net_trig) - np.sin(self.net_trig)
        else:
            derivative_term = np.cos(self.net_trig)
            
        return (self.alpha * derivative_term).T