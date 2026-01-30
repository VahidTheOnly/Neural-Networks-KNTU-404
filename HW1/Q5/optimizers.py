import numpy as np

class BaseOptimizer:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def update(self, params, grads):
        
        raise NotImplementedError

class SGD(BaseOptimizer):
    def update(self, params, grads):
        return params - self.lr * grads

class Momentum(BaseOptimizer):
    def __init__(self, learning_rate, beta=0.9):
        super().__init__(learning_rate)
        self.beta = beta
        self.v = None  # Velocity

    def update(self, params, grads):
        if self.v is None:
            self.v = np.zeros_like(params)
        self.v = self.beta * self.v + (1 - self.beta) * grads
        return params - self.lr * self.v


class Nesterov(BaseOptimizer):
    def __init__(self, learning_rate, beta=0.9):
        super().__init__(learning_rate)
        self.beta = beta
        self.v = None  # Velocity vector

    def update(self, params, grads):
        if self.v is None:
            self.v = np.zeros_like(params)

        v_prev = self.v.copy()
        
        self.v = self.beta * self.v - self.lr * grads
        
        params_update = -self.beta * v_prev + (1 + self.beta) * self.v
        
        return params + params_update


class AdaGrad(BaseOptimizer):
    def __init__(self, learning_rate, epsilon=1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.cache = None

    def update(self, params, grads):
        if self.cache is None:
            self.cache = np.zeros_like(params)
        self.cache += grads**2
        return params - self.lr * grads / (np.sqrt(self.cache) + self.epsilon)

class RMSProp(BaseOptimizer):
    def __init__(self, learning_rate, decay_rate=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = None

    def update(self, params, grads):
        if self.cache is None:
            self.cache = np.zeros_like(params)
        self.cache = self.decay_rate * self.cache + (1 - self.decay_rate) * grads**2
        return params - self.lr * grads / (np.sqrt(self.cache) + self.epsilon)

class AdaDelta(BaseOptimizer):
    def __init__(self, learning_rate=1.0, rho=0.95, epsilon=1e-8):
        super().__init__(learning_rate)
        self.rho = rho
        self.epsilon = epsilon
        self.E_grad = None   # Running average of squared gradients
        self.E_delta = None  # Running average of squared parameter updates

    def update(self, params, grads):
        if self.E_grad is None:
            self.E_grad = np.zeros_like(params)
            self.E_delta = np.zeros_like(params)
            
        self.E_grad = self.rho * self.E_grad + (1 - self.rho) * grads**2
        delta = np.sqrt(self.E_delta + self.epsilon) / np.sqrt(self.E_grad + self.epsilon) * grads
        self.E_delta = self.rho * self.E_delta + (1 - self.rho) * delta**2
        return params - self.lr * delta
        
class Adam(BaseOptimizer):
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon
        self.m, self.v, self.t = None, None, 0 # m: 1st moment, v: 2nd moment, t: timestep

    def update(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
            
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

class AdaMax(BaseOptimizer):
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon
        self.m, self.v, self.t = None, None, 0 # m: 1st moment, v: exponentially weighted infinity norm

    def update(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
            
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = np.maximum(self.beta2 * self.v, np.abs(grads))
        
        return params - (self.lr / (1 - self.beta1**self.t)) * self.m / (self.v + self.epsilon)

class Nadam(BaseOptimizer):
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon
        self.m, self.v, self.t = None, None, 0

    def update(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
            
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads**2
        
        m_hat = self.m / (1 - self.beta1**self.t) + (1 - self.beta1) * grads / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

class AMSGrad(Adam):
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate, beta1, beta2, epsilon)
        self.v_hat_max = None

    def update(self, params, grads):
        if self.m is None: self.m = np.zeros_like(params)
        if self.v is None: self.v = np.zeros_like(params)
        if self.v_hat_max is None: self.v_hat_max = np.zeros_like(params)
            
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads**2)
        
        v_hat = self.v / (1 - self.beta2**self.t)
        self.v_hat_max = np.maximum(self.v_hat_max, v_hat)
        
        # In the original AMSGrad paper, m_hat is not used, but it's common in implementations.
        # For consistency with the Adam base, we use bias correction for m.
        m_hat = self.m / (1 - self.beta1**self.t)
        
        return params - self.lr * m_hat / (np.sqrt(self.v_hat_max) + self.epsilon)

