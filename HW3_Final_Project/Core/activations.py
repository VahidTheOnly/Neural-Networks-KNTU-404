import numpy as np
from Core.base import Layer

class Sigmoid(Layer):
    """
    Standard Sigmoid Activation Function.
    Formula: f(x) = 1 / (1 + exp(-x))
    Range: (0, 1)
    """
    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, output_gradient, **kwargs):
        # Derivative: f(x) * (1 - f(x))
        # Element-wise multiplication with incoming gradient
        return output_gradient * (self.out * (1 - self.out))


class Tanh(Layer):
    """
    Hyperbolic Tangent Activation.
    Formula: f(x) = tanh(x)
    Range: (-1, 1)
    """
    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, output_gradient, **kwargs):
        # Derivative: 1 - tanh^2(x)
        return output_gradient * (1 - self.out ** 2)


class ReLU(Layer):
    """
    Rectified Linear Unit.
    Formula: f(x) = max(0, x)
    Range: [0, inf)
    """
    def __init__(self):
        super().__init__()
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, output_gradient, **kwargs):
        # Derivative: 1 if x > 0 else 0
        return output_gradient * (self.x > 0)


class Linear(Layer):
    """
    Linear (Identity) Activation.
    Used typically for the output layer in regression problems.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

    def backward(self, output_gradient, **kwargs):
        # Derivative is 1
        return output_gradient


class TrigonometricActivation(Layer):
    """
    Adaptive Trigonometric Activation Function (Required for Q3).
    Formula: f(x) = sin(alpha * x + theta)
    
    Parameters:
    - alpha: Frequency/Scaling factor (Trainable)
    - theta: Phase shift (Trainable)
    
    This layer supports both SGD (Backpropagation) and EKF training.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        
        # Initialize trainable parameters
        # We use a vector of alphas and thetas (one per neuron) for maximum flexibility
        self.alpha = np.random.uniform(0.5, 1.5, (input_dim, 1))
        self.theta = np.random.uniform(-0.1, 0.1, (input_dim, 1))
        
        # Cache
        self.x = None
        self.net_trig = None # stores (alpha * x + theta)

    def forward(self, x):
        self.x = x.reshape(-1, 1)
        self.net_trig = self.alpha * self.x + self.theta
        return np.sin(self.net_trig)

    def backward(self, output_gradient, **kwargs):
        """
        Standard Backpropagation for Trigonometric parameters.
        """
        lr = kwargs.get('lr_trig', kwargs.get('lr', 0.01))
        
        # Common derivative term: cos(alpha * x + theta)
        cos_term = np.cos(self.net_trig)
        
        # 1. Gradient w.r.t Input X (to pass to previous layer)
        # dy/dx = alpha * cos(...)
        # element-wise chain rule
        grad_input = output_gradient * self.alpha * cos_term
        
        # 2. Gradient w.r.t Parameters (alpha, theta)
        # dy/d_alpha = x * cos(...)
        grad_alpha = output_gradient * self.x * cos_term
        
        # dy/d_theta = 1 * cos(...)
        grad_theta = output_gradient * cos_term
        
        # Update Parameters (Gradient Descent)
        if self.trainable:
            self.alpha -= lr * grad_alpha
            self.theta -= lr * grad_theta
            
        return grad_input

    # --- EKF Specific Implementation (For Q3) ---

    def get_params(self):
        """
        Returns concatenated [alpha; theta] vector.
        """
        return np.concatenate([self.alpha.flatten(), self.theta.flatten()]).reshape(-1, 1)

    def set_params(self, params):
        """
        Updates alpha and theta from EKF state vector.
        """
        size = self.alpha.size
        self.alpha = params[:size].reshape(self.alpha.shape)
        self.theta = params[size:].reshape(self.theta.shape)

    def compute_jacobian(self, chain_grad=None):
        """
        Computes Jacobian matrix of output w.r.t parameters [alpha, theta].
        
        For a single time-step prediction (scalar output or independent vector outputs):
        The Jacobian H is a row vector (1, n_params) if output is scalar.
        
        H = [dy/d_alpha, dy/d_theta]
        """
        cos_term = np.cos(self.net_trig) # Shape (input_dim, 1)
        
        # Jacobian elements
        # If this is the output layer, chain_grad is 1. 
        # If not, chain_grad contains downstream gradients.
        grad_upstream = 1.0
        if chain_grad is not None:
            grad_upstream = chain_grad.item() if chain_grad.size == 1 else chain_grad

        # dy/d_alpha = x * cos(...)
        j_alpha = (self.x * cos_term) * grad_upstream
        
        # dy/d_theta = cos(...)
        j_theta = cos_term * grad_upstream
        
        # Flatten to match the get_params structure
        # We assume diagonal independence for vector outputs to keep H manageable, 
        # or simply flatten if it's a scalar output layer.
        return np.hstack([j_alpha.flatten(), j_theta.flatten()]).reshape(1, -1)
    
    def get_input_derivative_for_ekf(self):
        """
        Helper for EKF:
        Returns dy/dx to be propagated to the previous layer's Jacobian computation.
        dy/dx = alpha * cos(alpha*x + theta)
        """
        return self.alpha * np.cos(self.net_trig)