import numpy as np
from Core.base import Optimizer

class SGD(Optimizer):
    """
    Stochastic Gradient Descent Manager.
    
    In our architecture, the layers perform the weight updates internally 
    during the backward() call. This class serves as a scheduler and 
    hyperparameter container to pass the correct learning rates to the model.
    """
    def __init__(self, lr=0.01, lr_centers=None, lr_sigmas=None, lr_trig=None):
        """
        Args:
            lr: Base learning rate for weights (Dense, Recurrent).
            lr_centers: Learning rate specifically for RBF centers (Question 1).
            lr_sigmas: Learning rate specifically for RBF widths (Question 1).
            lr_trig: Learning rate for Trigonometric alpha/theta (Question 3).
        """
        self.lr = lr
        self.lr_centers = lr_centers if lr_centers is not None else lr
        self.lr_sigmas = lr_sigmas if lr_sigmas is not None else lr
        self.lr_trig = lr_trig if lr_trig is not None else lr

    def get_learning_rates(self):
        """Returns a dictionary to be unpacked into layer.backward(**kwargs)"""
        return {
            'lr': self.lr,
            'lr_c': self.lr_centers,
            'lr_s': self.lr_sigmas,
            'lr_trig': self.lr_trig
        }

    def update(self, model, x, y):
        """
        SGD logic is handled within the model's training loop (forward -> loss -> backward).
        This method is a placeholder if we wanted to decouple updates.
        """
        pass


class CompetitiveLearning(Optimizer):
    """
    Unsupervised Competitive Learning (Winner-Take-All) for RBF Centers.
    Fixed: Uses index shuffling to avoid modifying the original data in-place.
    """
    def __init__(self, num_centers, input_dim, learning_rate=0.05):
        self.num_centers = num_centers
        self.input_dim = input_dim
        self.lr = learning_rate
        self.centers = np.random.uniform(-1.5, 1.5, (num_centers, input_dim))

    def fit(self, data, epochs=50):
        """
        Args:
            data: Numpy array of shape (N, input_dim)
        """
        print(f"Starting Competitive Learning for {epochs} epochs...")
        n_samples = len(data)
        indices = np.arange(n_samples) # Create indices [0, 1, 2, ...]

        for epoch in range(epochs):
            # Shuffle indices instead of data to keep original X intact
            np.random.shuffle(indices)
            
            total_shift = 0
            for i in indices:
                x = data[i].reshape(1, -1)
                
                # Compute distances
                distances = np.linalg.norm(x - self.centers, axis=1)
                
                # Winner
                winner_idx = np.argmin(distances)
                
                # Update
                shift = self.lr * (x.flatten() - self.centers[winner_idx])
                self.centers[winner_idx] += shift
                total_shift += np.linalg.norm(shift)
            
            self.lr *= 0.99
            
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Total Shift: {total_shift:.4f}")
        
        return self.centers

    def update(self, model, x, y):
        pass


class EKFOptimizer(Optimizer):
    """
    Extended Kalman Filter (EKF) for Neural Network Training.
    Required for Question 3.
    
    Mathematical Foundation:
    Treats the weights (theta) as the state of a dynamic system.
    State Equation: theta(k+1) = theta(k) + q(k)  (Random walk assumption)
    Observation:    y(k) = h(theta(k), x(k)) + r(k)
    
    Update Equations:
    1. Error: e = d - y_pred
    2. Jacobian H: dy_pred / d_theta
    3. Kalman Gain K: P * H^T * inv(H * P * H^T + R)
    4. Update State: theta_new = theta + K * e
    5. Update Covariance P: P_new = P - K * H * P + Q
    """
    
    def __init__(self, layers, P_init=100.0, Q_val=1e-4, R_val=0.1):
        """
        Args:
            layers: List of layers in the model (must implement get_params/compute_jacobian).
            P_init: Initial value for diagonal covariance matrix (High uncertainty).
            Q_val: Process noise covariance (allows weights to keep adapting).
            R_val: Measurement noise covariance (trust in target data).
        """
        self.layers = layers
        self.R = np.array([[R_val]])
        self.Q_val = Q_val
        
        # 1. Aggregate all parameters into a global state vector
        self.total_params = sum(layer.get_params().size for layer in layers)
        
        if self.total_params == 0:
            raise ValueError("No trainable parameters found for EKF.")

        # 2. Initialize Covariance Matrix P (Diagonal)
        self.P = np.eye(self.total_params) * P_init
        
        # 3. Initialize Process Noise Q (Diagonal)
        self.Q = np.eye(self.total_params) * Q_val
        
        print(f"EKF Initialized. Total Parameters: {self.total_params}")

    def _get_global_theta(self):
        """Retrieves all parameters from layers as a single vector."""
        params_list = [layer.get_params().flatten() for layer in self.layers]
        # Filter out empty arrays (layers with no params)
        params_list = [p for p in params_list if p.size > 0]
        return np.concatenate(params_list).reshape(-1, 1)

    def _set_global_theta(self, theta):
        """Distributes the updated state vector back to layers."""
        start_idx = 0
        for layer in self.layers:
            n_params = layer.get_params().size
            if n_params > 0:
                layer_params = theta[start_idx : start_idx + n_params]
                layer.set_params(layer_params)
                start_idx += n_params

    def update(self, model, x, y):
        """
        Performs one EKF update step.
        Args:
            model: The model object (used to run forward pass).
            x: Input sample (single time step).
            y: Target value (single time step).
        """
        # --- 1. Forward Pass ---
        # We need the current activations to compute Jacobians
        y_pred = model.forward(x)
        
        # Error (Innovation)
        error = y - y_pred # Shape (1, 1) assuming scalar output
        
        # --- 2. Compute Global Jacobian H ---
        # H = [dy/d_theta1, dy/d_theta2, ...]
        # We need to traverse layers backwards to compute gradients via Chain Rule
        
        H_parts = []
        
        # Initial chain gradient (dy/dy = 1)
        chain_grad = np.ones((1, 1))
        
        # Iterate backwards through layers
        for layer in reversed(self.layers):
            if layer.get_params().size > 0:
                # Compute Jacobian for this layer's parameters given the chain gradient
                # Shape: (1, n_layer_params)
                h_layer = layer.compute_jacobian(chain_grad)
                H_parts.append(h_layer)
            
            # Update chain gradient for the next layer (moving backwards)
            # We need dy/dx_layer = dy/dy_layer * dy_layer/dx_layer
            # This requires a helper method in layers or manual calculation.
            # For Dense and Trig, we can implement/use 'get_input_derivative_for_ekf' logic.
            
            if hasattr(layer, 'get_input_derivative_for_ekf'):
                # Trig layer has this
                local_grad = layer.get_input_derivative_for_ekf()
                chain_grad = chain_grad * local_grad
            elif hasattr(layer, 'W'): 
                # Dense layer: derivative is W.T
                # chain_grad (1, out) @ W (out, in) -> (1, in)
                # Note: chain_grad is (1, out_dim)
                chain_grad = chain_grad @ layer.W
            else:
                # If layer doesn't propagate gradient (e.g. unknown), stop or assume identity
                pass

        # Since we collected parts backwards, reverse them to match theta order
        H = np.hstack(H_parts[::-1]) 
        
        # --- 3. Kalman Update ---
        
        # S = H * P * H.T + R
        # Use simple scalar division if output is 1D for speed
        S = H @ self.P @ H.T + self.R
        
        # K = P * H.T * inv(S)
        # Using pseudoinverse for stability
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
            
        K = self.P @ H.T @ S_inv
        
        # Update State: theta = theta + K * error
        theta_current = self._get_global_theta()
        theta_new = theta_current + K @ error
        self._set_global_theta(theta_new)
        
        # Update Covariance: P = P - K * H * P + Q
        # Joseph form is more numerically stable but this standard form is usually fine
        # P = (I - K H) P
        I = np.eye(self.total_params)
        self.P = (I - K @ H) @ self.P + self.Q
        
        return error.item()