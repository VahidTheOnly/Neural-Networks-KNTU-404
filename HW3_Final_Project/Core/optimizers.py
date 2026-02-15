import numpy as np
from Core.base import Optimizer

class SGD(Optimizer):
    def __init__(self, lr=0.01, lr_centers=None, lr_sigmas=None, lr_trig=None):
        self.lr = lr
        self.lr_centers = lr_centers if lr_centers is not None else lr
        self.lr_sigmas = lr_sigmas if lr_sigmas is not None else lr
        self.lr_trig = lr_trig if lr_trig is not None else lr

    def get_learning_rates(self):
        return {
            'lr': self.lr,
            'lr_c': self.lr_centers,
            'lr_s': self.lr_sigmas,
            'lr_trig': self.lr_trig
        }

    def update(self, model, x, y):
        pass

class CompetitiveLearning(Optimizer):
    def __init__(self, num_centers, input_dim, learning_rate=0.05):
        self.num_centers = num_centers
        self.input_dim = input_dim
        self.lr = learning_rate
        self.centers = np.random.uniform(-1.5, 1.5, (num_centers, input_dim))

    def fit(self, data, epochs=50):
        print(f"Starting Competitive Learning for {epochs} epochs...")
        n_samples = len(data)
        indices = np.arange(n_samples)

        for epoch in range(epochs):
            np.random.shuffle(indices)
            total_shift = 0
            for i in indices:
                x = data[i].reshape(1, -1)
                distances = np.linalg.norm(x - self.centers, axis=1)
                winner_idx = np.argmin(distances)
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
    def __init__(self, layers, P_init=100.0, Q_val=1e-4, R_val=0.1):
        self.layers = layers
        self.R = np.array([[R_val]])
        self.Q_val = Q_val
        
        self.total_params = sum(layer.get_params().size for layer in layers)
        if self.total_params == 0:
            raise ValueError("No trainable parameters found for EKF.")

        self.P = np.eye(self.total_params) * P_init
        self.Q = np.eye(self.total_params) * Q_val
        
        print(f"EKF Initialized. Total Parameters: {self.total_params}")

    def _get_global_theta(self):
        params_list = [layer.get_params().flatten() for layer in self.layers]
        params_list = [p for p in params_list if p.size > 0]
        return np.concatenate(params_list).reshape(-1, 1)

    def _set_global_theta(self, theta):
        start_idx = 0
        for layer in self.layers:
            n_params = layer.get_params().size
            if n_params > 0:
                layer_params = theta[start_idx : start_idx + n_params]
                layer.set_params(layer_params)
                start_idx += n_params

    def update(self, model, x, y):
        y_pred = model.forward(x)
        error = y - y_pred 
        
        H_parts = []
        
        chain_grad = np.ones((1, 1))
        
        for layer in reversed(self.layers):
            if layer.get_params().size > 0:
                h_layer = layer.compute_jacobian(chain_grad)
                H_parts.append(h_layer)
            
            if hasattr(layer, 'get_input_derivative_for_ekf'):
                local_grad = layer.get_input_derivative_for_ekf()
                chain_grad = chain_grad * local_grad 
            elif hasattr(layer, 'W'):
                chain_grad = chain_grad @ layer.W
            
        H = np.hstack(H_parts[::-1]) 
        
        # 3. Kalman Update
        S = H @ self.P @ H.T + self.R
        
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
            
        K = self.P @ H.T @ S_inv
        
        theta_current = self._get_global_theta()
        theta_new = theta_current + K @ error
        self._set_global_theta(theta_new)
        
        self.P = self.P - K @ H @ self.P + self.Q
        
        return error.item()