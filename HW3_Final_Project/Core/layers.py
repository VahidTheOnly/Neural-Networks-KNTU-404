import numpy as np
from Core.base import Layer
from Core.activations import Sigmoid  

class DenseLayer(Layer):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        limit = np.sqrt(6 / (input_dim + output_dim))
        self.W = np.random.uniform(-limit, limit, (output_dim, input_dim))
        self.b = np.zeros((output_dim, 1))
        
        self.X = None

    def forward(self, x):
        self.X = x.reshape(-1, 1)
        return self.W @ self.X + self.b

    def backward(self, output_gradient, **kwargs):
        lr = kwargs.get('lr', 0.01)
        grad_W = output_gradient @ self.X.T
        grad_b = output_gradient
        grad_input = self.W.T @ output_gradient
        
        if self.trainable:
            self.W -= lr * grad_W
            self.b -= lr * grad_b
        return grad_input

    def get_params(self):
        return np.concatenate([self.W.ravel(), self.b.ravel()]).reshape(-1, 1)

    def set_params(self, params):
        w_size = self.W.size
        self.W = params[:w_size].reshape(self.W.shape)
        self.b = params[w_size:].reshape(self.b.shape)

    def compute_jacobian(self, chain_grad=None):
        if chain_grad is None:
            chain_grad = np.ones((1, self.output_dim))
        
        chain_grad = chain_grad.reshape(1, -1)
        
        j_w = chain_grad.T @ self.X.T
        
        j_b = chain_grad.flatten()
        
        return np.hstack([j_w.ravel(), j_b]).reshape(1, -1)


class RoughRBFLayer(Layer):
    
    def __init__(self, input_dim, num_kernels):
        super().__init__()
        self.input_dim = input_dim
        self.num_kernels = num_kernels
        
        self.c_L = np.random.uniform(-1, 1, (num_kernels, input_dim))
        self.c_U = self.c_L + np.random.uniform(0.01, 0.2, (num_kernels, input_dim))
        
        self.s_L = np.random.uniform(0.5, 1.5, (num_kernels, 1))
        self.s_U = self.s_L + np.random.uniform(0.01, 0.2, (num_kernels, 1))
        
        self.X = None

    def init_centers(self, centers):
        self.c_L = centers
        self.c_U = centers + 0.05 
        self.s_L = np.ones_like(self.s_L)
        self.s_U = self.s_L + 0.1

    def forward(self, x):
        self.X = x.reshape(1, -1)
        
        self.dist_sq_L = np.sum((self.X - self.c_L)**2, axis=1).reshape(-1, 1)
        self.dist_sq_U = np.sum((self.X - self.c_U)**2, axis=1).reshape(-1, 1)
        
        self.phi_L = np.exp(-self.dist_sq_L / (2 * self.s_L**2))
        self.phi_U = np.exp(-self.dist_sq_U / (2 * self.s_U**2))
        
        return np.concatenate([self.phi_L, self.phi_U], axis=0)

    def backward(self, output_gradient, **kwargs):
        lr_c = kwargs.get('lr_c', 0.01)
        lr_s = kwargs.get('lr_s', 0.01)
        
        d_phi_L = output_gradient[:self.num_kernels]
        d_phi_U = output_gradient[self.num_kernels:]
        
        common_L = d_phi_L * self.phi_L / (self.s_L**2)
        grad_c_L = common_L * (self.X - self.c_L)
        grad_s_L = d_phi_L * self.phi_L * (self.dist_sq_L / (self.s_L**3))
        
        common_U = d_phi_U * self.phi_U / (self.s_U**2)
        grad_c_U = common_U * (self.X - self.c_U)
        grad_s_U = d_phi_U * self.phi_U * (self.dist_sq_U / (self.s_U**3))
        
        grad_input_L = np.sum(-common_L * (self.X - self.c_L), axis=0).reshape(-1, 1)
        grad_input_U = np.sum(-common_U * (self.X - self.c_U), axis=0).reshape(-1, 1)
        grad_input = grad_input_L + grad_input_U
        
        if self.trainable:
            self.c_L -= lr_c * grad_c_L
            self.c_U -= lr_c * grad_c_U
            self.s_L -= lr_s * grad_s_L
            self.s_U -= lr_s * grad_s_U
            
            self.c_U = np.maximum(self.c_L, self.c_U)
            self.s_L = np.maximum(self.s_L, 1e-4)
            self.s_U = np.maximum(self.s_L, self.s_U)

        return grad_input


class RoughDenseLayer(Layer):
    def __init__(self, input_dim, output_dim, alpha=0.5, beta=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.beta = beta
        
        self.W_L = np.random.uniform(-0.5, 0.5, (output_dim, input_dim))
        self.W_U = self.W_L + np.random.uniform(0.01, 0.2, (output_dim, input_dim))
        self.b_L = np.zeros((output_dim, 1))
        self.b_U = np.zeros((output_dim, 1))

    def forward(self, x):
        split_idx = x.shape[0] // 2
        self.phi_L = x[:split_idx]
        self.phi_U = x[split_idx:]
        
        self.net_L = self.W_L @ self.phi_L + self.b_L
        self.net_U = self.W_U @ self.phi_U + self.b_U
        
        return self.alpha * self.net_L + self.beta * self.net_U

    def backward(self, output_gradient, **kwargs):
        lr = kwargs.get('lr', 0.01)
        
        d_net_L = output_gradient * self.alpha
        d_net_U = output_gradient * self.beta
        
        grad_W_L = d_net_L @ self.phi_L.T
        grad_W_U = d_net_U @ self.phi_U.T
        
        grad_phi_L = self.W_L.T @ d_net_L
        grad_phi_U = self.W_U.T @ d_net_U
        grad_input = np.concatenate([grad_phi_L, grad_phi_U], axis=0)
        
        if self.trainable:
            self.W_L -= lr * grad_W_L
            self.W_U -= lr * grad_W_U
            self.b_L -= lr * d_net_L
            self.b_U -= lr * d_net_U
            self.W_U = np.maximum(self.W_L, self.W_U)
            
        return grad_input


class RoughHybridRecurrentLayer(Layer):
    def __init__(self, input_dim, hidden_dim, output_dim, memory_depth=1, alpha=0.5, beta=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.memory_depth = memory_depth
        self.alpha = alpha
        self.beta = beta
        
        self.activation_fn = Sigmoid()

        self.Wx_L = np.random.uniform(-0.5, 0.5, (hidden_dim, input_dim))
        self.Wx_U = self.Wx_L + 0.1
        self.Wy_L = np.random.uniform(-0.5, 0.5, (output_dim, hidden_dim))
        self.Wy_U = self.Wy_L + 0.1
        
        self.Wc1_L = [np.random.uniform(-0.1, 0.1, (hidden_dim, hidden_dim)) for _ in range(memory_depth)]
        self.Wc1_U = [w + 0.05 for w in self.Wc1_L]
        self.Wc2_L = [np.random.uniform(-0.1, 0.1, (hidden_dim, output_dim)) for _ in range(memory_depth)]
        self.Wc2_U = [w + 0.05 for w in self.Wc2_L]
        
        self.bh_L = np.zeros((hidden_dim, 1))
        self.bh_U = np.zeros((hidden_dim, 1))
        self.by_L = np.zeros((output_dim, 1))
        self.by_U = np.zeros((output_dim, 1))
        
        self.reset_state()

    def reset_state(self):
        self.h_history_L = [np.zeros((self.hidden_dim, 1)) for _ in range(self.memory_depth)]
        self.h_history_U = [np.zeros((self.hidden_dim, 1)) for _ in range(self.memory_depth)]
        self.y_history_L = [np.zeros((self.output_dim, 1)) for _ in range(self.memory_depth)]
        self.y_history_U = [np.zeros((self.output_dim, 1)) for _ in range(self.memory_depth)]

    def forward(self, x):
        self.X = x.reshape(-1, 1)
        self.cached_h_hist_L = [h.copy() for h in self.h_history_L]
        self.cached_h_hist_U = [h.copy() for h in self.h_history_U]
        self.cached_y_hist_L = [y.copy() for y in self.y_history_L]
        self.cached_y_hist_U = [y.copy() for y in self.y_history_U]
        
        net_h_L = self.Wx_L @ self.X + self.bh_L
        net_h_U = self.Wx_U @ self.X + self.bh_U
        
        for k in range(self.memory_depth):
            net_h_L += self.Wc1_L[k] @ self.h_history_L[k] + self.Wc2_L[k] @ self.y_history_L[k]
            net_h_U += self.Wc1_U[k] @ self.h_history_U[k] + self.Wc2_U[k] @ self.y_history_U[k]
            
        self.h_L = self.activation_fn.forward(net_h_L)
        self.h_U = self.activation_fn.forward(net_h_U)
        
        net_y_L = self.Wy_L @ self.h_L + self.by_L
        net_y_U = self.Wy_U @ self.h_U + self.by_U
        
        self.y_L = self.activation_fn.forward(net_y_L)
        self.y_U = self.activation_fn.forward(net_y_U)
        
        self.h_history_L = [self.h_L] + self.h_history_L[:-1]
        self.h_history_U = [self.h_U] + self.h_history_U[:-1]
        self.y_history_L = [self.y_L] + self.y_history_L[:-1]
        self.y_history_U = [self.y_U] + self.y_history_U[:-1]
        
        return self.alpha * self.y_L + self.beta * self.y_U

    def backward(self, output_gradient, **kwargs):
        lr = kwargs.get('lr', 0.01)
        dy_L = output_gradient * self.alpha
        dy_U = output_gradient * self.beta
        
        d_net_y_L = self.activation_fn.backward(dy_L) 
        
        d_net_y_L = dy_L * (self.y_L * (1 - self.y_L))
        d_net_y_U = dy_U * (self.y_U * (1 - self.y_U))
        
        grad_Wy_L = d_net_y_L @ self.h_L.T
        grad_Wy_U = d_net_y_U @ self.h_U.T
        grad_by_L = d_net_y_L
        grad_by_U = d_net_y_U
        
        dh_L = self.Wy_L.T @ d_net_y_L
        dh_U = self.Wy_U.T @ d_net_y_U
        
        d_net_h_L = dh_L * (self.h_L * (1 - self.h_L))
        d_net_h_U = dh_U * (self.h_U * (1 - self.h_U))
        
        grad_Wx_L = d_net_h_L @ self.X.T
        grad_Wx_U = d_net_h_U @ self.X.T
        
        grad_Wc1_L, grad_Wc1_U = [], []
        grad_Wc2_L, grad_Wc2_U = [], []
        
        for k in range(self.memory_depth):
            grad_Wc1_L.append(d_net_h_L @ self.cached_h_hist_L[k].T)
            grad_Wc1_U.append(d_net_h_U @ self.cached_h_hist_U[k].T)
            grad_Wc2_L.append(d_net_h_L @ self.cached_y_hist_L[k].T)
            grad_Wc2_U.append(d_net_h_U @ self.cached_y_hist_U[k].T)
            
        if self.trainable:
            self.Wx_L -= lr * grad_Wx_L
            self.Wx_U -= lr * grad_Wx_U
            self.Wy_L -= lr * grad_Wy_L
            self.Wy_U -= lr * grad_Wy_U
            self.bh_L -= lr * d_net_h_L 
            self.bh_U -= lr * d_net_h_U
            self.by_L -= lr * d_net_y_L
            self.by_U -= lr * d_net_y_U
            
            for k in range(self.memory_depth):
                self.Wc1_L[k] -= lr * grad_Wc1_L[k]
                self.Wc1_U[k] -= lr * grad_Wc1_U[k]
                self.Wc2_L[k] -= lr * grad_Wc2_L[k]
                self.Wc2_U[k] -= lr * grad_Wc2_U[k]
                
            self.Wx_U = np.maximum(self.Wx_L, self.Wx_U)
            self.Wy_U = np.maximum(self.Wy_L, self.Wy_U)
            
        grad_input = self.Wx_L.T @ d_net_h_L + self.Wx_U.T @ d_net_h_U
        return grad_input


class SimpleElmanLayer(Layer):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.Wx = np.random.uniform(-0.5, 0.5, (hidden_dim, input_dim))
        self.Wy = np.random.uniform(-0.5, 0.5, (output_dim, hidden_dim))
        self.Wc = np.random.uniform(-0.1, 0.1, (hidden_dim, hidden_dim)) # Context weight
        
        self.bh = np.zeros((hidden_dim, 1))
        self.by = np.zeros((output_dim, 1))
        
        self.reset_state()

    def reset_state(self):
        self.h_prev = np.zeros((self.hidden_dim, 1))

    def forward(self, x):
        self.X = x.reshape(-1, 1)
        self.h_prev_cached = self.h_prev.copy()
        
        net_h = self.Wx @ self.X + self.Wc @ self.h_prev + self.bh
        self.h = 1 / (1 + np.exp(-net_h)) 
        
        net_y = self.Wy @ self.h + self.by
        self.y = 1 / (1 + np.exp(-net_y))
        
        self.h_prev = self.h
        return self.y

    def backward(self, output_gradient, **kwargs):
        lr = kwargs.get('lr', 0.01)
        
        d_net_y = output_gradient * (self.y * (1 - self.y))
        grad_Wy = d_net_y @ self.h.T
        grad_by = d_net_y
        
        dh = self.Wy.T @ d_net_y
        d_net_h = dh * (self.h * (1 - self.h))
        
        grad_Wx = d_net_h @ self.X.T
        grad_bh = d_net_h
        grad_Wc = d_net_h @ self.h_prev_cached.T
        
        if self.trainable:
            self.Wx -= lr * grad_Wx
            self.Wy -= lr * grad_Wy
            self.Wc -= lr * grad_Wc
            self.bh -= lr * grad_bh
            self.by -= lr * grad_by
            
        return self.Wx.T @ d_net_h
    

class RoughIntervalOutputLayer(DenseLayer):
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim)
        self.W_L = np.random.uniform(-0.5, 0.5, (output_dim, input_dim))
        self.W_U = self.W_L + np.random.uniform(0.01, 0.2, (output_dim, input_dim))
        self.b_L = np.zeros((output_dim, 1))
        self.b_U = np.zeros((output_dim, 1))
        
    def forward(self, x):
        self.X = x.reshape(-1, 1)
        self.net_L = self.W_L @ self.X + self.b_L
        self.net_U = self.W_U @ self.X + self.b_U
        return np.concatenate([self.net_L, self.net_U], axis=0)

    def backward(self, output_gradient, **kwargs):
        lr = kwargs.get('lr', 0.01)
        split = output_gradient.shape[0] // 2
        d_L = output_gradient[:split]
        d_U = output_gradient[split:]
        
        grad_W_L = d_L @ self.X.T
        grad_W_U = d_U @ self.X.T
        grad_input = self.W_L.T @ d_L + self.W_U.T @ d_U
        
        if self.trainable:
            self.W_L -= lr * grad_W_L
            self.W_U -= lr * grad_W_U
            self.b_L -= lr * d_L
            self.b_U -= lr * d_U
            self.W_U = np.maximum(self.W_L, self.W_U)
            self.b_U = np.maximum(self.b_L, self.b_U)
        return grad_input


class RoughGMDHLayer(Layer):
    def __init__(self, input_dim, alpha=0.5, beta=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.alpha = alpha
        self.beta = beta
        self.num_pairs = int(input_dim * (input_dim - 1) / 2)
        
        self.W_L = np.random.uniform(-0.1, 0.1, (self.num_pairs, 6))
        self.W_U = self.W_L + 0.05
        
        self.pair_indices = []
        for i in range(input_dim):
            for j in range(i + 1, input_dim):
                self.pair_indices.append((i, j))

    def _compute_poly_features(self, x):
        x = x.flatten()
        feats = np.zeros((self.num_pairs, 6))
        for k, (i, j) in enumerate(self.pair_indices):
            xi, xj = x[i], x[j]
            feats[k, :] = [1.0, xi, xj, xi**2, xj**2, xi*xj]
        return feats

    def forward(self, x):
        self.X_in = x
        self.poly_feats = self._compute_poly_features(x)
        
        self.y_L = np.sum(self.W_L * self.poly_feats, axis=1).reshape(-1, 1)
        self.y_U = np.sum(self.W_U * self.poly_feats, axis=1).reshape(-1, 1)
        
        return self.alpha * self.y_L + self.beta * self.y_U

    def backward(self, output_gradient, **kwargs):
        lr = kwargs.get('lr', 0.01)
        d_y_L = output_gradient * self.alpha
        d_y_U = output_gradient * self.beta
        
        grad_W_L = d_y_L * self.poly_feats
        grad_W_U = d_y_U * self.poly_feats
        
        grad_input = np.zeros_like(self.X_in)
        x_flat = self.X_in.flatten()
        
        for k, (i, j) in enumerate(self.pair_indices):
            xi, xj = x_flat[i], x_flat[j]
            
            dpoly_dxi_L = self.W_L[k, 1] + 2*self.W_L[k, 3]*xi + self.W_L[k, 5]*xj
            dpoly_dxj_L = self.W_L[k, 2] + 2*self.W_L[k, 4]*xj + self.W_L[k, 5]*xi
            
            dpoly_dxi_U = self.W_U[k, 1] + 2*self.W_U[k, 3]*xi + self.W_U[k, 5]*xj
            dpoly_dxj_U = self.W_U[k, 2] + 2*self.W_U[k, 4]*xj + self.W_U[k, 5]*xi
            
            grad_input[i] += d_y_L[k] * dpoly_dxi_L + d_y_U[k] * dpoly_dxi_U
            grad_input[j] += d_y_L[k] * dpoly_dxj_L + d_y_U[k] * dpoly_dxj_U

        if self.trainable:
            self.W_L -= lr * grad_W_L
            self.W_U -= lr * grad_W_U
            self.W_U = np.maximum(self.W_L, self.W_U)
            
        return grad_input
