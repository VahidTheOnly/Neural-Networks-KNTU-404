import numpy as np
import pandas as pd

from activations import logsig, tansig, swish, logsig_derivative, tansig_derivative, swish_derivative

def generate_and_save_data(n_samples=100):

    def F_original(x):
        return (3 * logsig(1.7 * x) +
                4 * tansig(3 * x) +
                4 * swish(2.5 * x) +
                x**4 - 2 * x**2 + 0.1 * x + 1)

    x_data = np.linspace(-2, 2, n_samples)
    y_data = F_original(x_data)

    df = pd.DataFrame({'x': x_data, 'y': y_data})
    try:
        df.to_excel("generated_data.xlsx", index=False)
        print("data saved successfully")
    except Exception as e:
        print(f"error in saving data: {e}")
        
    return x_data, y_data

def F_model(x, alphas, thetas):
    a1, a2, a3, a4, a5 = alphas
    t1, t2, t3 = thetas
    
    return (a1 * logsig(t1 * x) +
            a2 * tansig(t2 * x) +
            a3 * swish(t3 * x) +
            (a4 * x**2 - 1) +
            a5 * x)

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def compute_gradients(x, y_true, alphas, thetas):
    
    y_pred = F_model(x, alphas, thetas)
    error = y_pred - y_true
    n = len(y_true) 
    
    a1, a2, a3, _, _ = alphas
    t1, t2, t3 = thetas

    # ∂L/∂αi = (2/n) * Σ( (y_pred - y_true) * ∂y_pred/∂αi )
    grad_a1 = (2/n) * np.sum(error * logsig(t1 * x))
    grad_a2 = (2/n) * np.sum(error * tansig(t2 * x))
    grad_a3 = (2/n) * np.sum(error * swish(t3 * x))
    grad_a4 = (2/n) * np.sum(error * x**2)
    grad_a5 = (2/n) * np.sum(error * x)
    
    grad_alphas = np.array([grad_a1, grad_a2, grad_a3, grad_a4, grad_a5])

    # ∂L/∂θi = (2/n) * Σ( (y_pred - y_true) * ∂y_pred/∂θi )
    # ∂y_pred/∂θ1 = α1 * x * logsig'(θ1*x)
    grad_t1 = (2/n) * np.sum(error * a1 * x * logsig_derivative(t1 * x))
    # ∂y_pred/∂θ2 = α2 * x * tansig'(θ2*x)
    grad_t2 = (2/n) * np.sum(error * a2 * x * tansig_derivative(t2 * x))
    # ∂y_pred/∂θ3 = α3 * x * swish'(θ3*x)
    grad_t3 = (2/n) * np.sum(error * a3 * x * swish_derivative(t3 * x))

    grad_thetas = np.array([grad_t1, grad_t2, grad_t3])
    
    return grad_alphas, grad_thetas
