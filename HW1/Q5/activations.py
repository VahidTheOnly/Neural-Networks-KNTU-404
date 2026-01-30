import numpy as np

# (Activation Functions)

def logsig(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def tansig(x):
    return np.tanh(x)

def swish(x):
    return x * logsig(x)


# (Derivatives)

def logsig_derivative(x):
    """
    f'(x) = f(x) * (1 - f(x))
    """
    fx = logsig(x)
    return fx * (1 - fx)

def tansig_derivative(x):
    """
    f'(x) = 1 - f(x)^2
    """
    fx = tansig(x)
    return 1 - fx**2

def swish_derivative(x):
    """
    f(x) = x * g(x) => f'(x) = g(x) + x * g'(x)
    where g(x) is logsig(x).
    """
    sx = logsig(x)
    return sx + x * logsig_derivative(x)
