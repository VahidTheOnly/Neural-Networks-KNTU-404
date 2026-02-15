import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DataHandler:

    @staticmethod
    def load_data(filepath, header=None):
        try:
            df = pd.read_excel(filepath, header=header)
            data = df.values.flatten()
            return data.astype(np.float64)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    @staticmethod
    def normalize_data(data):
        
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val - min_val == 0:
            return data, (min_val, max_val)
            
        normalized = (data - min_val) / (max_val - min_val)
        return normalized, (min_val, max_val)

    @staticmethod
    def denormalize_data(data, scaler_params):
        min_val, max_val = scaler_params
        return data * (max_val - min_val) + min_val

    @staticmethod
    def create_sequences(data, input_steps=5, prediction_horizon=3):
        X, y = [], []
        total_window = input_steps + prediction_horizon
        
        for i in range(len(data) - total_window + 1):
            window = data[i : i + input_steps]
            target_idx = i + input_steps + prediction_horizon - 1
            target = data[target_idx]
            
            X.append(window)
            y.append(target)
            
        return np.array(X), np.array(y).reshape(-1, 1)

    @staticmethod
    def train_test_split(X, y, train_ratio=0.7, shuffle=False):
        num_samples = len(X)
        indices = np.arange(num_samples)
        
        if shuffle:
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
            
        train_size = int(num_samples * train_ratio)
        
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return X_train, X_test, y_train, y_test

    @staticmethod
    def add_noise(data, noise_level_percent):
        amplitude = np.max(data) - np.min(data)
        std_dev = amplitude * noise_level_percent
        noise = np.random.normal(0, std_dev, data.shape)
        return data + noise

    
    @staticmethod
    def generate_classification_dataset():
        np.random.seed(42)

        def generate_class(centers, n_per_center, std):
            X_local = []
            for c in centers:
                X_local.append(np.random.randn(n_per_center, 2) * std + c)
            return np.vstack(X_local)

        red_centers = [(-0.8, 0.6), (-0.4, -0.3), (-0.5, -1.2), 
                       (0.4, 0.7), (0.6, -0.2), (1.0, -1.0)]
        X_red = generate_class(centers=red_centers, n_per_center=100, std=0.25)
        y_red = np.zeros((X_red.shape[0], 1)) 

        blue_centers = [(-1.5, 1.1), (-0.7, 1.4), (0.2, 1.3), (1.0, 1.2), 
                        (1.6, 0.7), (1.5, -1.2), (0.3, -1.0), (-0.8, -0.9), (-1.5, -0.2)]
        X_blue = generate_class(centers=blue_centers, n_per_center=100, std=0.22)
        y_blue = np.ones((X_blue.shape[0], 1)) 

        X = np.vstack([X_red, X_blue])
        y = np.vstack([y_red, y_blue])
        
        return X, y


    @staticmethod
    def generate_mackey_glass(n_steps, tau, x0, delta_t=0.1, beta=0.2, gamma=0.1, n=10):
        history = int(tau / delta_t)
        x = np.full(n_steps + history, x0)
        
        for i in range(history, n_steps + history - 1):
            x_tau = x[i - history]
            dxdt = (beta * x_tau) / (1 + x_tau**n) - gamma * x[i]
            x[i+1] = x[i] + dxdt * delta_t
            
        return x[history:]
