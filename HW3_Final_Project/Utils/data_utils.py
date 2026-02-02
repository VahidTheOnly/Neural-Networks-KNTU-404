import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DataHandler:
    """
    Utility class for data loading, preprocessing, and generation.
    Handles requirements for all HW3 questions.
    """

    @staticmethod
    def load_data(filepath, header=None):
        """
        Loads data from Excel files.
        Args:
            filepath: Path to .xlsx file.
            header: Row number to use as header (None for no header).
        Returns:
            np.array: Flattened numpy array of data.
        """
        try:
            df = pd.read_excel(filepath, header=header)
            # Assuming data is in the first column or is a single series
            data = df.values.flatten()
            return data.astype(np.float64)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    @staticmethod
    def normalize_data(data):
        """
        Min-Max Normalization to [0, 1].
        Returns:
            normalized_data: scaled data.
            scaler_params: (min_val, max_val) for denormalization.
        """
        min_val = np.min(data)
        max_val = np.max(data)
        # Avoid division by zero
        if max_val - min_val == 0:
            return data, (min_val, max_val)
            
        normalized = (data - min_val) / (max_val - min_val)
        return normalized, (min_val, max_val)

    @staticmethod
    def denormalize_data(data, scaler_params):
        """Reverts data to original scale."""
        min_val, max_val = scaler_params
        return data * (max_val - min_val) + min_val

    @staticmethod
    def create_sequences(data, input_steps=5, prediction_horizon=3):
        """
        Creates time-series windows (Sliding Window).
        
        Args:
            data: 1D numpy array.
            input_steps: Number of past steps to use as input (default 5).
            prediction_horizon: Which future step to predict (default 3).
                                1 means predict t+1, 3 means predict t+3.
        
        Returns:
            X: Input matrix (N, input_steps)
            y: Target vector (N, 1)
        """
        X, y = [], []
        # We need enough data for input + horizon
        # Indices: [0, 1, 2, 3, 4] -> predict index [4 + 3] = 7
        total_window = input_steps + prediction_horizon
        
        for i in range(len(data) - total_window + 1):
            window = data[i : i + input_steps]
            # Target is the step at 'prediction_horizon' steps after the window
            target_idx = i + input_steps + prediction_horizon - 1
            target = data[target_idx]
            
            X.append(window)
            y.append(target)
            
        return np.array(X), np.array(y).reshape(-1, 1)

    @staticmethod
    def train_test_split(X, y, train_ratio=0.7, shuffle=False):
        """
        Splits data into training and testing sets.
        For Time Series (Q1, Q2, Q4), shuffle MUST be False.
        For Classification (Q1-Part B), shuffle CAN be True.
        """
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
        """
        Adds Additive Gaussian Noise (Q3 - Part C).
        
        Args:
            data: Input signal.
            noise_level_percent: Percentage of signal amplitude (e.g., 0.1, 0.5, 1.0).
        """
        amplitude = np.max(data) - np.min(data)
        std_dev = amplitude * noise_level_percent
        noise = np.random.normal(0, std_dev, data.shape)
        return data + noise

    # --- Q1 Part B: Synthetic Classification Data ---
    
    @staticmethod
    def generate_classification_dataset():
        """
        Generates the Red/Blue class dataset specified in Question 1 Part B.
        Returns:
            X: (N, 2) coordinates
            y: (N, 1) labels (0 for Red, 1 for Blue)
        """
        np.random.seed(42)

        def generate_class(centers, n_per_center, std):
            X_local = []
            for c in centers:
                X_local.append(np.random.randn(n_per_center, 2) * std + c)
            return np.vstack(X_local)

        # Red Class Config
        red_centers = [(-0.8, 0.6), (-0.4, -0.3), (-0.5, -1.2), 
                       (0.4, 0.7), (0.6, -0.2), (1.0, -1.0)]
        X_red = generate_class(centers=red_centers, n_per_center=100, std=0.25)
        y_red = np.zeros((X_red.shape[0], 1)) # Label 0

        # Blue Class Config
        blue_centers = [(-1.5, 1.1), (-0.7, 1.4), (0.2, 1.3), (1.0, 1.2), 
                        (1.6, 0.7), (1.5, -1.2), (0.3, -1.0), (-0.8, -0.9), (-1.5, -0.2)]
        X_blue = generate_class(centers=blue_centers, n_per_center=100, std=0.22)
        y_blue = np.ones((X_blue.shape[0], 1)) # Label 1

        # Combine
        X = np.vstack([X_red, X_blue])
        y = np.vstack([y_red, y_blue])
        
        return X, y

    # --- Q3 Part E: Mackey-Glass Differential Equation ---

    @staticmethod
    def generate_mackey_glass(n_samples=1000, tau=17, delta_t=0.1):
        """
        Generates chaotic time series using Mackey-Glass equation.
        dx/dt = beta * x(t-tau) / (1 + x(t-tau)^n) - gamma * x(t)
        
        Standard params: beta=0.2, gamma=0.1, n=10, tau=17
        """
        beta = 0.2
        gamma = 0.1
        n = 10
        
        # Initialization history (needs to be at least tau long)
        history_len = int(tau / delta_t) + 1
        x = [1.2] * history_len # Initial condition
        
        generated_data = []
        
        for i in range(n_samples):
            current_x = x[-1]
            delayed_x = x[-history_len] # x(t - tau)
            
            # Euler Integration step
            dx_dt = (beta * delayed_x) / (1 + delayed_x**n) - (gamma * current_x)
            next_x = current_x + dx_dt * delta_t
            
            x.append(next_x)
            generated_data.append(next_x)
            
            # Keep buffer size manageable? 
            # Actually we just need to append and index backwards. 
            # For memory efficiency in huge loops we might pop, but here it's fine.
            
        return np.array(generated_data)