import numpy as np
import time
from tqdm import tqdm  # For progress bars
from Core.base import Layer
from Core.optimizers import EKFOptimizer, SGD

class Sequential:
    """
    Main Model Class.
    Manages the stack of layers, forward/backward passes, and the training loop.
    
    Capabilities:
    - Standard Backpropagation Training (SGD).
    - Extended Kalman Filter Training (EKF) for Q3.
    - Automatic state resetting for Recurrent Networks (Q2).
    - History tracking for visualization.
    """
    def __init__(self, layers=None):
        self.layers = layers if layers else []
        self.loss_fn = None
        self.optimizer = None
        self.history = {'loss': []}

    def add(self, layer):
        """Adds a layer to the stack."""
        if not isinstance(layer, Layer):
            raise TypeError("Object must inherit from core.base.Layer")
        self.layers.append(layer)

    def compile(self, loss, optimizer):
        """
        Configures the model for training.
        Args:
            loss: Instance of core.losses.Loss
            optimizer: Instance of core.optimizers.Optimizer (SGD or EKF)
        """
        self.loss_fn = loss
        self.optimizer = optimizer

    def reset_states(self):
        """Resets the memory of all recurrent layers in the model."""
        for layer in self.layers:
            layer.reset_state()

    def forward(self, x):
        """
        Passes input x through all layers sequentially.
        """
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, loss_grad, learning_rates):
        """
        Passes gradient backward through all layers.
        Used ONLY for SGD/Standard Backprop.
        """
        grad = loss_grad
        # Iterate backwards
        for layer in reversed(self.layers):
            # Pass dictionary of learning rates so the layer picks what it needs
            grad = layer.backward(grad, **learning_rates)
        return grad

    def fit(self, X, y, epochs=100, batch_size=1, verbose=True):
        """
        Main Training Loop.
        
        Handles two distinct training modes:
        1. Standard SGD: Forward -> Loss -> Backward (Update inside layers).
        2. EKF (Q3): Update via Optimizer -> Forward (Implicitly handled in EKF step).
        """
        if self.loss_fn is None or self.optimizer is None:
            raise RuntimeError("You must call .compile() before .fit()")

        # Determine Training Mode
        is_ekf = isinstance(self.optimizer, EKFOptimizer)
        mode_name = "EKF" if is_ekf else "SGD"
        
        n_samples = len(X)
        print(f"Starting Training | Mode: {mode_name} | Epochs: {epochs} | Samples: {n_samples}")
        
        start_time = time.time()

        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Reset Recurrent States at the start of each epoch (for Q2 stability)
            self.reset_states()
            
            # Create a progress bar for this epoch
            if verbose:
                pbar = tqdm(range(n_samples), desc=f"Epoch {epoch+1}/{epochs}", unit="sample")
            else:
                pbar = range(n_samples)

            # --- Sample Loop ---
            for i in pbar:
                xi = X[i]
                yi = y[i]

                # --- MODE 1: Extended Kalman Filter (EKF) ---
                if is_ekf:
                    # EKF Optimizer handles the entire update step (Forward & Update)
                    # It returns the error (innovation) for logging
                    current_error = self.optimizer.update(self, xi, yi)
                    
                    # Compute Loss for history (Squared Error proxy)
                    loss_val = 0.5 * (current_error ** 2)
                    epoch_loss += loss_val

                # --- MODE 2: Standard SGD / Backpropagation ---
                else:
                    # 1. Forward
                    y_pred = self.forward(xi)
                    
                    # 2. Compute Loss
                    loss_val = self.loss_fn.compute(y_pred, yi)
                    epoch_loss += loss_val
                    
                    # 3. Compute Gradient of Loss w.r.t Output
                    grad_loss = self.loss_fn.gradient(y_pred, yi)
                    
                    # 4. Backward & Update
                    # Retrieve learning rates from SGD optimizer
                    lrs = self.optimizer.get_learning_rates()
                    self.backward(grad_loss, lrs)

                # Update progress bar description
                if verbose and isinstance(pbar, tqdm):
                    pbar.set_postfix({'loss': f"{loss_val:.5f}"})

            # End of Epoch
            avg_loss = epoch_loss / n_samples
            self.history['loss'].append(avg_loss)
            
            if verbose:
                print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.6f}")

        total_time = time.time() - start_time
        print(f"Training Complete. Time: {total_time:.2f}s")
        return self.history

    def predict(self, X):
        """
        Generates predictions for the input dataset X.
        """
        self.reset_states()
        predictions = []
        for i in range(len(X)):
            pred = self.forward(X[i])
            predictions.append(pred)
        return np.array(predictions)

    def summary(self):
        """Prints a summary of the model architecture."""
        print("-" * 60)
        print(f"{'Layer (type)':<30} {'Param Shape / Details':<30}")
        print("=" * 60)
        
        total_params = 0
        for layer in self.layers:
            # Try to get output shape or param count
            try:
                params = layer.get_params().size
            except:
                params = "N/A"
            
            if params != "N/A":
                total_params += params
                
            print(f"{layer.name:<30} {f'Params: {params}':<30}")
            
        print("=" * 60)
        print(f"Total Trainable Parameters: {total_params}")
        print("-" * 60)