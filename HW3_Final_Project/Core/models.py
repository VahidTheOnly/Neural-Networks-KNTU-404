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
    - History tracking for visualization (Train & Validation).
    """
    def __init__(self, layers=None):
        self.layers = layers if layers else []
        self.loss_fn = None
        self.optimizer = None
        # Initialize history to track both training and validation loss
        self.history = {'loss': [], 'val_loss': []}

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

    def _compute_val_loss(self, X_val, y_val):
        """Helper to compute validation loss without training."""
        if X_val is None or y_val is None:
            return None
        
        # Reset states before validation to ensure clean history for recurrent nets
        self.reset_states()
        
        val_loss_sum = 0.0
        n_val = len(X_val)
        
        for i in range(n_val):
            # Forward pass only
            pred = self.forward(X_val[i])
            loss = self.loss_fn.compute(pred, y_val[i])
            val_loss_sum += loss
            
        return val_loss_sum / n_val

    def fit(self, X, y, validation_data=None, epochs=100, batch_size=1, verbose=True, log_freq=5):
        """
        Main Training Loop.
        
        Args:
            X, y: Training data.
            validation_data: Tuple (X_val, y_val) or None.
            epochs: Number of training iterations.
            verbose: Print logs.
            log_freq: Frequency of printing logs (e.g., 5 means print every 5th epoch).
        """
        if self.loss_fn is None or self.optimizer is None:
            raise RuntimeError("You must call .compile() before .fit()")

        # Determine Training Mode
        is_ekf = isinstance(self.optimizer, EKFOptimizer)
        mode_name = "EKF" if is_ekf else "SGD"
        
        n_samples = len(X)
        has_val = validation_data is not None
        
        print(f"Starting Training | Mode: {mode_name} | Epochs: {epochs} | Samples: {n_samples}")
        if has_val:
            print(f"Validation enabled | Val Samples: {len(validation_data[0])}")
        
        start_time = time.time()

        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Reset Recurrent States at the start of each epoch
            self.reset_states()
            
            # --- Logic to control printing frequency ---
            # Print if: verbose is True AND (First Epoch OR Last Epoch OR Multiple of log_freq)
            should_log = verbose and ((epoch + 1) % log_freq == 0 or epoch == 0 or (epoch + 1) == epochs)
            
            # Create a progress bar ONLY if we are logging this epoch
            if should_log:
                pbar = tqdm(range(n_samples), desc=f"Epoch {epoch+1}/{epochs}", unit="sample")
            else:
                pbar = range(n_samples) # Silent iterator

            # --- Training Loop ---
            for i in pbar:
                xi = X[i]
                yi = y[i]

                # --- MODE 1: EKF ---
                if is_ekf:
                    current_error = self.optimizer.update(self, xi, yi)
                    loss_val = 0.5 * (current_error ** 2)
                    epoch_loss += loss_val

                # --- MODE 2: SGD ---
                else:
                    # 1. Forward
                    y_pred = self.forward(xi)
                    
                    # 2. Compute Loss
                    loss_val = self.loss_fn.compute(y_pred, yi)
                    epoch_loss += loss_val
                    
                    # 3. Backward & Update
                    grad_loss = self.loss_fn.gradient(y_pred, yi)
                    lrs = self.optimizer.get_learning_rates()
                    self.backward(grad_loss, lrs)

                # Update progress bar description (only if it exists)
                if should_log and isinstance(pbar, tqdm):
                    pbar.set_postfix({'train_loss': f"{loss_val:.5f}"})

            # --- End of Epoch Calculations ---
            avg_train_loss = epoch_loss / n_samples
            self.history['loss'].append(avg_train_loss)
            
            # --- Validation Loop ---
            val_msg = ""
            if has_val:
                X_val, y_val = validation_data
                avg_val_loss = self._compute_val_loss(X_val, y_val)
                self.history['val_loss'].append(avg_val_loss)
                val_msg = f" | Val Loss: {avg_val_loss:.6f}"
            
            # Print summary only if should_log is True
            if should_log:
                print(f"Epoch {epoch+1} finished. Train Loss: {avg_train_loss:.6f}{val_msg}")

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