import numpy as np
import time
from tqdm import tqdm  
from Core.base import Layer
from Core.optimizers import EKFOptimizer, SGD

class Sequential:
    def __init__(self, layers=None):
        self.layers = layers if layers else []
        self.loss_fn = None
        self.optimizer = None
        self.history = {'loss': [], 'val_loss': []}

    def add(self, layer):
        if not isinstance(layer, Layer):
            raise TypeError("Object must inherit from core.base.Layer")
        self.layers.append(layer)

    def compile(self, loss, optimizer):
        self.loss_fn = loss
        self.optimizer = optimizer

    def reset_states(self):
        for layer in self.layers:
            layer.reset_state()

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, loss_grad, learning_rates):
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad, **learning_rates)
        return grad

    def _compute_val_loss(self, X_val, y_val):
        if X_val is None or y_val is None:
            return None
        
        self.reset_states()
        
        val_loss_sum = 0.0
        n_val = len(X_val)
        
        for i in range(n_val):
            pred = self.forward(X_val[i])
            loss = self.loss_fn.compute(pred, y_val[i])
            val_loss_sum += loss
            
        return val_loss_sum / n_val

    def fit(self, X, y, validation_data=None, epochs=100, batch_size=1, verbose=True, log_freq=5):
        if self.loss_fn is None or self.optimizer is None:
            raise RuntimeError("You must call .compile() before .fit()")

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
            
            self.reset_states()
            
            should_log = verbose and ((epoch + 1) % log_freq == 0 or epoch == 0 or (epoch + 1) == epochs)
            
            if should_log:
                pbar = tqdm(range(n_samples), desc=f"Epoch {epoch+1}/{epochs}", unit="sample")
            else:
                pbar = range(n_samples) 

            for i in pbar:
                xi = X[i]
                yi = y[i]

                if is_ekf:
                    current_error = self.optimizer.update(self, xi, yi)
                    loss_val = 0.5 * (current_error ** 2)
                    epoch_loss += loss_val

                else:
                    y_pred = self.forward(xi)
                    
                    loss_val = self.loss_fn.compute(y_pred, yi)
                    epoch_loss += loss_val
                    
                    grad_loss = self.loss_fn.gradient(y_pred, yi)
                    lrs = self.optimizer.get_learning_rates()
                    self.backward(grad_loss, lrs)

                if should_log and isinstance(pbar, tqdm):
                    pbar.set_postfix({'train_loss': f"{loss_val:.5f}"})

            avg_train_loss = epoch_loss / n_samples
            self.history['loss'].append(avg_train_loss)
            
            val_msg = ""
            if has_val:
                X_val, y_val = validation_data
                avg_val_loss = self._compute_val_loss(X_val, y_val)
                self.history['val_loss'].append(avg_val_loss)
                val_msg = f" | Val Loss: {avg_val_loss:.6f}"
            
            if should_log:
                print(f"Epoch {epoch+1} finished. Train Loss: {avg_train_loss:.6f}{val_msg}")

        total_time = time.time() - start_time
        print(f"Training Complete. Time: {total_time:.2f}s")
        return self.history

    def predict(self, X):
        self.reset_states()
        predictions = []
        for i in range(len(X)):
            pred = self.forward(X[i])
            predictions.append(pred)
        return np.array(predictions)

    def summary(self):
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