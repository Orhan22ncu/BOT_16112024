import tensorflow as tf
import numpy as np
from typing import List, Dict

class AdaptiveLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self,
                 initial_lr: float = 0.001,
                 min_lr: float = 1e-6,
                 patience: int = 5,
                 warmup_epochs: int = 5):
        super().__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.patience = patience
        self.warmup_epochs = warmup_epochs
        self.best_loss = float('inf')
        self.wait = 0
        
    def on_epoch_begin(self, epoch, logs=None):
        """Adjust learning rate at the beginning of each epoch"""
        if epoch < self.warmup_epochs:
            # Warmup phase: gradually increase learning rate
            lr = self.initial_lr * ((epoch + 1) / self.warmup_epochs)
        else:
            # After warmup: adjust based on performance
            if self.wait >= self.patience:
                lr = max(
                    self.min_lr,
                    tf.keras.backend.get_value(self.model.optimizer.lr) * 0.5
                )
                self.wait = 0
            else:
                lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        print(f"\nEpoch {epoch + 1}: Learning rate = {lr:.6f}")
    
    def on_epoch_end(self, epoch, logs=None):
        """Update best loss and wait counter"""
        current_loss = logs.get('val_loss') or logs.get('loss')
        
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1