import tensorflow as tf
import numpy as np
from typing import List, Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error

class ModelEvaluator:
    def __init__(self, metrics: List[str] = None):
        self.metrics = metrics or ['mse', 'mae', 'mape']
        self.results = {}
        
    def evaluate_model(self, 
                      model: tf.keras.Model,
                      X: np.ndarray,
                      y: np.ndarray):
        """Comprehensive model evaluation"""
        predictions = model.predict(X)
        
        metrics = {
            'mse': mean_squared_error(y, predictions),
            'mae': mean_absolute_error(y, predictions),
            'mape': self._calculate_mape(y, predictions),
            'r2': self._calculate_r2(y, predictions)
        }
        
        # Additional analysis
        residuals = y - predictions
        metrics.update({
            'residual_std': np.std(residuals),
            'prediction_bias': np.mean(residuals),
            'max_error': np.max(np.abs(residuals))
        })
        
        return metrics
    
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calculate Mean Absolute Percentage Error"""
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    def _calculate_r2(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calculate R-squared score"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)