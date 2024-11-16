import tensorflow as tf
import numpy as np
from typing import List, Dict, Any
from sklearn.model_selection import cross_val_score

class DynamicModelSelector:
    def __init__(self, models: List[tf.keras.Model]):
        self.models = models
        self.performance_history = {}
        self.current_best = None
        
    def select_best_model(self, X: np.ndarray, y: np.ndarray):
        """Select best performing model for current data"""
        scores = []
        
        for i, model in enumerate(self.models):
            # Cross-validation score
            score = np.mean(cross_val_score(model, X, y, cv=5))
            scores.append(score)
            
            # Update performance history
            if i not in self.performance_history:
                self.performance_history[i] = []
            self.performance_history[i].append(score)
        
        # Select best model
        best_idx = np.argmax(scores)
        self.current_best = self.models[best_idx]
        
        return self.current_best
    
    def get_ensemble_weights(self):
        """Calculate ensemble weights based on performance history"""
        weights = []
        
        for i in range(len(self.models)):
            if i in self.performance_history:
                # Use recent performance for weighting
                recent_perf = np.mean(self.performance_history[i][-5:])
                weights.append(recent_perf)
            else:
                weights.append(0)
        
        # Normalize weights
        weights = np.array(weights)
        return weights / np.sum(weights)
    
    def predict_with_ensemble(self, X: np.ndarray):
        """Make prediction using weighted ensemble"""
        weights = self.get_ensemble_weights()
        predictions = []
        
        for model, weight in zip(self.models, weights):
            pred = model.predict(X)
            predictions.append(pred * weight)
        
        return np.sum(predictions, axis=0)