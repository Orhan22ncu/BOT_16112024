import tensorflow as tf
import numpy as np
from typing import List, Dict, Any
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

class HybridEnsemble:
    def __init__(self, 
                 deep_models: List[tf.keras.Model],
                 ml_models: List[Any] = None,
                 weights: Dict[str, float] = None):
        self.deep_models = deep_models
        self.ml_models = ml_models or [
            GradientBoostingRegressor(),
            RandomForestRegressor(),
            XGBRegressor(),
            LGBMRegressor()
        ]
        self.weights = weights
        self.meta_learner = self._build_meta_learner()
        
    def _build_meta_learner(self):
        """Build meta-learner for ensemble stacking"""
        return tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train all models and meta-learner"""
        # Train deep learning models
        dl_predictions = []
        for model in self.deep_models:
            model.fit(X, y, epochs=100, verbose=0)
            dl_predictions.append(model.predict(X))
        
        # Train ML models
        ml_predictions = []
        for model in self.ml_models:
            model.fit(X, y)
            ml_predictions.append(model.predict(X).reshape(-1, 1))
        
        # Combine predictions for meta-learning
        meta_features = np.hstack(dl_predictions + ml_predictions)
        
        # Train meta-learner
        self.meta_learner.fit(meta_features, y, epochs=50, verbose=0)
        
        # Calculate optimal weights if not provided
        if self.weights is None:
            self.weights = self._optimize_weights(meta_features, y)
    
    def predict(self, X: np.ndarray):
        """Generate ensemble predictions"""
        # Get predictions from all models
        dl_predictions = [model.predict(X) for model in self.deep_models]
        ml_predictions = [model.predict(X).reshape(-1, 1) for model in self.ml_models]
        
        # Combine predictions
        meta_features = np.hstack(dl_predictions + ml_predictions)
        
        # Meta-learner prediction
        meta_pred = self.meta_learner.predict(meta_features)
        
        # Weighted average of base predictions and meta-predictions
        base_pred = np.average(meta_features, weights=self.weights, axis=1)
        
        return 0.7 * meta_pred + 0.3 * base_pred
    
    def _optimize_weights(self, predictions: np.ndarray, y_true: np.ndarray):
        """Optimize ensemble weights using validation performance"""
        n_models = predictions.shape[1]
        
        def objective(weights):
            weighted_pred = np.average(predictions, weights=weights, axis=1)
            return np.mean((weighted_pred - y_true) ** 2)
        
        # Optimize weights using scipy
        from scipy.optimize import minimize
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(n_models)]
        
        result = minimize(
            objective,
            x0=np.ones(n_models) / n_models,
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x