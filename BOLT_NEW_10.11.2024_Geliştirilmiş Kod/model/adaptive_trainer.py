import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple
from sklearn.model_selection import TimeSeriesSplit
from .meta_learner import MetaLearner
from .uncertainty_estimator import UncertaintyEstimator

class AdaptiveTrainer:
    def __init__(self,
                 base_model: tf.keras.Model,
                 meta_learner: MetaLearner,
                 uncertainty_estimator: UncertaintyEstimator):
        self.base_model = base_model
        self.meta_learner = meta_learner
        self.uncertainty_estimator = uncertainty_estimator
        self.training_history = []
        
    def train_adaptively(self,
                        X: np.ndarray,
                        y: np.ndarray,
                        validation_split: float = 0.2,
                        n_epochs: int = 100):
        """Adaptive training with dynamic adjustments"""
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Get meta-learning suggestions
            meta_params = self.meta_learner.suggest_parameters(
                X_train, y_train
            )
            
            # Train with dynamic adjustments
            history = self._train_with_adjustments(
                X_train, y_train,
                X_val, y_val,
                meta_params,
                n_epochs
            )
            
            self.training_history.append(history)
            
            # Update meta-learner
            self.meta_learner.update(history)
    
    def _train_with_adjustments(self,
                              X_train: np.ndarray,
                              y_train: np.ndarray,
                              X_val: np.ndarray,
                              y_val: np.ndarray,
                              meta_params: Dict,
                              n_epochs: int) -> Dict:
        """Train with dynamic parameter adjustments"""
        history = {'loss': [], 'val_loss': [], 'adjustments': []}
        
        for epoch in range(n_epochs):
            # Get uncertainty estimates
            uncertainties = self.uncertainty_estimator.estimate_uncertainty(X_train)
            
            # Adjust sample weights based on uncertainty
            sample_weights = self._calculate_sample_weights(uncertainties)
            
            # Train one epoch
            epoch_history = self.base_model.fit(
                X_train, y_train,
                sample_weight=sample_weights,
                validation_data=(X_val, y_val),
                epochs=1,
                verbose=0
            )
            
            # Record metrics
            history['loss'].append(epoch_history.history['loss'][0])
            history['val_loss'].append(epoch_history.history['val_loss'][0])
            
            # Dynamic adjustments
            adjustments = self._make_dynamic_adjustments(
                epoch_history.history,
                meta_params
            )
            history['adjustments'].append(adjustments)
            
            # Early stopping check
            if self._should_stop_early(history):
                break
        
        return history
    
    def _calculate_sample_weights(self, uncertainties: np.ndarray) -> np.ndarray:
        """Calculate sample weights based on uncertainty"""
        # Higher weights for more uncertain samples
        weights = 1 + uncertainties / np.max(uncertainties)
        return weights / np.sum(weights)
    
    def _make_dynamic_adjustments(self,
                                epoch_metrics: Dict,
                                meta_params: Dict) -> Dict:
        """Make dynamic adjustments to training parameters"""
        adjustments = {}
        
        # Learning rate adjustment
        if epoch_metrics['loss'][-1] > epoch_metrics['loss'][-2]:
            adjustments['learning_rate'] = meta_params['learning_rate'] * 0.8
        
        # Batch size adjustment
        if epoch_metrics['val_loss'][-1] < epoch_metrics['val_loss'][-2]:
            adjustments['batch_size'] = min(
                meta_params['batch_size'] * 2,
                256
            )
        
        return adjustments
    
    def _should_stop_early(self, history: Dict, patience: int = 10) -> bool:
        """Check early stopping condition"""
        if len(history['val_loss']) < patience:
            return False
            
        # Check if validation loss hasn't improved
        best_loss = min(history['val_loss'][:-patience])
        recent_best = min(history['val_loss'][-patience:])
        
        return recent_best >= best_loss
    
    def get_training_summary(self) -> Dict:
        """Get summary of training process"""
        summary = {
            'total_epochs': sum(len(h['loss']) for h in self.training_history),
            'final_loss': self.training_history[-1]['loss'][-1],
            'final_val_loss': self.training_history[-1]['val_loss'][-1],
            'adjustments_made': sum(
                len(h['adjustments']) for h in self.training_history
            )
        }
        
        # Calculate improvement metrics
        initial_loss = self.training_history[0]['loss'][0]
        final_loss = summary['final_loss']
        summary['improvement'] = (initial_loss - final_loss) / initial_loss
        
        return summary