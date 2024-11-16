import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import norm
import tensorflow_probability as tfp

class AdvancedUncertaintyEstimator:
    def __init__(self, 
                 model: tf.keras.Model,
                 n_samples: int = 100,
                 confidence_level: float = 0.95):
        self.model = model
        self.n_samples = n_samples
        self.confidence_level = confidence_level
        self.tfd = tfp.distributions
        
    def estimate_uncertainty(self, X: np.ndarray) -> Dict:
        """Estimate both epistemic and aleatoric uncertainty"""
        # Monte Carlo Dropout sampling
        predictions = []
        for _ in range(self.n_samples):
            pred = self.model(X, training=True)  # Enable dropout
            predictions.append(pred)
            
        predictions = np.array(predictions)
        
        # Calculate uncertainties
        epistemic = np.var(predictions, axis=0)  # Model uncertainty
        aleatoric = self._estimate_aleatoric(X)  # Data uncertainty
        
        # Calculate prediction intervals
        intervals = self._calculate_prediction_intervals(predictions)
        
        return {
            'epistemic': epistemic,
            'aleatoric': aleatoric,
            'total': epistemic + aleatoric,
            'prediction_intervals': intervals
        }
    
    def _estimate_aleatoric(self, X: np.ndarray) -> np.ndarray:
        """Estimate aleatoric uncertainty using probabilistic outputs"""
        # Multiple forward passes
        outputs = []
        for _ in range(self.n_samples):
            out = self.model(X, training=False)
            outputs.append(out)
            
        # Calculate variance of predictions
        return np.var(outputs, axis=0)
    
    def _calculate_prediction_intervals(self, 
                                     predictions: np.ndarray) -> Dict:
        """Calculate prediction intervals"""
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        z_score = norm.ppf((1 + self.confidence_level) / 2)
        
        lower = mean_pred - z_score * std_pred
        upper = mean_pred + z_score * std_pred
        
        return {
            'lower': lower,
            'upper': upper,
            'mean': mean_pred
        }
    
    def calculate_risk_metrics(self, X: np.ndarray) -> Dict:
        """Calculate risk metrics based on uncertainty"""
        uncertainty = self.estimate_uncertainty(X)
        
        # Value at Risk (VaR)
        var = self._calculate_var(uncertainty)
        
        # Expected Shortfall (ES)
        es = self._calculate_expected_shortfall(uncertainty)
        
        return {
            'value_at_risk': var,
            'expected_shortfall': es,
            'uncertainty_ratio': uncertainty['total'] / np.mean(uncertainty['total'])
        }
    
    def _calculate_var(self, uncertainty: Dict) -> float:
        """Calculate Value at Risk"""
        predictions = uncertainty['prediction_intervals']['mean']
        total_uncertainty = uncertainty['total']
        
        # Parametric VaR calculation
        z_score = norm.ppf(1 - self.confidence_level)
        var = predictions - z_score * np.sqrt(total_uncertainty)
        
        return var
    
    def _calculate_expected_shortfall(self, uncertainty: Dict) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        var = self._calculate_var(uncertainty)
        predictions = uncertainty['prediction_intervals']['mean']
        total_uncertainty = uncertainty['total']
        
        # Calculate ES using normal distribution assumption
        z_score = norm.ppf(1 - self.confidence_level)
        phi_z = norm.pdf(z_score)
        es = predictions - (phi_z / (1 - self.confidence_level)) * np.sqrt(total_uncertainty)
        
        return es
    
    def get_confidence_score(self, X: np.ndarray) -> float:
        """Calculate confidence score for predictions"""
        uncertainty = self.estimate_uncertainty(X)
        total_uncertainty = uncertainty['total']
        
        # Normalize uncertainty to [0, 1] range
        max_uncertainty = np.max(total_uncertainty)
        normalized_uncertainty = total_uncertainty / max_uncertainty
        
        # Convert to confidence score (inverse of uncertainty)
        confidence_score = 1 - normalized_uncertainty
        
        return confidence_score