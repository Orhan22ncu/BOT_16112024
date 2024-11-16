import tensorflow as tf
import numpy as np
from typing import List, Dict
import logging

class ErrorAnalyzer:
    def __init__(self, model: tf.keras.Model):
        self.model = model
        self.logger = self._setup_logger()
        self.error_patterns = {}
        
    def _setup_logger(self):
        """Setup logging configuration"""
        logger = logging.getLogger('ErrorAnalyzer')
        logger.setLevel(logging.DEBUG)
        
        handler = logging.FileHandler('error_analysis.log')
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(handler)
        
        return logger
    
    def analyze_errors(self, X: np.ndarray, y: np.ndarray):
        """Analyze prediction errors"""
        predictions = self.model.predict(X)
        errors = y - predictions
        
        # Analyze error patterns
        patterns = {
            'systematic_bias': self._check_systematic_bias(errors),
            'outliers': self._detect_outliers(errors),
            'error_clusters': self._analyze_error_clusters(X, errors)
        }
        
        self.error_patterns = patterns
        self._log_analysis_results(patterns)
        
        return patterns
    
    def _check_systematic_bias(self, errors: np.ndarray):
        """Check for systematic bias in predictions"""
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        return {
            'mean_error': mean_error,
            'std_error': std_error,
            'has_systematic_bias': abs(mean_error) > std_error
        }
    
    def _detect_outliers(self, errors: np.ndarray):
        """Detect outlier predictions"""
        threshold = 3 * np.std(errors)
        outliers = np.abs(errors) > threshold
        
        return {
            'num_outliers': np.sum(outliers),
            'outlier_indices': np.where(outliers)[0],
            'outlier_values': errors[outliers]
        }
    
    def _analyze_error_clusters(self, X: np.ndarray, errors: np.ndarray):
        """Analyze error clusters in feature space"""
        from sklearn.cluster import DBSCAN
        
        # Combine features and errors
        data = np.column_stack([X, errors])
        
        # Cluster analysis
        clustering = DBSCAN(eps=0.5, min_samples=5)
        clusters = clustering.fit_predict(data)
        
        return {
            'num_clusters': len(np.unique(clusters)),
            'cluster_sizes': np.bincount(clusters[clusters >= 0]),
            'cluster_labels': clusters
        }
    
    def _log_analysis_results(self, patterns: Dict):
        """Log analysis results"""
        self.logger.info("Error Analysis Results:")
        self.logger.info(f"Systematic Bias: {patterns['systematic_bias']}")
        self.logger.info(f"Outliers: {patterns['outliers']['num_outliers']}")
        self.logger.info(f"Error Clusters: {patterns['error_clusters']['num_clusters']}")
    
    def suggest_improvements(self):
        """Suggest improvements based on error analysis"""
        suggestions = []
        
        if self.error_patterns['systematic_bias']['has_systematic_bias']:
            suggestions.append(
                "Consider adding bias correction layer or retraining with balanced data"
            )
        
        if self.error_patterns['outliers']['num_outliers'] > 0:
            suggestions.append(
                "Implement outlier detection and handling in preprocessing"
            )
        
        if self.error_patterns['error_clusters']['num_clusters'] > 1:
            suggestions.append(
                "Consider using specialized models for different data clusters"
            )
        
        return suggestions