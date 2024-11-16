import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
import tensorflow as tf

class DynamicFeatureSelector:
    def __init__(self, n_features=10, selection_method='hybrid'):
        self.n_features = n_features
        self.selection_method = selection_method
        self.selected_features = None
        self.feature_importance = None
        self.pca = PCA(n_components=0.95)
        
    def select_features(self, X, y=None):
        """Dynamic feature selection based on multiple criteria"""
        if self.selection_method == 'hybrid':
            return self._hybrid_selection(X, y)
        elif self.selection_method == 'mutual_info':
            return self._mutual_info_selection(X, y)
        elif self.selection_method == 'pca':
            return self._pca_selection(X)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
    
    def _hybrid_selection(self, X, y):
        """Combine multiple feature selection methods"""
        # Mutual information scores
        mi_scores = mutual_info_regression(X, y)
        
        # PCA components
        pca_features = self.pca.fit_transform(X)
        
        # Combine scores
        combined_scores = self._combine_scores(mi_scores, self.pca.components_)
        
        # Select top features
        self.selected_features = np.argsort(combined_scores)[-self.n_features:]
        self.feature_importance = combined_scores
        
        return X[:, self.selected_features]
    
    def _mutual_info_selection(self, X, y):
        """Select features based on mutual information"""
        scores = mutual_info_regression(X, y)
        self.selected_features = np.argsort(scores)[-self.n_features:]
        self.feature_importance = scores
        
        return X[:, self.selected_features]
    
    def _pca_selection(self, X):
        """Select features using PCA"""
        pca_features = self.pca.fit_transform(X)
        self.selected_features = range(pca_features.shape[1])
        self.feature_importance = self.pca.explained_variance_ratio_
        
        return pca_features
    
    def _combine_scores(self, mi_scores, pca_components):
        """Combine different feature importance scores"""
        # Normalize scores
        mi_normalized = mi_scores / np.sum(mi_scores)
        pca_normalized = np.abs(pca_components[0]) / np.sum(np.abs(pca_components[0]))
        
        # Weighted combination
        return 0.7 * mi_normalized + 0.3 * pca_normalized
    
    def transform(self, X):
        """Transform new data using selected features"""
        if self.selected_features is None:
            raise ValueError("Feature selector has not been fitted yet")
        
        if self.selection_method == 'pca':
            return self.pca.transform(X)
        else:
            return X[:, self.selected_features]
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        if self.feature_importance is None:
            raise ValueError("Feature importance scores not available")
        
        return dict(enumerate(self.feature_importance))
    
    def update_selection(self, X, y, window_size=1000):
        """Update feature selection based on recent data"""
        if len(X) > window_size:
            X = X[-window_size:]
            y = y[-window_size:]
        
        return self.select_features(X, y)