import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA

class AdaptiveFeatureSelector:
    def __init__(self, n_features=10):
        self.n_features = n_features
        self.feature_selector = SelectKBest(score_func=f_regression, k=n_features)
        self.pca = PCA(n_components=0.95)  # 95% varyans korunumu
        
    def select_features(self, X, y):
        """Önemli özellikleri seç"""
        selected_features = self.feature_selector.fit_transform(X, y)
        feature_importance = self.feature_selector.scores_
        selected_indices = self.feature_selector.get_support()
        
        return selected_features, feature_importance, selected_indices
    
    def reduce_dimensions(self, X):
        """PCA ile boyut azaltma"""
        return self.pca.fit_transform(X)
    
    def get_feature_importance(self):
        """Özellik önem sıralaması"""
        return dict(zip(
            range(len(self.feature_selector.scores_)),
            self.feature_selector.scores_
        ))