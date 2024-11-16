import numpy as np
import pandas as pd
from typing import List, Dict, Union
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression

class FeatureProcessor:
    def __init__(self, 
                 n_components: float = 0.95,
                 feature_selection_method: str = 'mutual_info'):
        self.n_components = n_components
        self.feature_selection_method = feature_selection_method
        self.pca = PCA(n_components=n_components)
        self.selected_features = None
        
    def process_features(self, 
                        data: pd.DataFrame,
                        target: pd.Series = None) -> pd.DataFrame:
        """Complete feature processing pipeline"""
        df = data.copy()
        
        # Create interaction features
        df = self._create_interactions(df)
        
        # Create polynomial features
        df = self._create_polynomial_features(df)
        
        # Reduce dimensionality
        if target is not None:
            df = self._select_features(df, target)
        else:
            df = self._reduce_dimensions(df)
        
        return df
    
    def _create_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                df[f"{col1}_{col2}_interaction"] = df[col1] * df[col2]
        
        return df
    
    def _create_polynomial_features(self, 
                                  df: pd.DataFrame,
                                  degree: int = 2) -> pd.DataFrame:
        """Create polynomial features"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            for d in range(2, degree + 1):
                df[f"{col}_poly_{d}"] = df[col] ** d
        
        return df
    
    def _select_features(self, 
                        df: pd.DataFrame,
                        target: pd.Series) -> pd.DataFrame:
        """Select most important features"""
        if self.feature_selection_method == 'mutual_info':
            scores = mutual_info_regression(df, target)
            importance = pd.Series(scores, index=df.columns)
            selected = importance.nlargest(
                int(len(df.columns) * self.n_components)
            ).index
            
            self.selected_features = selected
            return df[selected]
        
        return df
    
    def _reduce_dimensions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reduce dimensions using PCA"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        pca_result = self.pca.fit_transform(df[numeric_cols])
        
        return pd.DataFrame(
            pca_result,
            columns=[f'PC_{i+1}' for i in range(pca_result.shape[1])]
        )