import numpy as np
import pandas as pd
from typing import List, Dict, Union
from scipy import stats

class FeatureValidator:
    def __init__(self, 
                 correlation_threshold: float = 0.95,
                 variance_threshold: float = 0.01):
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold
        self.removed_features = []
        
    def validate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Complete feature validation pipeline"""
        df = data.copy()
        
        # Remove low variance features
        df = self._remove_low_variance(df)
        
        # Remove highly correlated features
        df = self._remove_correlations(df)
        
        # Remove constant and quasi-constant features
        df = self._remove_constants(df)
        
        # Validate feature distributions
        df = self._validate_distributions(df)
        
        return df
    
    def _remove_low_variance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove features with low variance"""
        variances = df.var()
        low_variance = variances[variances < self.variance_threshold].index
        
        self.removed_features.extend(
            [(col, 'low_variance') for col in low_variance]
        )
        
        return df.drop(columns=low_variance)
    
    def _remove_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features"""
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [
            column for column in upper.columns 
            if any(upper[column] > self.correlation_threshold)
        ]
        
        self.removed_features.extend(
            [(col, 'high_correlation') for col in to_drop]
        )
        
        return df.drop(columns=to_drop)
    
    def _remove_constants(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove constant and quasi-constant features"""
        nunique = df.nunique()
        constants = nunique[nunique <= 1].index
        
        self.removed_features.extend(
            [(col, 'constant') for col in constants]
        )
        
        return df.drop(columns=constants)
    
    def _validate_distributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and transform feature distributions"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Test for normality
            _, p_value = stats.normaltest(df[col].dropna())
            
            # Apply transformations if needed
            if p_value < 0.05:
                if df[col].min() >= 0:
                    # Log transform for right-skewed data
                    df[f'{col}_log'] = np.log1p(df[col])
                else:
                    # Box-Cox transform for other cases
                    df[f'{col}_boxcox'], _ = stats.boxcox(
                        df[col] - df[col].min() + 1
                    )
        
        return df