import numpy as np
import pandas as pd
from typing import List, Dict, Union
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, mutual_info_regression

class AutoFeatureGenerator:
    def __init__(self, 
                 max_interactions: int = 2,
                 max_polynomials: int = 2,
                 n_best_features: int = 100):
        self.max_interactions = max_interactions
        self.max_polynomials = max_polynomials
        self.n_best_features = n_best_features
        self.feature_scores = {}
        
    def generate_features(self, 
                         data: pd.DataFrame,
                         target: pd.Series = None) -> pd.DataFrame:
        """Automated feature generation pipeline"""
        df = data.copy()
        
        # Generate basic statistical features
        df = self._generate_statistical_features(df)
        
        # Generate domain-specific features
        df = self._generate_domain_features(df)
        
        # Generate interaction features
        df = self._generate_interactions(df)
        
        # Select best features
        if target is not None:
            df = self._select_best_features(df, target)
        
        return df
    
    def _generate_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate statistical features"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Rolling statistics
        windows = [5, 10, 20]
        for col in numeric_cols:
            for window in windows:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window).std()
                df[f'{col}_rolling_min_{window}'] = df[col].rolling(window).min()
                df[f'{col}_rolling_max_{window}'] = df[col].rolling(window).max()
        
        return df
    
    def _generate_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate domain-specific features"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Momentum indicators
            df[f'{col}_momentum'] = df[col].diff()
            df[f'{col}_acceleration'] = df[f'{col}_momentum'].diff()
            
            # Relative strength
            df[f'{col}_rel_strength'] = df[col] / df[col].rolling(10).mean()
            
            # Volatility
            df[f'{col}_volatility'] = df[col].rolling(20).std()
        
        return df
    
    def _generate_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate interaction features"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        poly = PolynomialFeatures(
            degree=self.max_polynomials,
            interaction_only=True,
            include_bias=False
        )
        
        interactions = poly.fit_transform(df[numeric_cols])
        feature_names = poly.get_feature_names(numeric_cols)
        
        interaction_df = pd.DataFrame(
            interactions,
            columns=feature_names,
            index=df.index
        )
        
        return pd.concat([df, interaction_df], axis=1)
    
    def _select_best_features(self,
                            df: pd.DataFrame,
                            target: pd.Series) -> pd.DataFrame:
        """Select best features based on mutual information"""
        selector = SelectKBest(
            score_func=mutual_info_regression,
            k=self.n_best_features
        )
        
        selected_features = selector.fit_transform(df, target)
        selected_cols = df.columns[selector.get_support()]
        
        # Store feature scores
        self.feature_scores = dict(zip(
            df.columns,
            selector.scores_
        ))
        
        return pd.DataFrame(selected_features, columns=selected_cols)