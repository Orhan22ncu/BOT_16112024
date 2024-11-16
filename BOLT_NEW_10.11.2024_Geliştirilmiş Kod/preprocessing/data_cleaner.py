import numpy as np
import pandas as pd
from typing import Dict, List, Union
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler

class DataCleaner:
    def __init__(self, imputation_method: str = 'knn'):
        self.imputation_method = imputation_method
        self.imputer = KNNImputer(n_neighbors=5)
        self.scaler = RobustScaler()
        self.outlier_thresholds = {}
        
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Complete data cleaning pipeline"""
        df = data.copy()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Remove duplicates
        df = self._remove_duplicates(df)
        
        # Handle outliers
        df = self._handle_outliers(df)
        
        # Scale features
        df = self._scale_features(df)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using specified method"""
        if self.imputation_method == 'knn':
            return pd.DataFrame(
                self.imputer.fit_transform(df),
                columns=df.columns
            )
        else:
            return df.fillna(df.mean())
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows"""
        return df.drop_duplicates()
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using IQR method"""
        for column in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            self.outlier_thresholds[column] = {
                'lower': lower_bound,
                'upper': upper_bound
            }
            
            df[column] = df[column].clip(lower_bound, upper_bound)
        
        return df
    
    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale features using RobustScaler"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        return df