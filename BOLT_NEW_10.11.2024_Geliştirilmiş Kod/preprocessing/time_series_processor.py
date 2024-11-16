import numpy as np
import pandas as pd
from typing import List, Dict, Union
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import signal

class TimeSeriesProcessor:
    def __init__(self, 
                 seasonality_period: int = None,
                 smoothing_window: int = 5):
        self.seasonality_period = seasonality_period
        self.smoothing_window = smoothing_window
        
    def process_time_series(self, 
                          data: pd.DataFrame,
                          timestamp_col: str) -> pd.DataFrame:
        """Complete time series processing pipeline"""
        df = data.copy()
        df = self._add_time_features(df, timestamp_col)
        df = self._decompose_series(df)
        df = self._add_lagged_features(df)
        df = self._smooth_series(df)
        return df
    
    def _add_time_features(self, 
                          df: pd.DataFrame,
                          timestamp_col: str) -> pd.DataFrame:
        """Add time-based features"""
        df['hour'] = df[timestamp_col].dt.hour
        df['day'] = df[timestamp_col].dt.day
        df['month'] = df[timestamp_col].dt.month
        df['year'] = df[timestamp_col].dt.year
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
        
        return df
    
    def _decompose_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """Decompose time series into trend, seasonal, and residual"""
        if self.seasonality_period:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                try:
                    decomposition = seasonal_decompose(
                        df[col],
                        period=self.seasonality_period,
                        extrapolate_trend='freq'
                    )
                    
                    df[f'{col}_trend'] = decomposition.trend
                    df[f'{col}_seasonal'] = decomposition.seasonal
                    df[f'{col}_residual'] = decomposition.resid
                except:
                    continue
        
        return df
    
    def _add_lagged_features(self, 
                           df: pd.DataFrame,
                           lags: List[int] = [1, 7, 30]) -> pd.DataFrame:
        """Add lagged features"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def _smooth_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply smoothing to numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            df[f'{col}_smooth'] = signal.savgol_filter(
                df[col],
                window_length=self.smoothing_window,
                polyorder=2
            )
        
        return df