import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import coint, adfuller

class CorrelationAnalyzer:
    def __init__(self, window_sizes: List[int] = [5, 15, 30, 60]):
        self.window_sizes = window_sizes
        self.correlations = {}
        
    def analyze(self, btc_data: pd.DataFrame, bch_data: pd.DataFrame) -> Dict:
        """Analyze correlations between BTC and BCH"""
        self.correlations = {
            'price': self._analyze_price_correlation(btc_data, bch_data),
            'volume': self._analyze_volume_correlation(btc_data, bch_data),
            'volatility': self._analyze_volatility_correlation(btc_data, bch_data),
            'cointegration': self._analyze_cointegration(btc_data, bch_data),
            'lead_lag': self._analyze_lead_lag(btc_data, bch_data)
        }
        
        return self.correlations
    
    def _analyze_price_correlation(self, btc: pd.DataFrame, bch: pd.DataFrame) -> Dict:
        """Analyze price correlations across different timeframes"""
        price_corr = {}
        
        for window in self.window_sizes:
            rolling_corr = btc['close'].rolling(window).corr(bch['close'])
            pearson_corr, _ = pearsonr(btc['close'].iloc[-window:], bch['close'].iloc[-window:])
            
            price_corr[f'window_{window}'] = {
                'rolling': rolling_corr.iloc[-1],
                'pearson': pearson_corr
            }
            
        return price_corr
    
    def _analyze_volume_correlation(self, btc: pd.DataFrame, bch: pd.DataFrame) -> Dict:
        """Analyze volume correlations"""
        volume_corr = {}
        
        for window in self.window_sizes:
            vol_corr = btc['volume'].rolling(window).corr(bch['volume'])
            volume_corr[f'window_{window}'] = vol_corr.iloc[-1]
            
        return volume_corr
    
    def _analyze_volatility_correlation(self, btc: pd.DataFrame, bch: pd.DataFrame) -> Dict:
        """Analyze volatility correlations"""
        volatility_corr = {}
        
        for window in self.window_sizes:
            btc_vol = btc['close'].pct_change().rolling(window).std()
            bch_vol = bch['close'].pct_change().rolling(window).std()
            vol_corr = btc_vol.rolling(window).corr(bch_vol)
            
            volatility_corr[f'window_{window}'] = vol_corr.iloc[-1]
            
        return volatility_corr
    
    def _analyze_cointegration(self, btc: pd.DataFrame, bch: pd.DataFrame) -> Dict:
        """Test for cointegration between BTC and BCH"""
        score, pvalue, _ = coint(btc['close'], bch['close'])
        
        # Test stationarity of spread
        spread = btc['close'] - bch['close']
        adf_result = adfuller(spread)
        
        return {
            'is_cointegrated': pvalue < 0.05,
            'cointegration_score': score,
            'p_value': pvalue,
            'spread_stationarity': {
                'adf_statistic': adf_result[0],
                'p_value': adf_result[1],
                'is_stationary': adf_result[1] < 0.05
            }
        }
    
    def _analyze_lead_lag(self, btc: pd.DataFrame, bch: pd.DataFrame, max_lag: int = 10) -> Dict:
        """Analyze lead-lag relationships"""
        lead_lag = {}
        
        for window in self.window_sizes:
            cross_corr = []
            
            for lag in range(-max_lag, max_lag + 1):
                if lag < 0:
                    corr = btc['close'].iloc[:-lag].corr(bch['close'].iloc[-lag:])
                else:
                    corr = btc['close'].iloc[lag:].corr(bch['close'].iloc[:-lag if lag > 0 else None])
                cross_corr.append((lag, corr))
            
            optimal_lag = max(cross_corr, key=lambda x: abs(x[1]))
            lead_lag[f'window_{window}'] = {
                'optimal_lag': optimal_lag[0],
                'correlation': optimal_lag[1]
            }
            
        return lead_lag
    
    def get_trading_signals(self) -> Dict:
        """Generate trading signals based on correlation analysis"""
        signals = {
            'correlation_strength': self._calculate_correlation_strength(),
            'trading_recommendation': self._generate_recommendation(),
            'risk_level': self._assess_risk()
        }
        
        return signals
    
    def _calculate_correlation_strength(self) -> float:
        """Calculate overall correlation strength"""
        if not self.correlations:
            return 0.0
            
        weights = {'price': 0.4, 'volume': 0.3, 'volatility': 0.3}
        strength = 0.0
        
        for metric, weight in weights.items():
            if metric in self.correlations:
                # Use the longest timeframe correlation
                window_key = f'window_{self.window_sizes[-1]}'
                corr_value = self.correlations[metric][window_key]
                
                if isinstance(corr_value, dict):
                    corr_value = corr_value['rolling']
                    
                strength += abs(corr_value) * weight
                
        return strength
    
    def _generate_recommendation(self) -> str:
        """Generate trading recommendation based on correlations"""
        strength = self._calculate_correlation_strength()
        
        if strength > 0.8:
            return "Strong correlation: Consider paired trading strategies"
        elif strength > 0.5:
            return "Moderate correlation: Monitor for trading opportunities"
        else:
            return "Weak correlation: Trade independently"
    
    def _assess_risk(self) -> str:
        """Assess trading risk based on correlation analysis"""
        if not self.correlations:
            return "Unknown"
            
        volatility_corr = self.correlations['volatility'][f'window_{self.window_sizes[-1]}']
        cointegration = self.correlations['cointegration']['is_cointegrated']
        
        if volatility_corr > 0.8 and cointegration:
            return "Low"
        elif volatility_corr > 0.5:
            return "Medium"
        else:
            return "High"