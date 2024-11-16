import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import coint, adfuller
from sklearn.preprocessing import StandardScaler

class AdvancedCorrelationAnalyzer:
    def __init__(self, 
                 base_symbol: str = 'BTCUSDT',
                 target_symbol: str = 'BCHUSDT',
                 window_sizes: List[int] = [5, 15, 30, 60]):
        self.base_symbol = base_symbol
        self.target_symbol = target_symbol
        self.window_sizes = window_sizes
        self.scaler = StandardScaler()
        
    def analyze_correlations(self, 
                           btc_data: pd.DataFrame,
                           bch_data: pd.DataFrame) -> Dict:
        """Enhanced correlation analysis with multiple timeframes"""
        correlations = {
            'price_correlations': self._analyze_price_correlations(btc_data, bch_data),
            'volume_correlations': self._analyze_volume_correlations(btc_data, bch_data),
            'volatility_correlations': self._analyze_volatility_correlations(btc_data, bch_data),
            'momentum_correlations': self._analyze_momentum_correlations(btc_data, bch_data),
            'lead_lag': self._analyze_lead_lag_relationship(btc_data, bch_data),
            'cointegration': self._test_cointegration(btc_data['close'], bch_data['close']),
            'regime_correlations': self._analyze_regime_correlations(btc_data, bch_data)
        }
        
        # Calculate dynamic correlation strength
        correlations['dynamic_strength'] = self._calculate_dynamic_strength(correlations)
        
        return correlations
    
    def _analyze_price_correlations(self, 
                                  btc_data: pd.DataFrame,
                                  bch_data: pd.DataFrame) -> Dict:
        """Analyze price correlations across multiple timeframes"""
        correlations = {}
        
        for window in self.window_sizes:
            # Rolling correlation
            rolling_corr = btc_data['close'].rolling(window).corr(bch_data['close'])
            
            # Pearson correlation
            pearson_corr, _ = pearsonr(
                btc_data['close'].iloc[-window:],
                bch_data['close'].iloc[-window:]
            )
            
            # Spearman rank correlation
            spearman_corr, _ = spearmanr(
                btc_data['close'].iloc[-window:],
                bch_data['close'].iloc[-window:]
            )
            
            correlations[f'window_{window}'] = {
                'rolling': rolling_corr.iloc[-1],
                'pearson': pearson_corr,
                'spearman': spearman_corr
            }
        
        return correlations
    
    def _analyze_volume_correlations(self,
                                   btc_data: pd.DataFrame,
                                   bch_data: pd.DataFrame) -> Dict:
        """Analyze volume correlations"""
        volume_corr = {}
        
        for window in self.window_sizes:
            # Volume correlation
            vol_corr = btc_data['volume'].rolling(window).corr(bch_data['volume'])
            
            # Volume-price impact correlation
            btc_vol_price = (btc_data['volume'] * btc_data['close'].pct_change()).rolling(window)
            bch_vol_price = (bch_data['volume'] * bch_data['close'].pct_change()).rolling(window)
            impact_corr = btc_vol_price.corr(bch_vol_price)
            
            volume_corr[f'window_{window}'] = {
                'volume_correlation': vol_corr.iloc[-1],
                'impact_correlation': impact_corr.iloc[-1]
            }
        
        return volume_corr
    
    def _analyze_volatility_correlations(self,
                                       btc_data: pd.DataFrame,
                                       bch_data: pd.DataFrame) -> Dict:
        """Analyze volatility correlations"""
        volatility_corr = {}
        
        for window in self.window_sizes:
            # Calculate volatilities
            btc_vol = btc_data['close'].pct_change().rolling(window).std()
            bch_vol = bch_data['close'].pct_change().rolling(window).std()
            
            # Volatility correlation
            vol_corr = btc_vol.rolling(window).corr(bch_vol)
            
            # Realized volatility correlation
            btc_realized = np.sqrt(np.sum(btc_data['close'].pct_change().iloc[-window:]**2))
            bch_realized = np.sqrt(np.sum(bch_data['close'].pct_change().iloc[-window:]**2))
            
            volatility_corr[f'window_{window}'] = {
                'volatility_correlation': vol_corr.iloc[-1],
                'realized_correlation': np.corrcoef(btc_realized, bch_realized)[0,1]
            }
        
        return volatility_corr
    
    def _analyze_momentum_correlations(self,
                                     btc_data: pd.DataFrame,
                                     bch_data: pd.DataFrame) -> Dict:
        """Analyze momentum correlations"""
        momentum_corr = {}
        
        for window in self.window_sizes:
            # Calculate momentum indicators
            btc_rsi = self._calculate_rsi(btc_data['close'], window)
            bch_rsi = self._calculate_rsi(bch_data['close'], window)
            
            btc_macd = self._calculate_macd(btc_data['close'])
            bch_macd = self._calculate_macd(bch_data['close'])
            
            momentum_corr[f'window_{window}'] = {
                'rsi_correlation': btc_rsi.corr(bch_rsi),
                'macd_correlation': btc_macd.corr(bch_macd)
            }
        
        return momentum_corr
    
    def _analyze_lead_lag_relationship(self,
                                     btc_data: pd.DataFrame,
                                     bch_data: pd.DataFrame,
                                     max_lag: int = 10) -> Dict:
        """Analyze lead-lag relationships"""
        lead_lag = {}
        
        for window in self.window_sizes:
            cross_corr = []
            
            for lag in range(-max_lag, max_lag + 1):
                if lag < 0:
                    corr = btc_data['close'].iloc[:-lag].corr(
                        bch_data['close'].iloc[-lag:]
                    )
                else:
                    corr = btc_data['close'].iloc[lag:].corr(
                        bch_data['close'].iloc[:-lag if lag > 0 else None]
                    )
                cross_corr.append((lag, corr))
            
            # Find optimal lag
            optimal_lag = max(cross_corr, key=lambda x: abs(x[1]))
            
            lead_lag[f'window_{window}'] = {
                'optimal_lag': optimal_lag[0],
                'lag_correlation': optimal_lag[1],
                'cross_correlations': dict(cross_corr)
            }
        
        return lead_lag
    
    def _test_cointegration(self,
                           btc_prices: pd.Series,
                           bch_prices: pd.Series) -> Dict:
        """Test for cointegration relationship"""
        # Standardize prices
        btc_std = self.scaler.fit_transform(btc_prices.values.reshape(-1, 1)).flatten()
        bch_std = self.scaler.fit_transform(bch_prices.values.reshape(-1, 1)).flatten()
        
        # Perform cointegration test
        score, pvalue, _ = coint(btc_std, bch_std)
        
        # Test stationarity of spread
        spread = btc_std - bch_std
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
    
    def _analyze_regime_correlations(self,
                                   btc_data: pd.DataFrame,
                                   bch_data: pd.DataFrame) -> Dict:
        """Analyze correlations under different market regimes"""
        regime_corr = {}
        
        # Define market regimes based on volatility
        btc_vol = btc_data['close'].pct_change().rolling(20).std()
        regimes = pd.qcut(btc_vol, q=3, labels=['low_vol', 'med_vol', 'high_vol'])
        
        for regime in ['low_vol', 'med_vol', 'high_vol']:
            mask = regimes == regime
            regime_corr[regime] = {
                'price_correlation': btc_data.loc[mask, 'close'].corr(
                    bch_data.loc[mask, 'close']
                ),
                'volume_correlation': btc_data.loc[mask, 'volume'].corr(
                    bch_data.loc[mask, 'volume']
                )
            }
        
        return regime_corr
    
    def _calculate_dynamic_strength(self, correlations: Dict) -> float:
        """Calculate dynamic correlation strength"""
        weights = {
            'price': 0.4,
            'volume': 0.2,
            'volatility': 0.2,
            'momentum': 0.2
        }
        
        strength = 0
        
        # Weight different correlation components
        for component, weight in weights.items():
            if f'{component}_correlations' in correlations:
                # Get the most recent correlation value
                recent_corr = correlations[f'{component}_correlations'][
                    f'window_{self.window_sizes[-1]}'
                ]
                
                if isinstance(recent_corr, dict):
                    # Use the first correlation value if it's a dictionary
                    corr_value = list(recent_corr.values())[0]
                else:
                    corr_value = recent_corr
                
                strength += weight * abs(corr_value)
        
        return strength
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self,
                       prices: pd.Series,
                       fast: int = 12,
                       slow: int = 26) -> pd.Series:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        return exp1 - exp2