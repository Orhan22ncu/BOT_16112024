import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy import stats

class MarketRegimeDetector:
    def __init__(self, n_regimes: int = 3, window_size: int = 20):
        self.n_regimes = n_regimes
        self.window_size = window_size
        self.gmm = GaussianMixture(
            n_components=n_regimes,
            random_state=42,
            covariance_type='full'
        )
        self.scaler = StandardScaler()
        self.regime_history = []
        
    def detect_regime(self, data: pd.DataFrame) -> Dict:
        """Detect current market regime"""
        # Extract regime features
        features = self._extract_regime_features(data)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Detect regimes
        regimes = self.gmm.fit_predict(scaled_features)
        regime_probs = self.gmm.predict_proba(scaled_features)
        
        # Analyze regime characteristics
        regime_analysis = self._analyze_regime_characteristics(
            data, regimes, regime_probs
        )
        
        # Update regime history
        self.regime_history.append(regime_analysis)
        
        return regime_analysis
    
    def _extract_regime_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features for regime detection"""
        features = []
        
        # Price based features
        returns = data['close'].pct_change()
        log_returns = np.log1p(returns)
        
        for window in [5, 10, 20]:
            # Trend features
            features.append(returns.rolling(window).mean())
            features.append(log_returns.rolling(window).mean())
            
            # Volatility features
            features.append(returns.rolling(window).std())
            features.append(np.abs(returns).rolling(window).mean())
            
            # Momentum features
            features.append(self._calculate_rsi(data['close'], window))
            features.append(
                data['close'].rolling(window).mean() / data['close'] - 1
            )
        
        # Volume features
        volume_ma = data['volume'].rolling(self.window_size).mean()
        features.append(data['volume'] / volume_ma)
        
        # Combine features
        feature_matrix = pd.concat(features, axis=1).dropna()
        
        return feature_matrix.values
    
    def _analyze_regime_characteristics(self,
                                     data: pd.DataFrame,
                                     regimes: np.ndarray,
                                     regime_probs: np.ndarray) -> Dict:
        """Analyze characteristics of detected regimes"""
        regime_chars = {}
        
        for i in range(self.n_regimes):
            mask = regimes == i
            regime_data = data[mask]
            
            if len(regime_data) > 0:
                regime_chars[f'regime_{i}'] = {
                    'trend_strength': self._calculate_trend_strength(regime_data),
                    'volatility_level': self._calculate_volatility_level(regime_data),
                    'momentum_strength': self._calculate_momentum_strength(regime_data),
                    'volume_profile': self._analyze_volume_profile(regime_data),
                    'probability': regime_probs[-1][i]
                }
        
        # Determine current regime
        current_regime = {
            'regime_label': regimes[-1],
            'regime_probability': regime_probs[-1].max(),
            'characteristics': regime_chars[f'regime_{regimes[-1]}']
        }
        
        return {
            'current_regime': current_regime,
            'regime_characteristics': regime_chars,
            'regime_transitions': self._analyze_regime_transitions(regimes),
            'regime_stability': self._calculate_regime_stability(regimes)
        }
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength using multiple indicators"""
        # Linear regression slope
        x = np.arange(len(data))
        slope, _, r_value, _, _ = stats.linregress(x, data['close'])
        
        # Moving average convergence/divergence
        ma_short = data['close'].rolling(5).mean()
        ma_long = data['close'].rolling(20).mean()
        trend_strength = (ma_short.iloc[-1] / ma_long.iloc[-1] - 1)
        
        # Combine indicators
        return 0.5 * abs(slope) + 0.5 * abs(trend_strength)
    
    def _calculate_volatility_level(self, data: pd.DataFrame) -> float:
        """Calculate volatility level using multiple timeframes"""
        returns = data['close'].pct_change()
        
        # Historical volatility
        hist_vol = returns.std() * np.sqrt(252)  # Annualized
        
        # Parkinson volatility (using high-low range)
        high_low_vol = np.sqrt(
            (np.log(data['high'] / data['low']) ** 2).mean() * 252 / (4 * np.log(2))
        )
        
        return 0.5 * hist_vol + 0.5 * high_low_vol
    
    def _calculate_momentum_strength(self, data: pd.DataFrame) -> float:
        """Calculate momentum strength using multiple indicators"""
        # RSI
        rsi = self._calculate_rsi(data['close'])
        rsi_strength = abs(rsi.iloc[-1] - 50) / 50
        
        # Rate of change
        roc = (data['close'].iloc[-1] / data['close'].iloc[0] - 1)
        
        # Combine indicators
        return 0.5 * rsi_strength + 0.5 * abs(roc)
    
    def _analyze_volume_profile(self, data: pd.DataFrame) -> Dict:
        """Analyze volume profile characteristics"""
        volume = data['volume']
        price = data['close']
        
        return {
            'volume_trend': (volume.iloc[-1] / volume.mean() - 1),
            'volume_volatility': volume.std() / volume.mean(),
            'price_volume_correlation': price.corr(volume)
        }
    
    def _analyze_regime_transitions(self, regimes: np.ndarray) -> Dict:
        """Analyze regime transition patterns"""
        transitions = {}
        
        for i in range(len(regimes) - 1):
            from_regime = regimes[i]
            to_regime = regimes[i + 1]
            
            key = f'{from_regime}_to_{to_regime}'
            transitions[key] = transitions.get(key, 0) + 1
        
        # Calculate transition probabilities
        total_transitions = sum(transitions.values())
        transition_probs = {
            k: v / total_transitions for k, v in transitions.items()
        }
        
        return {
            'transition_counts': transitions,
            'transition_probabilities': transition_probs
        }
    
    def _calculate_regime_stability(self, regimes: np.ndarray) -> float:
        """Calculate regime stability metric"""
        regime_changes = np.sum(np.diff(regimes) != 0)
        stability = 1 - (regime_changes / (len(regimes) - 1))
        return stability
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def get_trading_parameters(self, regime: Dict) -> Dict:
        """Get optimal trading parameters for current regime"""
        regime_label = regime['current_regime']['regime_label']
        characteristics = regime['current_regime']['characteristics']
        
        # Adjust parameters based on regime characteristics
        if characteristics['trend_strength'] > 0.7:
            # Strong trend regime
            params = {
                'stop_loss': 0.02,
                'take_profit': 0.04,
                'leverage': 2.0,
                'position_size': 0.1
            }
        elif characteristics['volatility_level'] > 0.5:
            # High volatility regime
            params = {
                'stop_loss': 0.015,
                'take_profit': 0.03,
                'leverage': 1.5,
                'position_size': 0.05
            }
        else:
            # Range-bound regime
            params = {
                'stop_loss': 0.01,
                'take_profit': 0.02,
                'leverage': 1.0,
                'position_size': 0.075
            }
        
        return params</content>