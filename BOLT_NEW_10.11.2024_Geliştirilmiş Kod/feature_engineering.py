import numpy as np
import pandas as pd
from indicators import AdvancedIndicators, PatternRecognition

class FeatureEngineer:
    def __init__(self):
        self.indicators = AdvancedIndicators()
        self.patterns = PatternRecognition()
        
    def create_features(self, df):
        features = pd.DataFrame()
        
        # Fiyat özellikleri
        features['close'] = df['close']
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log1p(features['returns'])
        
        # Teknik göstergeler
        features['ema_9'] = self.indicators.calculate_ema(df['close'], 9)
        features['ema_21'] = self.indicators.calculate_ema(df['close'], 21)
        features['rsi'] = self.indicators.calculate_rsi(df['close'])
        
        # Bollinger Bands
        upper, middle, lower = self.indicators.calculate_bollinger_bands(df['close'])
        features['bb_upper'] = upper
        features['bb_middle'] = middle
        features['bb_lower'] = lower
        features['bb_width'] = (upper - lower) / middle
        
        # Stochastic
        k, d = self.indicators.calculate_stochastic(df['close'], df['high'], df['low'])
        features['stoch_k'] = k
        features['stoch_d'] = d
        
        # ADX
        adx, plus_di, minus_di = self.indicators.calculate_adx(
            df['high'], df['low'], df['close']
        )
        features['adx'] = adx
        features['plus_di'] = plus_di
        features['minus_di'] = minus_di
        
        # Volatilite özellikleri
        features['volatility'] = features['returns'].rolling(window=20).std()
        features['volatility_ratio'] = features['volatility'] / \
                                     features['volatility'].rolling(window=100).mean()
        
        # Örüntü tanıma özellikleri
        support, resistance = self.patterns.find_support_resistance(df['close'].values)
        features['distance_to_support'] = df['close'] - support.min()
        features['distance_to_resistance'] = resistance.max() - df['close']
        
        # Trend özellikleri
        features['trend'] = features['close'].rolling(window=20).apply(
            lambda x: self.patterns.detect_trend(x)
        )
        
        # Mum çubukları örüntüleri
        candlestick_patterns = self.patterns.detect_candlestick_patterns(
            df['open'], df['high'], df['low'], df['close']
        )
        for pattern, values in candlestick_patterns.items():
            features[f'pattern_{pattern}'] = values.astype(int)
        
        # NaN değerleri temizle
        features.dropna(inplace=True)
        
        return features