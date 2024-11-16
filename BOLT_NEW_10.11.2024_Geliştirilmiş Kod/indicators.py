import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from scipy.stats import linregress

class AdvancedIndicators:
    @staticmethod
    def calculate_ema(data, period):
        return pd.Series(data).ewm(span=period, adjust=False).mean().values
    
    @staticmethod
    def calculate_sma(data, period):
        return pd.Series(data).rolling(window=period).mean().values
    
    @staticmethod
    def calculate_bollinger_bands(data, period=20, std_dev=2):
        sma = AdvancedIndicators.calculate_sma(data, period)
        rolling_std = pd.Series(data).rolling(window=period).std()
        upper_band = sma + (rolling_std * std_dev)
        lower_band = sma - (rolling_std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def calculate_rsi(data, period=14):
        delta = pd.Series(data).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_stochastic(data, high, low, period=14, k_period=3, d_period=3):
        lowest_low = pd.Series(low).rolling(window=period).min()
        highest_high = pd.Series(high).rolling(window=period).max()
        
        k = 100 * ((data - lowest_low) / (highest_high - lowest_low))
        k = k.rolling(window=k_period).mean()
        d = k.rolling(window=d_period).mean()
        return k, d
    
    @staticmethod
    def calculate_adx(high, low, close, period=14):
        plus_dm = high.diff()
        minus_dm = low.diff()
        tr = pd.DataFrame({
            'h-l': high - low,
            'h-pc': abs(high - close.shift(1)),
            'l-pc': abs(low - close.shift(1))
        }).max(axis=1)
        
        plus_di = 100 * (plus_dm.rolling(period).mean() / tr.rolling(period).mean())
        minus_di = 100 * (minus_dm.rolling(period).mean() / tr.rolling(period).mean())
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        return adx, plus_di, minus_di

class PatternRecognition:
    @staticmethod
    def find_support_resistance(data, order=5):
        highs = argrelextrema(data, np.greater, order=order)[0]
        lows = argrelextrema(data, np.less, order=order)[0]
        
        resistance_levels = data[highs]
        support_levels = data[lows]
        
        return support_levels, resistance_levels
    
    @staticmethod
    def detect_trend(data, period=14):
        slope = linregress(np.arange(len(data[-period:])), data[-period:])[0]
        return slope
    
    @staticmethod
    def detect_divergence(price, indicator, window=20):
        price_highs = argrelextrema(price[-window:], np.greater, order=3)[0]
        price_lows = argrelextrema(price[-window:], np.less, order=3)[0]
        ind_highs = argrelextrema(indicator[-window:], np.greater, order=3)[0]
        ind_lows = argrelextrema(indicator[-window:], np.less, order=3)[0]
        
        bullish_div = False
        bearish_div = False
        
        if len(price_lows) >= 2 and len(ind_lows) >= 2:
            if price[-price_lows[-1]] < price[-price_lows[-2]] and \
               indicator[-ind_lows[-1]] > indicator[-ind_lows[-2]]:
                bullish_div = True
        
        if len(price_highs) >= 2 and len(ind_highs) >= 2:
            if price[-price_highs[-1]] > price[-price_highs[-2]] and \
               indicator[-ind_highs[-1]] < indicator[-ind_highs[-2]]:
                bearish_div = True
        
        return bullish_div, bearish_div
    
    @staticmethod
    def detect_candlestick_patterns(open_prices, high, low, close):
        patterns = {}
        
        # Doji
        body = abs(close - open_prices)
        wick_up = high - np.maximum(open_prices, close)
        wick_down = np.minimum(open_prices, close) - low
        patterns['doji'] = body <= (wick_up + wick_down) * 0.1
        
        # Hammer
        patterns['hammer'] = (wick_down > body * 2) & (wick_up < body * 0.5)
        
        # Shooting Star
        patterns['shooting_star'] = (wick_up > body * 2) & (wick_down < body * 0.5)
        
        return patterns