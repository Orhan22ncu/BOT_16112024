import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

class MarketRegimeDetector:
    def __init__(self, n_regimes=3):
        self.n_regimes = n_regimes
        self.gmm = GaussianMixture(n_components=n_regimes, random_state=42)
        self.scaler = StandardScaler()
        
    def detect_regime(self, features):
        """
        Piyasa rejimini tespit et:
        0: Düşük volatilite / Trend
        1: Yüksek volatilite / Range
        2: Kriz / Aşırı volatilite
        """
        scaled_features = self.scaler.fit_transform(features)
        regime = self.gmm.predict(scaled_features)
        regime_probs = self.gmm.predict_proba(scaled_features)
        return regime, regime_probs
    
    def adjust_parameters(self, regime):
        """Rejime göre model parametrelerini ayarla"""
        params = {
            0: {  # Trend rejimi
                'stop_loss': 0.02,
                'take_profit': 0.04,
                'leverage': 1.0
            },
            1: {  # Range rejimi
                'stop_loss': 0.015,
                'take_profit': 0.03,
                'leverage': 0.75
            },
            2: {  # Kriz rejimi
                'stop_loss': 0.01,
                'take_profit': 0.02,
                'leverage': 0.5
            }
        }
        return params[regime]