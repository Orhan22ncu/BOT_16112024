import numpy as np
import pandas as pd
from scipy.stats import norm

class RiskManager:
    def __init__(self, max_position_size=0.1, max_drawdown=0.2):
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.positions = []
        self.equity_curve = []
    
    def calculate_position_size(self, capital, volatility, risk_per_trade=0.02):
        """Kelly Criterion ve volatilite bazlı pozisyon boyutu hesaplama"""
        kelly_fraction = self.kelly_criterion(self.positions)
        vol_adjusted_size = 1 / (volatility * np.sqrt(252))  # Yıllık volatilite
        position_size = min(
            capital * self.max_position_size,
            capital * kelly_fraction,
            capital * vol_adjusted_size,
            capital * risk_per_trade
        )
        return position_size
    
    def kelly_criterion(self, trades, fraction=0.5):
        if not trades:
            return 0.1  # Başlangıç değeri
        
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t < 0]
        
        if not losses:
            return self.max_position_size
        
        win_rate = len(wins) / len(trades)
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0
        
        if avg_loss == 0:
            return self.max_position_size
        
        kelly = win_rate - ((1 - win_rate) / (avg_win / avg_loss))
        return max(0, min(self.max_position_size, kelly * fraction))
    
    def calculate_var(self, returns, confidence=0.95):
        """Value at Risk hesaplama"""
        return np.percentile(returns, (1 - confidence) * 100)
    
    def calculate_cvar(self, returns, confidence=0.95):
        """Conditional Value at Risk hesaplama"""
        var = self.calculate_var(returns, confidence)
        return np.mean(returns[returns <= var])
    
    def should_trade(self, current_drawdown):
        """Drawdown kontrolü"""
        return current_drawdown < self.max_drawdown