import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.stats import norm
from .correlation_analyzer import CorrelationAnalyzer
from .market_regime_detector import MarketRegimeDetector

class RiskManager:
    def __init__(self,
                 max_position_size: float = 0.1,
                 max_drawdown: float = 0.2,
                 correlation_analyzer: CorrelationAnalyzer = None,
                 regime_detector: MarketRegimeDetector = None):
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.correlation_analyzer = correlation_analyzer
        self.regime_detector = regime_detector
        self.positions = []
        self.equity_curve = []
        self.risk_metrics = {}
        
    def calculate_position_size(self,
                              capital: float,
                              volatility: float,
                              risk_per_trade: float = 0.02) -> float:
        """Calculate optimal position size using Kelly Criterion and volatility"""
        # Get Kelly fraction
        kelly_fraction = self.kelly_criterion(self.positions)
        
        # Calculate volatility-adjusted size
        vol_adjusted_size = 1 / (volatility * np.sqrt(252))
        
        # Calculate final position size with multiple constraints
        position_size = min(
            capital * self.max_position_size,
            capital * kelly_fraction,
            capital * vol_adjusted_size,
            capital * risk_per_trade
        )
        
        return position_size
    
    def kelly_criterion(self, trades: List[float], fraction: float = 0.5) -> float:
        """Calculate Kelly Criterion for position sizing"""
        if not trades:
            return 0.1  # Default starting value
        
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
    
    def calculate_risk_metrics(self,
                             btc_data: pd.DataFrame,
                             bch_data: pd.DataFrame) -> Dict:
        """Calculate comprehensive risk metrics"""
        self.risk_metrics = {
            'market_risk': self._calculate_market_risk(bch_data),
            'correlation_risk': self._calculate_correlation_risk(btc_data, bch_data),
            'regime_risk': self._calculate_regime_risk(bch_data),
            'liquidity_risk': self._calculate_liquidity_risk(bch_data),
            'tail_risk': self._calculate_tail_risk(bch_data),
            'portfolio_risk': self._calculate_portfolio_risk()
        }
        
        return self.risk_metrics
    
    def _calculate_market_risk(self, data: pd.DataFrame) -> Dict:
        """Calculate market risk metrics"""
        returns = data['close'].pct_change().dropna()
        
        var_95 = self.calculate_var(returns, confidence=0.95)
        var_99 = self.calculate_var(returns, confidence=0.99)
        cvar_95 = self.calculate_cvar(returns, confidence=0.95)
        
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'volatility': volatility,
            'annualized_volatility': volatility * np.sqrt(252)
        }
    
    def _calculate_correlation_risk(self,
                                  btc_data: pd.DataFrame,
                                  bch_data: pd.DataFrame) -> Dict:
        """Calculate correlation-based risk metrics"""
        if self.correlation_analyzer:
            correlations = self.correlation_analyzer.analyze(btc_data, bch_data)
            
            return {
                'price_correlation': correlations['price']['rolling'],
                'volume_correlation': correlations['volume']['rolling'],
                'correlation_risk_score': self._calculate_correlation_risk_score(correlations)
            }
        
        return {}
    
    def _calculate_regime_risk(self, data: pd.DataFrame) -> Dict:
        """Calculate regime-based risk metrics"""
        if self.regime_detector:
            regime = self.regime_detector.detect_regime(data)
            
            return {
                'current_regime': regime['current_regime']['regime_label'],
                'regime_volatility': regime['current_regime']['characteristics']['volatility_level'],
                'regime_stability': regime['regime_stability']
            }
        
        return {}
    
    def _calculate_liquidity_risk(self, data: pd.DataFrame) -> Dict:
        """Calculate liquidity risk metrics"""
        volume = data['volume']
        price = data['close']
        
        avg_daily_volume = volume.rolling(20).mean().iloc[-1]
        volume_volatility = volume.pct_change().std()
        
        return {
            'average_daily_volume': avg_daily_volume,
            'volume_volatility': volume_volatility,
            'amihud_illiquidity': abs(price.pct_change() / volume).mean()
        }
    
    def _calculate_tail_risk(self, data: pd.DataFrame) -> Dict:
        """Calculate tail risk metrics"""
        returns = data['close'].pct_change().dropna()
        
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'tail_risk_score': abs(skewness) + (kurtosis - 3) / 2
        }
    
    def _calculate_portfolio_risk(self) -> Dict:
        """Calculate portfolio-level risk metrics"""
        if not self.equity_curve:
            return {}
        
        equity_array = np.array(self.equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        
        return {
            'max_drawdown': self.calculate_max_drawdown(returns),
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'sortino_ratio': self.calculate_sortino_ratio(returns)
        }
    
    def _calculate_correlation_risk_score(self, correlations: Dict) -> float:
        """Calculate correlation risk score"""
        weights = {
            'price': 0.4,
            'volume': 0.3,
            'volatility': 0.3
        }
        
        risk_score = 0
        for metric, weight in weights.items():
            if metric in correlations:
                correlation = abs(correlations[metric]['rolling'])
                risk_score += correlation * weight
        
        return risk_score
    
    def should_trade(self, current_drawdown: float) -> bool:
        """Determine if trading should continue based on risk metrics"""
        if current_drawdown >= self.max_drawdown:
            return False
        
        # Check risk thresholds
        if self.risk_metrics:
            market_risk = self.risk_metrics['market_risk']
            if market_risk['volatility'] > 0.5:  # 50% annualized volatility
                return False
            
            if 'regime_risk' in self.risk_metrics:
                regime_risk = self.risk_metrics['regime_risk']
                if regime_risk['regime_volatility'] > 0.8:
                    return False
        
        return True
    
    @staticmethod
    def calculate_var(returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns, (1 - confidence) * 100)
    
    @staticmethod
    def calculate_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk"""
        var = np.percentile(returns, (1 - confidence) * 100)
        return np.mean(returns[returns <= var])
    
    @staticmethod
    def calculate_max_drawdown(returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative / running_max - 1
        return abs(min(drawdowns))
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * np.mean(excess_returns) / np.std(returns)
    
    @staticmethod
    def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-6
        return np.sqrt(252) * np.mean(excess_returns) / downside_std