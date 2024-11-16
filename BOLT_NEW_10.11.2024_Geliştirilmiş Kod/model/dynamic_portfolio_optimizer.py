import numpy as np
from typing import Dict, List, Tuple
from scipy.optimize import minimize
import pandas as pd

class DynamicPortfolioOptimizer:
    def __init__(self,
                 risk_free_rate: float = 0.02,
                 target_volatility: float = 0.15):
        self.risk_free_rate = risk_free_rate
        self.target_volatility = target_volatility
        self.weights_history = []
        
    def optimize_portfolio(self,
                         returns: np.ndarray,
                         covariance: np.ndarray,
                         constraints: Dict = None) -> Dict:
        """Optimize portfolio weights using mean-variance optimization"""
        n_assets = len(returns)
        
        # Define optimization constraints
        bounds = [(0, 1) for _ in range(n_assets)]
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
        ]
        
        # Initial guess (equal weights)
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            self._portfolio_objective,
            initial_weights,
            args=(returns, covariance),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_weights = result.x
        self.weights_history.append(optimal_weights)
        
        # Calculate portfolio metrics
        metrics = self._calculate_portfolio_metrics(
            optimal_weights,
            returns,
            covariance
        )
        
        return {
            'weights': optimal_weights,
            'metrics': metrics
        }
    
    def _portfolio_objective(self,
                           weights: np.ndarray,
                           returns: np.ndarray,
                           covariance: np.ndarray) -> float:
        """Portfolio optimization objective function"""
        portfolio_return = np.sum(returns * weights)
        portfolio_volatility = np.sqrt(
            np.dot(weights.T, np.dot(covariance, weights))
        )
        
        # Sharpe ratio (negative for minimization)
        sharpe = -(portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        # Add penalty for deviation from target volatility
        volatility_penalty = abs(portfolio_volatility - self.target_volatility)
        
        return sharpe + volatility_penalty
    
    def _calculate_portfolio_metrics(self,
                                   weights: np.ndarray,
                                   returns: np.ndarray,
                                   covariance: np.ndarray) -> Dict:
        """Calculate portfolio performance metrics"""
        portfolio_return = np.sum(returns * weights)
        portfolio_volatility = np.sqrt(
            np.dot(weights.T, np.dot(covariance, weights))
        )
        
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        # Calculate diversification ratio
        weighted_volatilities = np.sqrt(np.diag(covariance)) * weights
        diversification_ratio = portfolio_volatility / np.sum(weighted_volatilities)
        
        return {
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'diversification_ratio': diversification_ratio
        }
    
    def rebalance_portfolio(self,
                           current_weights: np.ndarray,
                           target_weights: np.ndarray,
                           transaction_costs: float = 0.001) -> Dict:
        """Calculate optimal portfolio rebalancing"""
        # Calculate required trades
        trades = target_weights - current_weights
        
        # Calculate transaction costs
        costs = np.sum(np.abs(trades)) * transaction_costs
        
        # Check if rebalancing is worth it
        if self._should_rebalance(current_weights, target_weights, costs):
            return {
                'execute_rebalance': True,
                'trades': trades,
                'costs': costs
            }
        
        return {
            'execute_rebalance': False,
            'trades': np.zeros_like(trades),
            'costs': 0
        }
    
    def _should_rebalance(self,
                         current: np.ndarray,
                         target: np.ndarray,
                         costs: float) -> bool:
        """Determine if rebalancing is beneficial"""
        # Calculate tracking error
        tracking_error = np.sqrt(np.sum((current - target) ** 2))
        
        # Simple threshold-based decision
        return tracking_error > costs * 2  # Rebalance if tracking error > 2x costs
    
    def get_portfolio_analysis(self) -> Dict:
        """Analyze portfolio weight history"""
        weights_history = np.array(self.weights_history)
        
        return {
            'turnover': self._calculate_turnover(weights_history),
            'concentration': self._calculate_concentration(weights_history[-1]),
            'stability': self._calculate_stability(weights_history)
        }
    
    def _calculate_turnover(self, weights_history: np.ndarray) -> float:
        """Calculate portfolio turnover"""
        if len(weights_history) < 2:
            return 0.0
            
        changes = np.abs(weights_history[1:] - weights_history[:-1])
        return np.mean(np.sum(changes, axis=1))
    
    def _calculate_concentration(self, weights: np.ndarray) -> float:
        """Calculate portfolio concentration (Herfindahl index)"""
        return np.sum(weights ** 2)
    
    def _calculate_stability(self, weights_history: np.ndarray) -> float:
        """Calculate portfolio stability"""
        if len(weights_history) < 2:
            return 1.0
            
        weight_changes = np.std(weights_history, axis=0)
        return 1 - np.mean(weight_changes)