import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.optimize import minimize
from sklearn.model_selection import TimeSeriesSplit
from .market_regime_detector import MarketRegimeDetector
from .correlation_analyzer import CorrelationAnalyzer

class StrategyOptimizer:
    def __init__(self,
                 base_params: Dict,
                 regime_detector: MarketRegimeDetector,
                 correlation_analyzer: CorrelationAnalyzer):
        self.base_params = base_params
        self.regime_detector = regime_detector
        self.correlation_analyzer = correlation_analyzer
        self.optimal_params = {}
        self.performance_history = []
        
    def optimize_strategy(self,
                        btc_data: pd.DataFrame,
                        bch_data: pd.DataFrame,
                        objective: str = 'sharpe') -> Dict:
        """Optimize trading strategy parameters"""
        # Detect market regime
        regime = self.regime_detector.detect_regime(bch_data)
        
        # Analyze correlations
        correlations = self.correlation_analyzer.analyze(btc_data, bch_data)
        
        # Optimize parameters based on regime and correlations
        optimal_params = self._optimize_parameters(
            btc_data,
            bch_data,
            regime,
            correlations,
            objective
        )
        
        # Validate parameters
        validation_results = self._validate_parameters(
            optimal_params,
            btc_data,
            bch_data
        )
        
        self.optimal_params = optimal_params
        return {
            'optimal_params': optimal_params,
            'validation_results': validation_results,
            'regime_info': regime,
            'correlation_info': correlations
        }
    
    def _optimize_parameters(self,
                           btc_data: pd.DataFrame,
                           bch_data: pd.DataFrame,
                           regime: Dict,
                           correlations: Dict,
                           objective: str) -> Dict:
        """Optimize strategy parameters"""
        param_bounds = self._get_parameter_bounds(regime, correlations)
        
        def objective_function(params):
            return -self._calculate_objective(
                params,
                btc_data,
                bch_data,
                objective
            )
        
        # Optimize using scipy
        result = minimize(
            objective_function,
            x0=list(self.base_params.values()),
            bounds=param_bounds,
            method='SLSQP'
        )
        
        optimized_params = dict(zip(self.base_params.keys(), result.x))
        return self._adjust_parameters(optimized_params, regime, correlations)
    
    def _get_parameter_bounds(self,
                            regime: Dict,
                            correlations: Dict) -> List[Tuple]:
        """Get parameter bounds based on regime and correlations"""
        correlation_strength = correlations.get('price', {}).get('rolling', 0)
        regime_volatility = regime['current_regime']['characteristics']['volatility_level']
        
        bounds = {
            'entry_threshold': (0.001, 0.01),
            'exit_threshold': (0.002, 0.02),
            'stop_loss': (0.01, 0.05),
            'take_profit': (0.02, 0.10),
            'position_size': (0.1, 0.5),
            'leverage': (1.0, 5.0)
        }
        
        # Adjust bounds based on regime and correlations
        if regime_volatility > 0.8:  # High volatility
            bounds['stop_loss'] = (0.02, 0.08)
            bounds['take_profit'] = (0.04, 0.15)
            bounds['leverage'] = (1.0, 3.0)
        
        if correlation_strength > 0.8:  # Strong correlation
            bounds['position_size'] = (0.2, 0.6)
            bounds['leverage'] = (1.5, 5.0)
        
        return list(bounds.values())
    
    def _calculate_objective(self,
                           params: np.ndarray,
                           btc_data: pd.DataFrame,
                           bch_data: pd.DataFrame,
                           objective: str) -> float:
        """Calculate objective function value"""
        param_dict = dict(zip(self.base_params.keys(), params))
        returns = self._simulate_strategy(param_dict, btc_data, bch_data)
        
        if objective == 'sharpe':
            return self._calculate_sharpe_ratio(returns)
        elif objective == 'sortino':
            return self._calculate_sortino_ratio(returns)
        elif objective == 'calmar':
            return self._calculate_calmar_ratio(returns)
        else:
            return np.mean(returns)
    
    def _simulate_strategy(self,
                         params: Dict,
                         btc_data: pd.DataFrame,
                         bch_data: pd.DataFrame) -> np.ndarray:
        """Simulate trading strategy with given parameters"""
        returns = []
        position = 0
        entry_price = 0
        
        for i in range(1, len(bch_data)):
            if position == 0:  # No position
                if self._check_entry_signal(params, btc_data.iloc[i], bch_data.iloc[i]):
                    position = 1
                    entry_price = bch_data.iloc[i]['close']
            else:  # In position
                if self._check_exit_signal(params, btc_data.iloc[i], bch_data.iloc[i], entry_price):
                    returns.append(
                        (bch_data.iloc[i]['close'] - entry_price) / entry_price * params['leverage']
                    )
                    position = 0
        
        return np.array(returns)
    
    def _check_entry_signal(self,
                          params: Dict,
                          btc_row: pd.Series,
                          bch_row: pd.Series) -> bool:
        """Check entry signal conditions"""
        # Price divergence
        btc_return = btc_row['close'] / btc_row['open'] - 1
        bch_return = bch_row['close'] / bch_row['open'] - 1
        
        # Volume confirmation
        volume_ratio = bch_row['volume'] / bch_row['volume'].rolling(20).mean()
        
        return (abs(btc_return - bch_return) > params['entry_threshold'] and
                volume_ratio > 1.2)
    
    def _check_exit_signal(self,
                          params: Dict,
                          btc_row: pd.Series,
                          bch_row: pd.Series,
                          entry_price: float) -> bool:
        """Check exit signal conditions"""
        current_price = bch_row['close']
        price_change = (current_price - entry_price) / entry_price
        
        # Stop loss
        if price_change < -params['stop_loss']:
            return True
        
        # Take profit
        if price_change > params['take_profit']:
            return True
        
        # Technical exit
        btc_return = btc_row['close'] / btc_row['open'] - 1
        bch_return = bch_row['close'] / bch_row['open'] - 1
        
        return abs(btc_return - bch_return) < params['exit_threshold']
    
    def _adjust_parameters(self,
                         params: Dict,
                         regime: Dict,
                         correlations: Dict) -> Dict:
        """Fine-tune parameters based on market conditions"""
        regime_type = regime['current_regime']['regime_label']
        correlation_strength = correlations.get('price', {}).get('rolling', 0)
        
        adjusted_params = params.copy()
        
        # Adjust based on regime
        if regime_type == 'high_volatility':
            adjusted_params['stop_loss'] *= 1.2
            adjusted_params['take_profit'] *= 1.2
            adjusted_params['leverage'] *= 0.8
        elif regime_type == 'low_volatility':
            adjusted_params['entry_threshold'] *= 0.8
            adjusted_params['exit_threshold'] *= 0.8
        
        # Adjust based on correlation
        if correlation_strength > 0.8:
            adjusted_params['position_size'] *= 1.2
            adjusted_params['leverage'] *= 1.2
        elif correlation_strength < 0.5:
            adjusted_params['position_size'] *= 0.8
            adjusted_params['leverage'] *= 0.8
        
        return adjusted_params
    
    def _validate_parameters(self,
                           params: Dict,
                           btc_data: pd.DataFrame,
                           bch_data: pd.DataFrame) -> Dict:
        """Validate optimized parameters"""
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        validation_results = []
        
        for train_idx, test_idx in tscv.split(bch_data):
            # Simulate strategy on test set
            btc_test = btc_data.iloc[test_idx]
            bch_test = bch_data.iloc[test_idx]
            
            returns = self._simulate_strategy(params, btc_test, bch_test)
            
            validation_results.append({
                'sharpe': self._calculate_sharpe_ratio(returns),
                'sortino': self._calculate_sortino_ratio(returns),
                'max_drawdown': self._calculate_max_drawdown(returns),
                'win_rate': self._calculate_win_rate(returns)
            })
        
        return {
            'mean_metrics': pd.DataFrame(validation_results).mean().to_dict(),
            'std_metrics': pd.DataFrame(validation_results).std().to_dict()
        }
    
    @staticmethod
    def _calculate_sharpe_ratio(returns: np.ndarray) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0:
            return 0
        return np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
    
    @staticmethod
    def _calculate_sortino_ratio(returns: np.ndarray) -> float:
        """Calculate Sortino ratio"""
        if len(returns) == 0:
            return 0
        negative_returns = returns[returns < 0]
        downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 1e-6
        return np.mean(returns) / downside_std * np.sqrt(252)
    
    @staticmethod
    def _calculate_calmar_ratio(returns: np.ndarray) -> float:
        """Calculate Calmar ratio"""
        if len(returns) == 0:
            return 0
        max_drawdown = StrategyOptimizer._calculate_max_drawdown(returns)
        return np.mean(returns) / (max_drawdown + 1e-6) * 252
    
    @staticmethod
    def _calculate_max_drawdown(returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(returns) == 0:
            return 0
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative / running_max - 1
        return abs(min(drawdowns))
    
    @staticmethod
    def _calculate_win_rate(returns: np.ndarray) -> float:
        """Calculate win rate"""
        if len(returns) == 0:
            return 0
        return np.mean(returns > 0)</content>