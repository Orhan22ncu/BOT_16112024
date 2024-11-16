import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from .market_data_collector import MarketDataCollector
from .dual_market_predictor import DualMarketPredictor
from .correlation_analyzer import CorrelationAnalyzer
from .market_regime_detector import MarketRegimeDetector
from .strategy_optimizer import StrategyOptimizer
from .risk_manager import RiskManager
from binance.client import Client
import logging

class TradingEngine:
    def __init__(self,
                 api_key: str,
                 api_secret: str,
                 base_symbol: str = 'BTCUSDT',
                 target_symbol: str = 'BCHUSDT',
                 initial_balance: float = 10000):
        self.client = Client(api_key, api_secret)
        self.base_symbol = base_symbol
        self.target_symbol = target_symbol
        self.initial_balance = initial_balance
        
        # Initialize components
        self.data_collector = MarketDataCollector(api_key, api_secret)
        self.correlation_analyzer = CorrelationAnalyzer()
        self.regime_detector = MarketRegimeDetector()
        self.predictor = DualMarketPredictor()
        self.risk_manager = RiskManager(
            correlation_analyzer=self.correlation_analyzer,
            regime_detector=self.regime_detector
        )
        self.strategy_optimizer = StrategyOptimizer(
            base_params={
                'entry_threshold': 0.005,
                'exit_threshold': 0.003,
                'stop_loss': 0.02,
                'take_profit': 0.04,
                'position_size': 0.1,
                'leverage': 2.0
            },
            regime_detector=self.regime_detector,
            correlation_analyzer=self.correlation_analyzer
        )
        
        # Trading state
        self.current_position = None
        self.trades_history = []
        self.performance_metrics = {}
        
    async def start_trading(self):
        """Start trading engine"""
        try:
            # Fetch initial market data
            btc_data, bch_data = self.data_collector.fetch_dual_market_data()
            
            # Initialize analysis
            correlations = self.correlation_analyzer.analyze(btc_data, bch_data)
            regime = self.regime_detector.detect_regime(bch_data)
            
            # Optimize strategy
            strategy_params = self.strategy_optimizer.optimize_strategy(
                btc_data, bch_data
            )
            
            # Calculate risk metrics
            risk_metrics = self.risk_manager.calculate_risk_metrics(
                btc_data, bch_data
            )
            
            while True:
                await self._trading_loop(
                    btc_data,
                    bch_data,
                    strategy_params,
                    risk_metrics
                )
                
        except Exception as e:
            logging.error(f"Trading engine error: {e}")
            raise
    
    async def _trading_loop(self,
                          btc_data: pd.DataFrame,
                          bch_data: pd.DataFrame,
                          strategy_params: Dict,
                          risk_metrics: Dict):
        """Main trading loop"""
        try:
            # Update market data
            btc_data, bch_data = self.data_collector.fetch_dual_market_data()
            
            # Generate predictions
            predictions = self.predictor.predict_dual_markets(
                *self.predictor.prepare_dual_input(btc_data, bch_data)
            )
            
            # Check risk conditions
            if not self.risk_manager.should_trade(self._calculate_drawdown()):
                logging.warning("Risk conditions not met, skipping trade")
                return
            
            # Execute trading logic
            if self.current_position is None:
                await self._check_entry_conditions(
                    btc_data,
                    bch_data,
                    predictions,
                    strategy_params
                )
            else:
                await self._check_exit_conditions(
                    btc_data,
                    bch_data,
                    predictions,
                    strategy_params
                )
            
            # Update performance metrics
            self._update_performance_metrics()
            
        except Exception as e:
            logging.error(f"Trading loop error: {e}")
            raise
    
    async def _check_entry_conditions(self,
                                    btc_data: pd.DataFrame,
                                    bch_data: pd.DataFrame,
                                    predictions: Dict,
                                    strategy_params: Dict):
        """Check and execute entry conditions"""
        # Get latest prices
        btc_price = btc_data['close'].iloc[-1]
        bch_price = bch_data['close'].iloc[-1]
        
        # Calculate entry signals
        price_divergence = abs(
            btc_data['close'].pct_change().iloc[-1] -
            bch_data['close'].pct_change().iloc[-1]
        )
        
        # Check entry conditions
        if (price_divergence > strategy_params['entry_threshold'] and
            predictions['direction_prediction'][-1] > 0.7):
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                self.initial_balance,
                predictions['volatility_prediction'][-1]
            )
            
            # Execute entry
            try:
                order = self.client.futures_create_order(
                    symbol=self.target_symbol,
                    side='BUY',
                    type='MARKET',
                    quantity=position_size
                )
                
                self.current_position = {
                    'entry_price': bch_price,
                    'size': position_size,
                    'entry_time': pd.Timestamp.now(),
                    'order_id': order['orderId']
                }
                
                logging.info(f"Entered position: {self.current_position}")
                
            except Exception as e:
                logging.error(f"Entry execution error: {e}")
                raise
    
    async def _check_exit_conditions(self,
                                   btc_data: pd.DataFrame,
                                   bch_data: pd.DataFrame,
                                   predictions: Dict,
                                   strategy_params: Dict):
        """Check and execute exit conditions"""
        if self.current_position is None:
            return
            
        current_price = bch_data['close'].iloc[-1]
        entry_price = self.current_position['entry_price']
        position_pnl = (current_price - entry_price) / entry_price
        
        # Check exit conditions
        should_exit = (
            position_pnl < -strategy_params['stop_loss'] or
            position_pnl > strategy_params['take_profit'] or
            predictions['direction_prediction'][-1] < 0.3
        )
        
        if should_exit:
            try:
                # Execute exit
                order = self.client.futures_create_order(
                    symbol=self.target_symbol,
                    side='SELL',
                    type='MARKET',
                    quantity=self.current_position['size']
                )
                
                # Record trade
                trade = {
                    'entry_price': self.current_position['entry_price'],
                    'exit_price': current_price,
                    'size': self.current_position['size'],
                    'pnl': position_pnl,
                    'entry_time': self.current_position['entry_time'],
                    'exit_time': pd.Timestamp.now(),
                    'duration': pd.Timestamp.now() - self.current_position['entry_time']
                }
                
                self.trades_history.append(trade)
                self.current_position = None
                
                logging.info(f"Exited position: {trade}")
                
            except Exception as e:
                logging.error(f"Exit execution error: {e}")
                raise
    
    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown"""
        if not self.trades_history:
            return 0.0
            
        equity_curve = self._calculate_equity_curve()
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        return abs(min(drawdown))
    
    def _calculate_equity_curve(self) -> np.ndarray:
        """Calculate equity curve"""
        equity = [self.initial_balance]
        
        for trade in self.trades_history:
            equity.append(
                equity[-1] * (1 + trade['pnl'])
            )
        
        return np.array(equity)
    
    def _update_performance_metrics(self):
        """Update trading performance metrics"""
        if not self.trades_history:
            return
            
        equity_curve = self._calculate_equity_curve()
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        self.performance_metrics = {
            'total_trades': len(self.trades_history),
            'win_rate': np.mean([t['pnl'] > 0 for t in self.trades_history]),
            'average_return': np.mean([t['pnl'] for t in self.trades_history]),
            'sharpe_ratio': self.risk_manager.calculate_sharpe_ratio(returns),
            'max_drawdown': self._calculate_drawdown(),
            'current_balance': equity_curve[-1],
            'total_return': (equity_curve[-1] - self.initial_balance) / self.initial_balance
        }
    
    def get_status(self) -> Dict:
        """Get current trading status"""
        return {
            'current_position': self.current_position,
            'performance_metrics': self.performance_metrics,
            'risk_metrics': self.risk_manager.risk_metrics,
            'trades_count': len(self.trades_history)
        }
    
    async def stop_trading(self):
        """Stop trading and close positions"""
        if self.current_position is not None:
            try:
                # Close position
                order = self.client.futures_create_order(
                    symbol=self.target_symbol,
                    side='SELL',
                    type='MARKET',
                    quantity=self.current_position['size']
                )
                
                logging.info("Closed position on trading stop")
                self.current_position = None
                
            except Exception as e:
                logging.error(f"Error closing position on stop: {e}")
                raise