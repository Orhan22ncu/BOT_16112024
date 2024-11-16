import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple
from .multi_agent_system import MultiAgentSystem
from .market_regime_detector import MarketRegimeDetector
from .risk_manager import RiskManager
from .neural_architecture_search import NeuralArchitectureSearch

class AdvancedTradingSystem:
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 config: Dict = None):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config or {}
        
        # Initialize components
        self.multi_agent_system = MultiAgentSystem(state_size, action_size)
        self.regime_detector = MarketRegimeDetector()
        self.risk_manager = RiskManager()
        self.nas = NeuralArchitectureSearch(input_shape=(state_size,))
        
        # Performance tracking
        self.performance_metrics = {
            'total_profit': 0.0,
            'win_rate': 0.0,
            'trades': []
        }
    
    def make_trading_decision(self,
                            state: np.ndarray,
                            market_state: Dict) -> Dict:
        """Make trading decision using ensemble of agents"""
        # Get market regime
        regime = self.regime_detector.detect_regime(market_state)
        
        # Get ensemble action
        action, confidence = self.multi_agent_system.get_ensemble_action(
            state, market_state
        )
        
        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            market_state['capital'],
            market_state['volatility']
        )
        
        # Adjust for market conditions
        position_size = self._adjust_position_size(
            position_size,
            confidence,
            regime
        )
        
        return {
            'action': action,
            'confidence': confidence,
            'position_size': position_size,
            'regime': regime
        }
    
    def _adjust_position_size(self,
                            base_size: float,
                            confidence: float,
                            regime: Dict) -> float:
        """Adjust position size based on confidence and regime"""
        # Reduce size in high volatility regime
        if regime['current_regime']['characteristics']['volatility_level'] > 0.8:
            base_size *= 0.5
        
        # Adjust based on confidence
        base_size *= confidence
        
        return base_size
    
    def update_system(self,
                     market_state: Dict,
                     performance_metrics: Dict):
        """Update and adapt the trading system"""
        # Update performance metrics
        self._update_performance_metrics(performance_metrics)
        
        # Adapt agents to market conditions
        self.multi_agent_system.adapt_to_market(
            market_state,
            performance_metrics
        )
        
        # Update risk parameters
        self.risk_manager.update_risk_params(market_state)
        
        # Trigger architecture search if needed
        if self._should_optimize_architecture():
            self._optimize_architecture()
    
    def _update_performance_metrics(self, metrics: Dict):
        """Update system performance metrics"""
        self.performance_metrics['total_profit'] += metrics['profit']
        self.performance_metrics['trades'].append(metrics)
        
        # Calculate win rate
        trades = self.performance_metrics['trades']
        winning_trades = sum(1 for t in trades if t['profit'] > 0)
        self.performance_metrics['win_rate'] = winning_trades / len(trades)
    
    def _should_optimize_architecture(self) -> bool:
        """Determine if architecture optimization is needed"""
        # Check recent performance
        recent_trades = self.performance_metrics['trades'][-100:]
        if len(recent_trades) < 100:
            return False
        
        recent_profit = sum(t['profit'] for t in recent_trades)
        return recent_profit < 0
    
    def _optimize_architecture(self):
        """Optimize neural architecture"""
        print("\n🔄 Başlatılıyor: Neural Architecture Search")
        
        # Prepare training data
        X = np.array([t['state'] for t in self.performance_metrics['trades']])
        y = np.array([t['profit'] for t in self.performance_metrics['trades']])
        
        # Run architecture search
        new_model, performance = self.nas.search(X, y)
        
        print(f"✨ Yeni model performansı: {performance:.4f}")
        
        # Update agent models if improved
        if performance > self.performance_metrics['win_rate']:
            for agent in self.multi_agent_system.agents.values():
                agent.model = tf.keras.models.clone_model(new_model)
                agent.model.compile(
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=agent.learning_rate
                    ),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
            print("✅ Agent modelleri güncellendi")
    
    def save_system(self, path: str):
        """Save the entire trading system"""
        # Save agents
        self.multi_agent_system.save_agents(f"{path}/agents")
        
        # Save performance metrics
        import json
        with open(f"{path}/metrics.json", 'w') as f:
            json.dump(self.performance_metrics, f)
    
    def load_system(self, path: str):
        """Load the entire trading system"""
        # Load agents
        self.multi_agent_system.load_agents(f"{path}/agents")
        
        # Load performance metrics
        import json
        with open(f"{path}/metrics.json", 'r') as f:
            self.performance_metrics = json.load(f)