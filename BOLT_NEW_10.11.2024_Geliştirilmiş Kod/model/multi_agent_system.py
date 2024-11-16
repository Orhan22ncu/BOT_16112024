import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple
from .adaptive_agent import AdaptiveAgent
from .market_regime_detector import MarketRegimeDetector

class MultiAgentSystem:
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 n_agents: int = 3):
        self.state_size = state_size
        self.action_size = action_size
        self.n_agents = n_agents
        
        # Initialize specialized agents
        self.agents = {
            'trend_following': AdaptiveAgent(state_size, action_size),
            'mean_reversion': AdaptiveAgent(state_size, action_size),
            'volatility_trading': AdaptiveAgent(state_size, action_size)
        }
        
        # Market regime detector
        self.regime_detector = MarketRegimeDetector()
        
        # Agent weights
        self.agent_weights = {agent: 1.0/n_agents for agent in self.agents}
    
    def get_ensemble_action(self,
                          state: np.ndarray,
                          market_state: Dict) -> Tuple[int, float]:
        """Get weighted ensemble action from all agents"""
        # Detect current market regime
        regime = self.regime_detector.detect_regime(market_state)
        
        # Get actions from all agents
        agent_actions = {}
        for name, agent in self.agents.items():
            action, confidence = agent.get_action(state)
            agent_actions[name] = (action, confidence)
        
        # Adjust weights based on regime
        self._adjust_weights(regime)
        
        # Weighted voting
        action_votes = np.zeros(self.action_size)
        for name, (action, confidence) in agent_actions.items():
            action_votes[action] += confidence * self.agent_weights[name]
        
        # Select final action
        final_action = np.argmax(action_votes)
        confidence = action_votes[final_action]
        
        return final_action, confidence
    
    def _adjust_weights(self, regime: Dict):
        """Adjust agent weights based on market regime"""
        regime_type = regime['current_regime']['regime_label']
        
        if regime_type == 'trending':
            self.agent_weights.update({
                'trend_following': 0.5,
                'mean_reversion': 0.2,
                'volatility_trading': 0.3
            })
        elif regime_type == 'mean_reverting':
            self.agent_weights.update({
                'trend_following': 0.2,
                'mean_reversion': 0.5,
                'volatility_trading': 0.3
            })
        elif regime_type == 'high_volatility':
            self.agent_weights.update({
                'trend_following': 0.3,
                'mean_reversion': 0.2,
                'volatility_trading': 0.5
            })
    
    def train_agents(self, batch_size: int = 32):
        """Train all agents"""
        for agent in self.agents.values():
            agent.train(batch_size)
    
    def adapt_to_market(self,
                       market_state: Dict,
                       performance_metrics: Dict):
        """Adapt all agents to market conditions"""
        for agent in self.agents.values():
            agent.adapt_to_market(market_state, performance_metrics)
    
    def save_agents(self, path: str):
        """Save all agent models"""
        for name, agent in self.agents.items():
            agent.model.save(f"{path}/{name}_model")
    
    def load_agents(self, path: str):
        """Load all agent models"""
        for name in self.agents.keys():
            self.agents[name].model = tf.keras.models.load_model(
                f"{path}/{name}_model"
            )