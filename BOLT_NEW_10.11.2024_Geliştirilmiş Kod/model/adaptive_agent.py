import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple
from .reinforcement_optimizer import ReinforcementOptimizer
from .uncertainty_estimator import UncertaintyEstimator
from .risk_manager import RiskManager

class AdaptiveAgent:
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 learning_rate: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Core components
        self.model = self._build_model()
        self.reinforcement_optimizer = ReinforcementOptimizer(
            action_space=action_size,
            state_size=state_size
        )
        self.uncertainty_estimator = UncertaintyEstimator(self.model)
        self.risk_manager = RiskManager()
        
        # Memory for experience replay
        self.memory = []
        self.gamma = 0.95  # Discount factor
        
    def _build_model(self) -> tf.keras.Model:
        """Build neural network model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(self.action_size, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def get_action(self, state: np.ndarray, explore: bool = True) -> Tuple[int, float]:
        """Get action using epsilon-greedy policy"""
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size), 0.0
        
        # Get action probabilities
        action_probs = self.model.predict(state.reshape(1, -1))[0]
        
        # Get uncertainty estimate
        uncertainty = self.uncertainty_estimator.estimate_uncertainty(state)
        
        # Adjust probabilities based on uncertainty
        adjusted_probs = self._adjust_probabilities(action_probs, uncertainty)
        
        # Select action
        action = np.argmax(adjusted_probs)
        confidence = adjusted_probs[action]
        
        return action, confidence
    
    def _adjust_probabilities(self,
                            probs: np.ndarray,
                            uncertainty: float) -> np.ndarray:
        """Adjust action probabilities based on uncertainty"""
        # Reduce confidence in predictions when uncertainty is high
        adjusted_probs = probs * (1 - uncertainty)
        
        # Ensure probabilities sum to 1
        return adjusted_probs / np.sum(adjusted_probs)
    
    def train(self, batch_size: int = 32):
        """Train the agent using experience replay"""
        if len(self.memory) < batch_size:
            return
        
        # Sample random batch from memory
        batch = np.random.choice(self.memory, batch_size, replace=False)
        
        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch])
        
        # Calculate target Q-values
        target = rewards + self.gamma * np.amax(
            self.model.predict(next_states), axis=1
        ) * (1 - dones)
        
        # Get current Q-values and update with target values
        target_f = self.model.predict(states)
        for i, action in enumerate(actions):
            target_f[i][action] = target[i]
        
        # Train the model
        self.model.fit(states, target_f, epochs=1, verbose=0)
        
        # Update epsilon for exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def remember(self,
                 state: np.ndarray,
                 action: int,
                 reward: float,
                 next_state: np.ndarray,
                 done: bool):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
        
        # Limit memory size
        if len(self.memory) > 10000:
            self.memory.pop(0)
    
    def adapt_to_market(self,
                       market_state: Dict,
                       performance_metrics: Dict):
        """Adapt agent's behavior to market conditions"""
        # Update risk parameters
        self.risk_manager.update_risk_params(market_state)
        
        # Adjust learning rate based on performance
        if performance_metrics['recent_profit'] < 0:
            self.learning_rate *= 0.9
        else:
            self.learning_rate *= 1.1
        
        # Update model optimizer
        tf.keras.backend.set_value(
            self.model.optimizer.learning_rate,
            self.learning_rate
        )
        
        # Adjust exploration rate
        self._adjust_exploration(performance_metrics)
    
    def _adjust_exploration(self, performance_metrics: Dict):
        """Adjust exploration rate based on performance"""
        if performance_metrics['win_rate'] < 0.4:
            # Increase exploration when performance is poor
            self.epsilon = min(0.9, self.epsilon * 1.2)
        else:
            # Reduce exploration when performing well
            self.epsilon = max(0.1, self.epsilon * 0.8)