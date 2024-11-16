import numpy as np
from collections import deque
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam

class TradingEnvironment:
    def __init__(self, data, initial_balance=10000):
        self.data = data
        self.initial_balance = initial_balance
        self.reset()
        
    def reset(self):
        self.balance = self.initial_balance
        self.position = 0  # 0: nakit, 1: long pozisyon
        self.current_step = 0
        self.trades = []
        return self._get_state()
    
    def _get_state(self):
        return np.array(self.data[self.current_step])
    
    def step(self, action):
        current_price = self.data[self.current_step][-1]  # Son kapanış fiyatı
        done = self.current_step >= len(self.data) - 1
        
        # Aksiyonları uygula (0: Bekle, 1: Al, 2: Sat)
        reward = 0
        
        if action == 1 and self.position == 0:  # Al
            self.position = 1
            self.entry_price = current_price
            self.trades.append(('BUY', current_price))
            reward = -0.1  # İşlem maliyeti
            
        elif action == 2 and self.position == 1:  # Sat
            profit = (current_price - self.entry_price) / self.entry_price
            reward = profit * 100  # Yüzdelik kar/zarar
            self.position = 0
            self.trades.append(('SELL', current_price))
            self.balance *= (1 + profit)
        
        # Pozisyondayken fiyat değişimlerini takip et
        elif self.position == 1:
            profit = (current_price - self.entry_price) / self.entry_price
            reward = profit * 10  # Pozisyon tutma ödülü/cezası
        
        self.current_step += 1
        next_state = self._get_state() if not done else None
        
        return next_state, reward, done

class DQLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Gelecek ödül discount faktörü
        self.epsilon = 1.0  # Keşif oranı
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_counter = 0
        
    def _build_model(self):
        model = Sequential([
            LSTM(64, input_shape=(self.state_size, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        if training and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = np.reshape(state, [1, self.state_size, 1])
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states = []
        targets = []
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = np.reshape(next_state, [1, self.state_size, 1])
                target = reward + self.gamma * np.amax(
                    self.target_model.predict(next_state, verbose=0)[0]
                )
            
            state = np.reshape(state, [1, self.state_size, 1])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            
            states.append(state[0])
            targets.append(target_f[0])
        
        self.model.fit(
            np.array(states),
            np.array(targets),
            epochs=1,
            verbose=0
        )
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def save(self, filepath):
        self.model.save(filepath)
    
    def load(self, filepath):
        self.model.load_weights(filepath)