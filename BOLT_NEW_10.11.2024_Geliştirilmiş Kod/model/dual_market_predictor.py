import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from .correlation_analyzer import CorrelationAnalyzer
from .market_regime_detector import MarketRegimeDetector
from .transformer import TransformerBlock

class DualMarketPredictor:
    def __init__(self,
                 sequence_length: int = 60,
                 n_features: int = 32,
                 correlation_threshold: float = 0.5):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.correlation_threshold = correlation_threshold
        self.correlation_analyzer = CorrelationAnalyzer()
        self.regime_detector = MarketRegimeDetector()
        self.model = self._build_model()
        
    def _build_model(self) -> tf.keras.Model:
        """Build dual-input prediction model"""
        # BTC input branch
        btc_input = tf.keras.layers.Input(shape=(self.sequence_length, self.n_features))
        btc_transformer = TransformerBlock(embed_dim=32, num_heads=4, ff_dim=32)(btc_input)
        btc_lstm = tf.keras.layers.LSTM(64, return_sequences=True)(btc_transformer)
        
        # BCH input branch
        bch_input = tf.keras.layers.Input(shape=(self.sequence_length, self.n_features))
        bch_transformer = TransformerBlock(embed_dim=32, num_heads=4, ff_dim=32)(bch_input)
        bch_lstm = tf.keras.layers.LSTM(64, return_sequences=True)(bch_transformer)
        
        # Correlation attention mechanism
        correlation_attention = tf.keras.layers.Attention()([btc_lstm, bch_lstm])
        
        # Combine features
        combined = tf.keras.layers.Concatenate()([
            tf.keras.layers.GlobalAveragePooling1D()(correlation_attention),
            tf.keras.layers.GlobalAveragePooling1D()(btc_lstm),
            tf.keras.layers.GlobalAveragePooling1D()(bch_lstm)
        ])
        
        # Dense layers
        x = tf.keras.layers.Dense(128, activation='relu')(combined)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # Multiple outputs
        price_pred = tf.keras.layers.Dense(1, name='price_prediction')(x)
        direction_pred = tf.keras.layers.Dense(1, activation='sigmoid', name='direction_prediction')(x)
        volatility_pred = tf.keras.layers.Dense(1, activation='softplus', name='volatility_prediction')(x)
        
        model = tf.keras.Model(
            inputs=[btc_input, bch_input],
            outputs=[price_pred, direction_pred, volatility_pred]
        )
        
        model.compile(
            optimizer='adam',
            loss={
                'price_prediction': 'huber_loss',
                'direction_prediction': 'binary_crossentropy',
                'volatility_prediction': 'mse'
            },
            loss_weights={
                'price_prediction': 1.0,
                'direction_prediction': 0.3,
                'volatility_prediction': 0.3
            }
        )
        
        return model
    
    def prepare_dual_input(self,
                          btc_data: pd.DataFrame,
                          bch_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare dual market input data"""
        # Analyze correlations
        correlations = self.correlation_analyzer.analyze(btc_data, bch_data)
        
        # Detect market regimes
        btc_regime = self.regime_detector.detect_regime(btc_data)
        bch_regime = self.regime_detector.detect_regime(bch_data)
        
        # Create feature matrices
        btc_features = self._create_feature_matrix(btc_data, btc_regime)
        bch_features = self._create_feature_matrix(bch_data, bch_regime)
        
        # Add correlation features
        btc_features = self._add_correlation_features(btc_features, correlations)
        bch_features = self._add_correlation_features(bch_features, correlations)
        
        return btc_features, bch_features
    
    def _create_feature_matrix(self,
                             data: pd.DataFrame,
                             regime: Dict) -> np.ndarray:
        """Create feature matrix for each market"""
        features = []
        
        for i in range(len(data) - self.sequence_length):
            window = data.iloc[i:i+self.sequence_length]
            
            # Price features
            price_features = self._extract_price_features(window)
            
            # Volume features
            volume_features = self._extract_volume_features(window)
            
            # Technical features
            tech_features = self._extract_technical_features(window)
            
            # Regime features
            regime_features = self._extract_regime_features(regime, i)
            
            # Combine all features
            combined = np.concatenate([
                price_features,
                volume_features,
                tech_features,
                regime_features
            ])
            
            features.append(combined)
        
        return np.array(features)
    
    def _extract_price_features(self, window: pd.DataFrame) -> np.ndarray:
        """Extract price-based features"""
        returns = window['close'].pct_change()
        log_returns = np.log1p(returns)
        
        features = [
            returns.values,
            log_returns.values,
            window['high'].div(window['low']).values,
            window['close'].div(window['open']).values
        ]
        
        return np.column_stack(features)
    
    def _extract_volume_features(self, window: pd.DataFrame) -> np.ndarray:
        """Extract volume-based features"""
        volume_changes = window['volume'].pct_change()
        
        features = [
            volume_changes.values,
            window['volume'].div(window['volume'].rolling(10).mean()).values,
            (window['volume'] * window['close']).values  # Volume * Price
        ]
        
        return np.column_stack(features)
    
    def _extract_technical_features(self, window: pd.DataFrame) -> np.ndarray:
        """Extract technical indicators"""
        # Moving averages
        ma_5 = window['close'].rolling(5).mean()
        ma_20 = window['close'].rolling(20).mean()
        
        # RSI
        delta = window['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        features = [
            window['close'].div(ma_5).values,
            window['close'].div(ma_20).values,
            rsi.values
        ]
        
        return np.column_stack(features)
    
    def _extract_regime_features(self, regime: Dict, index: int) -> np.ndarray:
        """Extract regime-based features"""
        return np.array([
            regime['trend_strength'][index],
            regime['volatility_regime'][index],
            regime['momentum_regime'][index]
        ])
    
    def _add_correlation_features(self,
                                features: np.ndarray,
                                correlations: Dict) -> np.ndarray:
        """Add correlation-based features"""
        correlation_features = np.array([
            correlations['price']['rolling'],
            correlations['volume']['rolling'],
            correlations['volatility']['rolling']
        ])
        
        return np.column_stack([features, correlation_features])
    
    def predict_dual_markets(self,
                           btc_features: np.ndarray,
                           bch_features: np.ndarray) -> Dict:
        """Make predictions for both markets"""
        predictions = self.model.predict([btc_features, bch_features])
        
        return {
            'price_prediction': predictions[0],
            'direction_prediction': predictions[1],
            'volatility_prediction': predictions[2]
        }
    
    def evaluate_prediction_quality(self,
                                  predictions: Dict,
                                  actual_values: Dict) -> Dict:
        """Evaluate prediction quality"""
        evaluation = {}
        
        # Price prediction error
        price_mae = np.mean(np.abs(predictions['price_prediction'] - actual_values['price']))
        evaluation['price_mae'] = price_mae
        
        # Direction accuracy
        direction_accuracy = np.mean(
            (predictions['direction_prediction'] > 0.5) == actual_values['direction']
        )
        evaluation['direction_accuracy'] = direction_accuracy
        
        # Volatility prediction error
        vol_mse = np.mean(
            (predictions['volatility_prediction'] - actual_values['volatility']) ** 2
        )
        evaluation['volatility_mse'] = vol_mse
        
        return evaluation</content>