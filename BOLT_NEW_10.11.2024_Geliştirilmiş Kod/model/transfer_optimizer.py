import tensorflow as tf
import numpy as np
from typing import List, Dict, Any

class TransferOptimizer:
    def __init__(self, 
                 source_models: List[tf.keras.Model],
                 target_model: tf.keras.Model):
        self.source_models = source_models
        self.target_model = target_model
        self.adaptation_layers = {}
        
    def adapt_knowledge(self, 
                       source_data: List[Dict[str, np.ndarray]],
                       target_data: Dict[str, np.ndarray]):
        """Transfer and adapt knowledge from source models"""
        # Extract features from source models
        source_features = self._extract_source_features(source_data)
        
        # Create adaptation layers
        self._create_adaptation_layers(source_features)
        
        # Train adaptation layers
        self._train_adaptation_layers(
            source_features,
            target_data['features']
        )
        
        # Fine-tune target model
        self._fine_tune_target_model(target_data)
    
    def _extract_source_features(self, source_data: List[Dict[str, np.ndarray]]):
        """Extract features from source models"""
        features = []
        for model, data in zip(self.source_models, source_data):
            # Get intermediate layer outputs
            feature_extractor = tf.keras.Model(
                inputs=model.input,
                outputs=model.layers[-2].output
            )
            features.append(feature_extractor.predict(data['features']))
        return features
    
    def _create_adaptation_layers(self, source_features: List[np.ndarray]):
        """Create adaptation layers for each source model"""
        for i, features in enumerate(source_features):
            adaptation_layer = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(features.shape[-1])
            ])
            self.adaptation_layers[f'source_{i}'] = adaptation_layer
    
    def _train_adaptation_layers(self,
                               source_features: List[np.ndarray],
                               target_features: np.ndarray):
        """Train adaptation layers"""
        for i, features in enumerate(source_features):
            adaptation_layer = self.adaptation_layers[f'source_{i}']
            
            # Train adaptation layer
            adaptation_layer.compile(
                optimizer='adam',
                loss='mse'
            )
            
            adaptation_layer.fit(
                features,
                target_features,
                epochs=50,
                batch_size=32,
                verbose=0
            )
    
    def _fine_tune_target_model(self, target_data: Dict[str, np.ndarray]):
        """Fine-tune target model with adapted knowledge"""
        # Freeze early layers
        for layer in self.target_model.layers[:-2]:
            layer.trainable = False
        
        # Fine-tune
        self.target_model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss='mse'
        )
        
        self.target_model.fit(
            target_data['features'],
            target_data['labels'],
            epochs=20,
            batch_size=32,
            verbose=0
        )
    
    def predict_with_transfer(self, X: np.ndarray):
        """Make predictions using transferred knowledge"""
        # Get source model predictions
        source_predictions = []
        for i, model in enumerate(self.source_models):
            features = model.layers[-2].output
            adapted_features = self.adaptation_layers[f'source_{i}'](features)
            source_predictions.append(adapted_features)
        
        # Combine predictions
        combined_features = tf.keras.layers.Concatenate()(source_predictions)
        
        # Final prediction
        return self.target_model(combined_features)