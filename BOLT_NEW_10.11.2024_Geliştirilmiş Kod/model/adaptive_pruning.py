import tensorflow as tf
import numpy as np
from typing import Dict
from tqdm import tqdm

class AdaptivePruning:
    def __init__(self, 
                 model: tf.keras.Model,
                 target_sparsity: float = 0.5):
        self.model = model
        self.target_sparsity = target_sparsity
        self.pruned_model = None
        self.importance_scores = {}
        
    def prune_model(self, X: np.ndarray, y: np.ndarray) -> tf.keras.Model:
        """Prune model while maintaining performance"""
        print("\n✂️ Model budama başlatılıyor...")
        
        # Calculate importance scores
        print("Önem skorları hesaplanıyor...")
        self._calculate_importance_scores(X, y)
        
        # Create pruned model
        print("Budanmış model oluşturuluyor...")
        self.pruned_model = self._create_pruned_model()
        
        # Fine-tune pruned model
        print("Model ince ayarı yapılıyor...")
        self._fine_tune_pruned_model(X, y)
        
        stats = self.get_compression_stats()
        print("\n📊 Sıkıştırma İstatistikleri:")
        print(f"Orijinal parametreler: {stats['original_parameters']:,}")
        print(f"Budanmış parametreler: {stats['pruned_parameters']:,}")
        print(f"Sıkıştırma oranı: {stats['compression_ratio']:.2%}")
        
        return self.pruned_model
    
    def _calculate_importance_scores(self, X: np.ndarray, y: np.ndarray):
        """Calculate importance scores for weights"""
        for layer in tqdm(self.model.layers, desc="Katman Analizi"):
            if isinstance(layer, tf.keras.layers.Dense):
                weights = layer.get_weights()[0]
                
                # Calculate scores based on weight magnitude and gradients
                with tf.GradientTape() as tape:
                    predictions = self.model(X)
                    loss = tf.keras.losses.mean_squared_error(y, predictions)
                
                gradients = tape.gradient(loss, layer.trainable_variables)
                grad_weights = gradients[0].numpy()
                
                # Combine magnitude and gradient information
                importance = np.abs(weights * grad_weights)
                self.importance_scores[layer.name] = importance
    
    def _create_pruned_model(self) -> tf.keras.Model:
        """Create pruned version of the model"""
        pruned_model = tf.keras.models.clone_model(self.model)
        
        for layer in pruned_model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                weights = layer.get_weights()
                importance = self.importance_scores[layer.name]
                
                # Calculate threshold
                threshold = np.percentile(
                    importance,
                    self.target_sparsity * 100
                )
                
                # Create binary mask
                mask = importance > threshold
                
                # Apply mask to weights
                weights[0] = weights[0] * mask
                layer.set_weights(weights)
        
        return pruned_model
    
    def _fine_tune_pruned_model(self, X: np.ndarray, y: np.ndarray):
        """Fine-tune pruned model"""
        self.pruned_model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss='mse'
        )
        
        with tqdm(total=20, desc="İnce Ayar") as pbar:
            self.pruned_model.fit(
                X, y,
                epochs=20,
                batch_size=32,
                verbose=0,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        patience=5,
                        restore_best_weights=True
                    ),
                    tf.keras.callbacks.LambdaCallback(
                        on_epoch_end=lambda epoch, logs: pbar.update(1)
                    )
                ]
            )
    
    def get_compression_stats(self) -> Dict:
        """Get model compression statistics"""
        original_params = np.sum([
            np.prod(v.get_shape().as_list())
            for v in self.model.trainable_variables
        ])
        
        pruned_params = np.sum([
            np.count_nonzero(v.numpy())
            for v in self.pruned_model.trainable_variables
        ])
        
        return {
            'original_parameters': original_params,
            'pruned_parameters': pruned_params,
            'compression_ratio': pruned_params / original_params,
            'sparsity': 1 - (pruned_params / original_params)
        }