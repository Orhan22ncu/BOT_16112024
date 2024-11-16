import tensorflow as tf
import numpy as np
from typing import List, Dict

class ModelDistillation:
    def __init__(self, 
                 teacher_model: tf.keras.Model,
                 temperature: float = 3.0):
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.student_model = None
        
    def create_student_model(self, 
                           architecture: List[Dict],
                           input_shape: tuple):
        """Create compressed student model"""
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=input_shape))
        
        for layer_config in architecture:
            layer_type = layer_config.pop('type')
            if hasattr(tf.keras.layers, layer_type):
                layer_class = getattr(tf.keras.layers, layer_type)
                model.add(layer_class(**layer_config))
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
        
        self.student_model = model
        return model
    
    def distill_knowledge(self,
                         X_train: np.ndarray,
                         y_train: np.ndarray,
                         epochs: int = 100,
                         batch_size: int = 32):
        """Perform knowledge distillation"""
        if self.student_model is None:
            raise ValueError("Student model not created")
        
        # Get soft targets from teacher
        teacher_predictions = self._get_soft_predictions(X_train)
        
        # Compile student model with distillation loss
        self.student_model.compile(
            optimizer='adam',
            loss=self._distillation_loss(teacher_predictions),
            metrics=['mae']
        )
        
        # Train student model
        history = self.student_model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self._get_callbacks()
        )
        
        return history
    
    def _get_soft_predictions(self, X: np.ndarray):
        """Get softened predictions from teacher model"""
        predictions = self.teacher_model.predict(X)
        return self._soften_predictions(predictions)
    
    def _soften_predictions(self, predictions: np.ndarray):
        """Apply temperature softening to predictions"""
        return tf.nn.softmax(predictions / self.temperature)
    
    def _distillation_loss(self, teacher_predictions):
        """Custom distillation loss function"""
        def loss(y_true, y_pred):
            # Soften student predictions
            y_pred_soft = self._soften_predictions(y_pred)
            
            # Calculate distillation loss
            distillation_loss = tf.keras.losses.KLDivergence()(
                teacher_predictions,
                y_pred_soft
            )
            
            # Calculate student loss
            student_loss = tf.keras.losses.MeanSquaredError()(
                y_true,
                y_pred
            )
            
            # Combine losses
            alpha = 0.1  # Balance between distillation and student loss
            return alpha * student_loss + (1 - alpha) * distillation_loss
        
        return loss
    
    def _get_callbacks(self):
        """Get training callbacks"""
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
    
    def evaluate_compression(self, X_test: np.ndarray, y_test: np.ndarray):
        """Evaluate compression performance"""
        teacher_results = self.teacher_model.evaluate(X_test, y_test)
        student_results = self.student_model.evaluate(X_test, y_test)
        
        # Compare model sizes
        teacher_size = self._get_model_size(self.teacher_model)
        student_size = self._get_model_size(self.student_model)
        
        return {
            'teacher_performance': teacher_results,
            'student_performance': student_results,
            'compression_ratio': student_size / teacher_size,
            'size_reduction': 1 - (student_size / teacher_size)
        }
    
    def _get_model_size(self, model: tf.keras.Model):
        """Calculate model size in parameters"""
        return np.sum([
            np.prod(v.get_shape().as_list())
            for v in model.trainable_variables
        ])