import tensorflow as tf
import numpy as np
from typing import List, Dict, Any

class MultiTaskLearner:
    def __init__(self, 
                 shared_layers: List[tf.keras.layers.Layer],
                 task_specific_layers: Dict[str, List[tf.keras.layers.Layer]],
                 task_weights: Dict[str, float] = None):
        self.shared_layers = shared_layers
        self.task_specific_layers = task_specific_layers
        self.task_weights = task_weights or {
            task: 1.0 for task in task_specific_layers.keys()
        }
        self.model = self._build_model()
        
    def _build_model(self) -> tf.keras.Model:
        """Build multi-task model architecture"""
        # Shared input
        inputs = tf.keras.Input(shape=self.shared_layers[0].input_shape[1:])
        
        # Shared layers
        x = inputs
        for layer in self.shared_layers:
            x = layer(x)
        
        # Task-specific outputs
        outputs = {}
        for task_name, layers in self.task_specific_layers.items():
            task_output = x
            for layer in layers:
                task_output = layer(task_output)
            outputs[task_name] = task_output
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def compile_model(self, 
                     task_losses: Dict[str, tf.keras.losses.Loss],
                     task_metrics: Dict[str, List[str]],
                     optimizer: tf.keras.optimizers.Optimizer):
        """Compile model with task-specific losses and metrics"""
        self.model.compile(
            optimizer=optimizer,
            loss=task_losses,
            metrics=task_metrics,
            loss_weights=self.task_weights
        )
    
    def train(self,
             train_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
             validation_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = None,
             epochs: int = 100,
             batch_size: int = 32):
        """Train multi-task model"""
        # Prepare data
        X_train = train_data[list(train_data.keys())[0]][0]
        y_train = {task: data[1] for task, data in train_data.items()}
        
        if validation_data:
            X_val = validation_data[list(validation_data.keys())[0]][0]
            y_val = {task: data[1] for task, data in validation_data.items()}
        else:
            X_val = None
            y_val = None
        
        # Train model
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val) if validation_data else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self._get_callbacks()
        )
        
        return history
    
    def _get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
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
    
    def predict(self, 
               X: np.ndarray,
               tasks: List[str] = None) -> Dict[str, np.ndarray]:
        """Make predictions for specified tasks"""
        predictions = self.model.predict(X)
        
        if tasks is None:
            tasks = list(self.task_specific_layers.keys())
        
        return {task: predictions[task] for task in tasks}
    
    def evaluate(self,
                X: np.ndarray,
                y: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Evaluate model performance on each task"""
        return self.model.evaluate(X, y, return_dict=True)