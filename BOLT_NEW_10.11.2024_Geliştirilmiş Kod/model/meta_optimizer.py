import tensorflow as tf
import numpy as np
from typing import Callable, Dict
import optuna
from tqdm import tqdm

class MetaOptimizer:
    def __init__(self, 
                 model_builder: Callable,
                 n_trials: int = 100):
        self.model_builder = model_builder
        self.n_trials = n_trials
        self.best_params = None
        self.best_score = float('inf')
    
    def optimize(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Optimize hyperparameters"""
        print("\n🔍 Meta-optimizasyon başlatılıyor...")
        study = optuna.create_study(direction='minimize')
        
        with tqdm(total=self.n_trials, desc="Hiperparametre Optimizasyonu") as pbar:
            study.optimize(
                lambda trial: self._objective(trial, X, y),
                n_trials=self.n_trials,
                callbacks=[lambda study, trial: pbar.update(1)]
            )
        
        self.best_params = study.best_params
        print(f"\n✨ En iyi hiperparametreler bulundu!")
        print(f"En iyi skor: {study.best_value:.4f}")
        
        return self.best_params
    
    def _objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """Objective function for hyperparameter optimization"""
        # Get model
        model = self.model_builder()
        
        # Hyperparameter search space
        batch_size = trial.suggest_int('batch_size', 16, 256)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
        
        # Configure optimizer
        if optimizer_name == 'Adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate)
        elif optimizer_name == 'RMSprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate)
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate)
        
        # Compile model
        model.compile(optimizer=optimizer, loss='mse')
        
        # Train and evaluate
        history = model.fit(
            X, y,
            validation_split=0.2,
            epochs=50,
            batch_size=batch_size,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=10,
                    restore_best_weights=True
                )
            ]
        )
        
        val_loss = min(history.history['val_loss'])
        
        if val_loss < self.best_score:
            self.best_score = val_loss
        
        return val_loss