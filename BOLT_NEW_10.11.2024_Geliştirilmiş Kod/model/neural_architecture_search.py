import tensorflow as tf
import numpy as np
from typing import List, Dict, Tuple
import optuna
from tqdm import tqdm
import logging

class NeuralArchitectureSearch:
    def __init__(self, 
                 input_shape: tuple,
                 n_trials: int = 100,
                 max_layers: int = 10):
        self.input_shape = input_shape
        self.n_trials = n_trials
        self.max_layers = max_layers
        self.best_model = None
        self.best_score = float('inf')
        
        # Logger ayarları
        self.logger = logging.getLogger(__name__)
        
    def search(self, X: np.ndarray, y: np.ndarray) -> Tuple[tf.keras.Model, float]:
        """Perform neural architecture search"""
        self.logger.info("🔍 Neural Architecture Search başlatılıyor...")
        
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        with tqdm(total=self.n_trials, desc="Mimari Arama") as pbar:
            study.optimize(
                lambda trial: self._objective(trial, X, y),
                n_trials=self.n_trials,
                callbacks=[
                    lambda study, trial: pbar.update(1),
                    self._log_trial
                ]
            )
        
        self.logger.info(f"\n✨ En iyi mimari bulundu!")
        self.logger.info(f"En iyi skor: {study.best_value:.4f}")
        
        return self.best_model, study.best_value
    
    def _log_trial(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        """Log trial results"""
        if trial.value < self.best_score:
            self.logger.info(f"Yeni en iyi skor: {trial.value:.4f} (Trial {trial.number})")
    
    def _build_layer(self, trial: optuna.Trial, layer_idx: int, n_layers: int) -> tf.keras.layers.Layer:
        """Build a single layer based on trial suggestions"""
        layer_type = trial.suggest_categorical(
            f'layer_{layer_idx}_type',
            ['Dense', 'LSTM', 'GRU', 'Conv1D']
        )
        
        units = trial.suggest_int(f'layer_{layer_idx}_units', 16, 512, log=True)
        activation = trial.suggest_categorical(
            f'layer_{layer_idx}_activation',
            ['relu', 'elu', 'tanh']
        )
        
        if layer_type == 'Dense':
            layer = tf.keras.layers.Dense(units, activation=activation)
        elif layer_type == 'LSTM':
            layer = tf.keras.layers.LSTM(
                units,
                return_sequences=(layer_idx < n_layers-1)
            )
        elif layer_type == 'GRU':
            layer = tf.keras.layers.GRU(
                units,
                return_sequences=(layer_idx < n_layers-1)
            )
        elif layer_type == 'Conv1D':
            kernel_size = trial.suggest_int(f'layer_{layer_idx}_kernel', 2, 5)
            layer = tf.keras.layers.Conv1D(
                units,
                kernel_size,
                activation=activation
            )
            
        return layer
    
    def _objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """Objective function for architecture search"""
        # Define architecture
        n_layers = trial.suggest_int('n_layers', 1, self.max_layers)
        
        # Build model
        model = tf.keras.Sequential([tf.keras.layers.Input(shape=self.input_shape)])
        
        # Add layers
        for i in range(n_layers):
            # Add main layer
            model.add(self._build_layer(trial, i, n_layers))
            
            # Add regularization
            dropout = trial.suggest_float(f'layer_{i}_dropout', 0.0, 0.5)
            if dropout > 0:
                model.add(tf.keras.layers.Dropout(dropout))
            
            # Add normalization
            if trial.suggest_categorical(f'layer_{i}_norm', [True, False]):
                model.add(tf.keras.layers.BatchNormalization())
        
        # Output layer
        model.add(tf.keras.layers.Dense(1))
        
        # Compile model
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        # Train and evaluate
        try:
            history = model.fit(
                X, y,
                validation_split=0.2,
                epochs=50,
                batch_size=trial.suggest_categorical('batch_size', [16, 32, 64]),
                verbose=0,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.2,
                        patience=5,
                        min_lr=1e-6
                    )
                ]
            )
            
            val_loss = min(history.history['val_loss'])
            
            if val_loss < self.best_score:
                self.best_score = val_loss
                self.best_model = model
                
            return val_loss
            
        except Exception as e:
            self.logger.warning(f"Trial failed: {str(e)}")
            raise optuna.exceptions.TrialPruned()