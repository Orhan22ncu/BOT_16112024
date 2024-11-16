import optuna
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import tensorflow as tf

class AutoModelSelector:
    def __init__(self, input_shape, n_trials=100):
        self.input_shape = input_shape
        self.n_trials = n_trials
        self.best_model = None
        self.best_score = float('inf')
        
    def search(self, X, y):
        study = optuna.create_study(direction='minimize')
        study.optimize(
            lambda trial: self._objective(trial, X, y),
            n_trials=self.n_trials
        )
        
        return self.best_model, study.best_params
    
    def _objective(self, trial, X, y):
        # Define model architecture search space
        n_layers = trial.suggest_int('n_layers', 1, 5)
        model = tf.keras.Sequential()
        
        # Input layer
        model.add(tf.keras.layers.Input(shape=self.input_shape))
        
        # Hidden layers
        for i in range(n_layers):
            layer_type = trial.suggest_categorical(
                f'layer_{i}_type',
                ['LSTM', 'GRU', 'Conv1D', 'Dense']
            )
            units = trial.suggest_int(f'layer_{i}_units', 16, 256)
            
            if layer_type == 'LSTM':
                model.add(tf.keras.layers.LSTM(units, return_sequences=(i < n_layers-1)))
            elif layer_type == 'GRU':
                model.add(tf.keras.layers.GRU(units, return_sequences=(i < n_layers-1)))
            elif layer_type == 'Conv1D':
                kernel_size = trial.suggest_int(f'layer_{i}_kernel', 2, 5)
                model.add(tf.keras.layers.Conv1D(units, kernel_size, padding='same'))
            else:
                model.add(tf.keras.layers.Dense(units))
            
            # Add regularization
            dropout = trial.suggest_float(f'layer_{i}_dropout', 0.0, 0.5)
            if dropout > 0:
                model.add(tf.keras.layers.Dropout(dropout))
        
        # Output layer
        model.add(tf.keras.layers.Dense(1))
        
        # Compile model
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss='huber_loss',
            metrics=['mae']
        )
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=32,
                verbose=0,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        patience=10,
                        restore_best_weights=True
                    )
                ]
            )
            
            scores.append(min(history.history['val_loss']))
        
        mean_score = np.mean(scores)
        if mean_score < self.best_score:
            self.best_score = mean_score
            self.best_model = model
        
        return mean_score