import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold
from typing import List, Dict, Any

class EnsembleTrainer:
    def __init__(self, 
                 model_builders: List[callable],
                 n_splits: int = 5,
                 weights_strategy: str = 'adaptive'):
        self.model_builders = model_builders
        self.n_splits = n_splits
        self.weights_strategy = weights_strategy
        self.models = []
        self.weights = []
        
    def train(self, X, y, validation_data=None):
        kf = KFold(n_splits=self.n_splits, shuffle=True)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            fold_models = []
            for builder in self.model_builders:
                model = builder()
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=100,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            patience=10,
                            restore_best_weights=True
                        )
                    ]
                )
                fold_models.append(model)
            
            self.models.extend(fold_models)
        
        self._compute_ensemble_weights(validation_data)
        
    def _compute_ensemble_weights(self, validation_data):
        if self.weights_strategy == 'adaptive':
            performances = []
            X_val, y_val = validation_data
            
            for model in self.models:
                pred = model.predict(X_val)
                perf = -np.mean((pred - y_val) ** 2)  # Negative MSE
                performances.append(perf)
            
            # Softmax normalization of performances
            performances = np.array(performances)
            self.weights = np.exp(performances) / np.sum(np.exp(performances))
        else:
            # Equal weights
            self.weights = np.ones(len(self.models)) / len(self.models)
    
    def predict(self, X):
        predictions = []
        for model, weight in zip(self.models, self.weights):
            pred = model.predict(X)
            predictions.append(pred * weight)
        
        return np.sum(predictions, axis=0)
    
    def save_ensemble(self, path: str):
        for i, model in enumerate(self.models):
            model.save(f"{path}/model_{i}")
        np.save(f"{path}/weights.npy", self.weights)
    
    def load_ensemble(self, path: str):
        self.models = []
        for i in range(len(self.model_builders) * self.n_splits):
            model = tf.keras.models.load_model(f"{path}/model_{i}")
            self.models.append(model)
        self.weights = np.load(f"{path}/weights.npy")