import numpy as np
from sklearn.metrics import pairwise_distances
from typing import List, Tuple, Callable
import tensorflow as tf

class ActiveLearner:
    def __init__(self,
                 model: tf.keras.Model,
                 initial_samples: int = 100,
                 query_strategy: str = 'uncertainty'):
        self.model = model
        self.initial_samples = initial_samples
        self.query_strategy = query_strategy
        self.labeled_indices = []
        self.unlabeled_indices = []
        
    def initialize(self, X: np.ndarray, y: np.ndarray = None):
        """Initialize active learning process"""
        n_samples = len(X)
        self.unlabeled_indices = list(range(n_samples))
        
        # Initial random sampling
        initial_indices = np.random.choice(
            n_samples,
            size=min(self.initial_samples, n_samples),
            replace=False
        )
        
        self.labeled_indices = list(initial_indices)
        self.unlabeled_indices = list(
            set(range(n_samples)) - set(self.labeled_indices)
        )
    
    def query(self, 
             X: np.ndarray,
             n_instances: int = 1) -> List[int]:
        """Query most informative instances"""
        if self.query_strategy == 'uncertainty':
            return self._uncertainty_sampling(X, n_instances)
        elif self.query_strategy == 'diversity':
            return self._diversity_sampling(X, n_instances)
        elif self.query_strategy == 'expected_improvement':
            return self._expected_improvement_sampling(X, n_instances)
        else:
            raise ValueError(f"Unknown query strategy: {self.query_strategy}")
    
    def _uncertainty_sampling(self,
                            X: np.ndarray,
                            n_instances: int) -> List[int]:
        """Query instances with highest prediction uncertainty"""
        # Get predictions and uncertainties
        predictions = []
        for _ in range(10):  # Monte Carlo sampling with dropout
            pred = self.model(X[self.unlabeled_indices], training=True)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        uncertainties = np.std(predictions, axis=0)
        
        # Select most uncertain instances
        uncertain_idx = np.argsort(uncertainties.flatten())[-n_instances:]
        return [self.unlabeled_indices[idx] for idx in uncertain_idx]
    
    def _diversity_sampling(self,
                          X: np.ndarray,
                          n_instances: int) -> List[int]:
        """Query diverse instances using k-means++"""
        # Calculate distances to labeled instances
        distances = pairwise_distances(
            X[self.unlabeled_indices],
            X[self.labeled_indices]
        )
        
        # Select diverse instances
        selected_indices = []
        min_distances = np.min(distances, axis=1)
        
        for _ in range(n_instances):
            # Select instance with maximum minimum distance
            idx = np.argmax(min_distances)
            selected_indices.append(self.unlabeled_indices[idx])
            
            # Update distances
            if len(selected_indices) < n_instances:
                new_distances = pairwise_distances(
                    X[self.unlabeled_indices],
                    X[selected_indices[-1]].reshape(1, -1)
                )
                min_distances = np.minimum(min_distances, new_distances.flatten())
        
        return selected_indices
    
    def _expected_improvement_sampling(self,
                                    X: np.ndarray,
                                    n_instances: int) -> List[int]:
        """Query instances with highest expected improvement"""
        # Get predictions and uncertainties
        predictions = []
        for _ in range(10):
            pred = self.model(X[self.unlabeled_indices], training=True)
            predictions.append(pred)
        
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Calculate expected improvement
        best_value = np.max(mean_pred)
        z = (mean_pred - best_value) / std_pred
        ei = std_pred * (z * norm.cdf(z) + norm.pdf(z))
        
        # Select instances with highest EI
        ei_idx = np.argsort(ei.flatten())[-n_instances:]
        return [self.unlabeled_indices[idx] for idx in ei_idx]
    
    def update(self, queried_indices: List[int]):
        """Update labeled and unlabeled sets"""
        self.labeled_indices.extend(queried_indices)
        self.unlabeled_indices = list(
            set(self.unlabeled_indices) - set(queried_indices)
        )
    
    def train(self,
             X: np.ndarray,
             y: np.ndarray,
             validation_data: Tuple[np.ndarray, np.ndarray] = None):
        """Train model on labeled data"""
        history = self.model.fit(
            X[self.labeled_indices],
            y[self.labeled_indices],
            validation_data=validation_data,
            epochs=100,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=10,
                    restore_best_weights=True
                )
            ]
        )
        return history