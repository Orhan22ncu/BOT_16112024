import tensorflow as tf
import numpy as np
from collections import deque
from typing import List, Dict, Any

class ContinualLearner:
    def __init__(self, 
                 model: tf.keras.Model,
                 memory_size: int = 1000,
                 rehearsal_ratio: float = 0.3):
        self.model = model
        self.memory_size = memory_size
        self.rehearsal_ratio = rehearsal_ratio
        self.memory_buffer = deque(maxlen=memory_size)
        self.task_boundaries = []
        
    def update(self, X_new: np.ndarray, y_new: np.ndarray):
        """Update model with new data while preventing catastrophic forgetting"""
        # Store new examples in memory
        self._update_memory(X_new, y_new)
        
        # Get rehearsal data
        X_rehearsal, y_rehearsal = self._get_rehearsal_data()
        
        # Combine new and rehearsal data
        X_combined = np.concatenate([X_new, X_rehearsal])
        y_combined = np.concatenate([y_new, y_rehearsal])
        
        # Update model
        history = self.model.fit(
            X_combined, y_combined,
            epochs=10,
            batch_size=32,
            callbacks=self._get_callbacks()
        )
        
        # Evaluate catastrophic forgetting
        forgetting_metric = self._evaluate_forgetting(X_rehearsal, y_rehearsal)
        
        return {
            'training_history': history.history,
            'forgetting_metric': forgetting_metric
        }
    
    def _update_memory(self, X: np.ndarray, y: np.ndarray):
        """Update memory buffer with new examples"""
        # Reservoir sampling for memory update
        for i in range(len(X)):
            if len(self.memory_buffer) < self.memory_size:
                self.memory_buffer.append((X[i], y[i]))
            else:
                j = np.random.randint(0, i + 1)
                if j < self.memory_size:
                    self.memory_buffer[j] = (X[i], y[i])
    
    def _get_rehearsal_data(self):
        """Get rehearsal data from memory buffer"""
        n_rehearsal = int(self.memory_size * self.rehearsal_ratio)
        indices = np.random.choice(
            len(self.memory_buffer),
            size=min(n_rehearsal, len(self.memory_buffer)),
            replace=False
        )
        
        X_rehearsal = []
        y_rehearsal = []
        
        for idx in indices:
            x, y = self.memory_buffer[idx]
            X_rehearsal.append(x)
            y_rehearsal.append(y)
        
        return np.array(X_rehearsal), np.array(y_rehearsal)
    
    def _evaluate_forgetting(self, X: np.ndarray, y: np.ndarray):
        """Evaluate catastrophic forgetting on memory buffer"""
        initial_performance = self.model.evaluate(X, y, verbose=0)
        return initial_performance
    
    def _get_callbacks(self):
        """Get training callbacks"""
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ]
    
    def detect_concept_drift(self, X: np.ndarray, window_size: int = 100):
        """Detect concept drift in data stream"""
        predictions = self.model.predict(X)
        
        # Calculate prediction statistics in sliding window
        windows = self._create_sliding_windows(predictions, window_size)
        stats = self._calculate_window_statistics(windows)
        
        # Detect significant changes
        drift_points = self._detect_distribution_changes(stats)
        
        return drift_points
    
    def _create_sliding_windows(self, data: np.ndarray, window_size: int):
        """Create sliding windows from data"""
        return [
            data[i:i + window_size]
            for i in range(0, len(data) - window_size + 1)
        ]
    
    def _calculate_window_statistics(self, windows: List[np.ndarray]):
        """Calculate statistics for each window"""
        stats = []
        for window in windows:
            stats.append({
                'mean': np.mean(window),
                'std': np.std(window),
                'skewness': self._calculate_skewness(window),
                'kurtosis': self._calculate_kurtosis(window)
            })
        return stats
    
    def _detect_distribution_changes(self, stats: List[Dict]):
        """Detect significant changes in distribution statistics"""
        change_points = []
        baseline = stats[0]
        
        for i, stat in enumerate(stats[1:], 1):
            if self._is_significant_change(baseline, stat):
                change_points.append(i)
                baseline = stat
        
        return change_points
    
    def _is_significant_change(self, baseline: Dict, current: Dict):
        """Determine if change is significant"""
        threshold = 2.0  # Standard deviations
        
        return any(
            abs(baseline[key] - current[key]) > threshold * baseline['std']
            for key in ['mean', 'skewness', 'kurtosis']
        )
    
    @staticmethod
    def _calculate_skewness(data):
        """Calculate distribution skewness"""
        return np.mean(((data - np.mean(data)) / np.std(data)) ** 3)
    
    @staticmethod
    def _calculate_kurtosis(data):
        """Calculate distribution kurtosis"""
        return np.mean(((data - np.mean(data)) / np.std(data)) ** 4)