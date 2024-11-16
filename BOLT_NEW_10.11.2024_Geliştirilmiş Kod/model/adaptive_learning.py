import tensorflow as tf
import numpy as np
from typing import List, Dict, Any

class AdaptiveLearningSystem:
    def __init__(self, base_model, adaptation_rate=0.01):
        self.base_model = base_model
        self.adaptation_rate = adaptation_rate
        self.meta_learner = self._build_meta_learner()
        self.task_memory = {}
        
    def _build_meta_learner(self):
        return tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
    
    def adapt_to_task(self, task_data: Dict[str, np.ndarray], task_id: str):
        """Adapt model to new task while maintaining previous knowledge"""
        if task_id not in self.task_memory:
            self.task_memory[task_id] = {
                'data_summary': self._compute_task_summary(task_data),
                'performance_history': []
            }
        
        # Compute task similarity
        task_embedding = self._get_task_embedding(task_data)
        similar_tasks = self._find_similar_tasks(task_embedding)
        
        # Transfer learning from similar tasks
        adapted_model = self._transfer_knowledge(similar_tasks)
        
        # Fine-tune on new task
        adapted_model = self._fine_tune(adapted_model, task_data)
        
        # Update task memory
        self._update_task_memory(task_id, adapted_model)
        
        return adapted_model
    
    def _compute_task_summary(self, task_data):
        """Compute statistical summary of task data"""
        summary = {
            'mean': np.mean(task_data['features'], axis=0),
            'std': np.std(task_data['features'], axis=0),
            'feature_importance': self._compute_feature_importance(task_data)
        }
        return summary
    
    def _get_task_embedding(self, task_data):
        """Generate embedding for task characterization"""
        features = task_data['features']
        labels = task_data['labels']
        
        # Statistical features
        stat_features = np.concatenate([
            np.mean(features, axis=0),
            np.std(features, axis=0),
            np.percentile(features, [25, 50, 75], axis=0).flatten()
        ])
        
        # Temporal features
        temporal_features = self._extract_temporal_features(features)
        
        # Combine features
        task_embedding = np.concatenate([stat_features, temporal_features])
        return task_embedding
    
    def _find_similar_tasks(self, task_embedding, n_similar=3):
        """Find most similar tasks in memory"""
        similarities = {}
        for task_id, task_info in self.task_memory.items():
            stored_embedding = task_info['data_summary']['embedding']
            similarity = self._compute_similarity(task_embedding, stored_embedding)
            similarities[task_id] = similarity
        
        # Get top similar tasks
        similar_tasks = sorted(similarities.items(), 
                             key=lambda x: x[1], 
                             reverse=True)[:n_similar]
        return similar_tasks
    
    def _transfer_knowledge(self, similar_tasks):
        """Transfer knowledge from similar tasks"""
        adapted_weights = []
        for task_id, similarity in similar_tasks:
            task_weights = self.task_memory[task_id]['model_weights']
            adapted_weights.append(task_weights * similarity)
        
        # Combine weights
        final_weights = np.sum(adapted_weights, axis=0)
        
        # Create new model with adapted weights
        adapted_model = tf.keras.models.clone_model(self.base_model)
        adapted_model.set_weights(final_weights)
        
        return adapted_model
    
    def _fine_tune(self, model, task_data, epochs=10):
        """Fine-tune model on new task"""
        model.fit(
            task_data['features'],
            task_data['labels'],
            epochs=epochs,
            batch_size=32,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=3)
            ]
        )
        return model
    
    def _update_task_memory(self, task_id, model):
        """Update task memory with new model"""
        self.task_memory[task_id]['model_weights'] = model.get_weights()
        
    def _compute_similarity(self, embedding1, embedding2):
        """Compute cosine similarity between task embeddings"""
        return np.dot(embedding1, embedding2) / \
               (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    def _extract_temporal_features(self, features):
        """Extract temporal features from time series data"""
        # Implement temporal feature extraction
        pass
    
    def _compute_feature_importance(self, task_data):
        """Compute feature importance scores"""
        # Implement feature importance calculation
        pass