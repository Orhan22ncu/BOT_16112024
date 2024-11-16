import tensorflow as tf
import numpy as np
import time
from typing import List, Dict
import psutil
import os

class PerformanceProfiler:
    def __init__(self):
        self.metrics = {}
        self.start_time = None
        
    def start_profiling(self):
        """Start performance profiling"""
        self.start_time = time.time()
        self.metrics = {
            'memory_usage': [],
            'cpu_usage': [],
            'inference_times': [],
            'batch_processing_times': []
        }
    
    def profile_inference(self, model: tf.keras.Model, X: np.ndarray):
        """Profile model inference performance"""
        # Memory usage
        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        self.metrics['memory_usage'].append(memory_usage)
        
        # CPU usage
        cpu_usage = psutil.cpu_percent()
        self.metrics['cpu_usage'].append(cpu_usage)
        
        # Inference time
        start_time = time.time()
        model.predict(X)
        inference_time = time.time() - start_time
        self.metrics['inference_times'].append(inference_time)
        
        return {
            'memory_mb': memory_usage,
            'cpu_percent': cpu_usage,
            'inference_ms': inference_time * 1000
        }
    
    def profile_batch_processing(self, 
                               model: tf.keras.Model,
                               X: np.ndarray,
                               batch_sizes: List[int]):
        """Profile different batch size performances"""
        results = {}
        
        for batch_size in batch_sizes:
            times = []
            for i in range(0, len(X), batch_size):
                batch = X[i:i + batch_size]
                start_time = time.time()
                model.predict(batch)
                times.append(time.time() - start_time)
            
            results[batch_size] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'throughput': batch_size / np.mean(times)
            }
        
        return results
    
    def get_optimization_suggestions(self):
        """Generate optimization suggestions"""
        suggestions = []
        
        # Memory optimization
        if np.mean(self.metrics['memory_usage']) > 1000:  # 1GB
            suggestions.append(
                "Consider using batch processing to reduce memory usage"
            )
        
        # CPU optimization
        if np.mean(self.metrics['cpu_usage']) > 80:
            suggestions.append(
                "Consider model quantization or hardware acceleration"
            )
        
        # Inference optimization
        mean_inference = np.mean(self.metrics['inference_times'])
        if mean_inference > 0.1:  # 100ms
            suggestions.append(
                "Consider model pruning or architecture optimization"
            )
        
        return suggestions