import tensorflow as tf
import numpy as np
from typing import List, Dict

class DynamicBatchScheduler:
    def __init__(self, 
                 min_batch_size: int = 16,
                 max_batch_size: int = 256,
                 growth_factor: float = 1.5):
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.growth_factor = growth_factor
        self.current_batch_size = min_batch_size
        self.performance_history = []
        
    def update_batch_size(self, current_loss: float):
        """Dynamically adjust batch size based on training progress"""
        self.performance_history.append(current_loss)
        
        if len(self.performance_history) >= 3:
            recent_improvement = (self.performance_history[-2] - 
                                self.performance_history[-1])
            
            if recent_improvement > 0:  # Loss is decreasing
                self._increase_batch_size()
            else:
                self._decrease_batch_size()
    
    def _increase_batch_size(self):
        """Increase batch size"""
        new_size = min(
            self.max_batch_size,
            int(self.current_batch_size * self.growth_factor)
        )
        self.current_batch_size = new_size
    
    def _decrease_batch_size(self):
        """Decrease batch size"""
        new_size = max(
            self.min_batch_size,
            int(self.current_batch_size / self.growth_factor)
        )
        self.current_batch_size = new_size
    
    def get_current_batch_size(self):
        """Get current batch size"""
        return self.current_batch_size