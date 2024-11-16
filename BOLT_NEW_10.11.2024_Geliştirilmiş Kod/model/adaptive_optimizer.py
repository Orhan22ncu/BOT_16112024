import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer
import numpy as np

class AdaptiveLearningOptimizer(Optimizer):
    def __init__(self, 
                 learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-7,
                 name="AdaptiveLearningOptimizer",
                 **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", learning_rate)
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self.epsilon = epsilon
        
    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")
            self.add_slot(var, "v")
            self.add_slot(var, "gradient_history")
    
    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        gradient_history = self.get_slot(var, "gradient_history")
        
        beta_1_t = self._get_hyper("beta_1", var_dtype)
        beta_2_t = self._get_hyper("beta_2", var_dtype)
        
        # Update moments
        m_t = beta_1_t * m + (1 - beta_1_t) * grad
        v_t = beta_2_t * v + (1 - beta_2_t) * tf.square(grad)
        
        # Update gradient history
        gradient_history_t = gradient_history.assign(grad)
        
        # Compute adaptive learning rate
        adaptive_lr = lr_t * tf.sqrt(1 - beta_2_t) / (1 - beta_1_t)
        
        # Apply updates
        var_update = var - adaptive_lr * m_t / (tf.sqrt(v_t) + self.epsilon)
        
        # Update slots
        m.assign(m_t)
        v.assign(v_t)
        
        return var_update
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "beta_1": self._serialize_hyperparameter("beta_1"),
            "beta_2": self._serialize_hyperparameter("beta_2"),
            "epsilon": self.epsilon,
        })
        return config