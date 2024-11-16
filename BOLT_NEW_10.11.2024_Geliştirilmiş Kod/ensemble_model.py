from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Dense
import numpy as np

class EnsembleModel:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights is not None else [1/len(models)] * len(models)
    
    def predict(self, X):
        predictions = []
        for model, weight in zip(self.models, self.weights):
            pred = model.predict(X, verbose=0)
            predictions.append(pred * weight)
        
        return np.sum(predictions, axis=0)
    
    def update_weights(self, performances):
        """Softmax ile model ağırlıklarını güncelle"""
        exp_perf = np.exp(performances)
        self.weights = exp_perf / np.sum(exp_perf)
    
    def create_stacked_model(self, input_shape):
        """Stacking ensemble modeli oluştur"""
        inputs = Input(shape=input_shape)
        predictions = []
        
        for model in self.models:
            pred = model(inputs)
            predictions.append(pred)
        
        combined = Concatenate()(predictions)
        final_output = Dense(1, activation='linear')(combined)
        
        return Model(inputs=inputs, outputs=final_output)