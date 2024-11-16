import tensorflow as tf
import numpy as np
from collections import deque
import threading
import queue
import time

class RealtimePredictor:
    def __init__(self, model, feature_engineer, buffer_size=100):
        self.model = model
        self.feature_engineer = feature_engineer
        self.prediction_buffer = deque(maxlen=buffer_size)
        self.data_queue = queue.Queue()
        self.is_running = False
        
    def start(self):
        self.is_running = True
        self.prediction_thread = threading.Thread(target=self._prediction_loop)
        self.prediction_thread.start()
    
    def stop(self):
        self.is_running = False
        self.prediction_thread.join()
    
    def _prediction_loop(self):
        batch_data = []
        while self.is_running:
            try:
                data = self.data_queue.get(timeout=0.1)
                batch_data.append(data)
                
                if len(batch_data) >= 32:  # Batch size
                    predictions = self._process_batch(batch_data)
                    self.prediction_buffer.extend(predictions)
                    batch_data = []
                    
            except queue.Empty:
                if batch_data:
                    predictions = self._process_batch(batch_data)
                    self.prediction_buffer.extend(predictions)
                    batch_data = []
    
    def _process_batch(self, batch_data):
        features = self.feature_engineer.create_features(batch_data)
        predictions = self.model.predict(features, verbose=0)
        return predictions
    
    def get_latest_prediction(self):
        return self.prediction_buffer[-1] if self.prediction_buffer else None
    
    @tf.function
    def predict_single(self, features):
        """Optimize single prediction with TF function"""
        return self.model(features, training=False)