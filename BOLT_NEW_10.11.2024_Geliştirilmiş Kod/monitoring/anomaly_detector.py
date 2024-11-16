import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

class AnomalyDetector:
    def __init__(self, contamination=0.1):
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.autoencoder = self._build_autoencoder()
        
    def _build_autoencoder(self):
        encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu')
        ])
        
        decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(128, activation='sigmoid')
        ])
        
        autoencoder = tf.keras.Sequential([encoder, decoder])
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder
    
    def fit(self, data):
        scaled_data = self.scaler.fit_transform(data)
        self.isolation_forest.fit(scaled_data)
        self.autoencoder.fit(scaled_data, scaled_data, 
                           epochs=100, batch_size=32, verbose=0)
    
    def detect_anomalies(self, data, threshold=3):
        scaled_data = self.scaler.transform(data)
        
        # Isolation Forest anomaly scores
        if_scores = self.isolation_forest.score_samples(scaled_data)
        if_anomalies = if_scores < np.percentile(if_scores, 5)
        
        # Autoencoder reconstruction error
        reconstructed = self.autoencoder.predict(scaled_data)
        reconstruction_error = np.mean(np.abs(scaled_data - reconstructed), axis=1)
        ae_anomalies = reconstruction_error > np.mean(reconstruction_error) + \
                      threshold * np.std(reconstruction_error)
        
        # Combine both methods
        return np.logical_or(if_anomalies, ae_anomalies)