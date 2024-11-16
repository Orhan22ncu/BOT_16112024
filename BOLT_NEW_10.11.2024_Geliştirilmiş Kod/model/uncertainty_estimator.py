import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from typing import Dict
from tqdm import tqdm

class UncertaintyEstimator:
    def __init__(self, model: tf.keras.Model, num_samples: int = 100):
        self.model = model
        self.num_samples = num_samples
        self.tfd = tfp.distributions
        
    def estimate_uncertainty(self, X: np.ndarray) -> Dict:
        """Estimate both aleatoric and epistemic uncertainty"""
        print("\n🔍 Belirsizlik analizi başlatılıyor...")
        predictions = []
        
        # Monte Carlo Dropout sampling
        with tqdm(total=self.num_samples, desc="Monte Carlo Örnekleme") as pbar:
            for _ in range(self.num_samples):
                pred = self.model(X, training=True)  # Enable dropout during inference
                predictions.append(pred)
                pbar.update(1)
        
        predictions = np.array(predictions)
        
        # Calculate uncertainties
        mean_pred = np.mean(predictions, axis=0)
        epistemic = np.var(predictions, axis=0)  # Model uncertainty
        aleatoric = self._estimate_aleatoric(X)  # Data uncertainty
        
        print("\n📊 Belirsizlik Metrikleri:")
        print(f"Epistemik (Model) Belirsizliği: {np.mean(epistemic):.4f}")
        print(f"Aleatorik (Veri) Belirsizliği: {np.mean(aleatoric):.4f}")
        
        return {
            'prediction': mean_pred,
            'epistemic_uncertainty': epistemic,
            'aleatoric_uncertainty': aleatoric,
            'total_uncertainty': epistemic + aleatoric
        }
    
    def _estimate_aleatoric(self, X: np.ndarray) -> np.ndarray:
        """Estimate aleatoric uncertainty using probabilistic outputs"""
        predictions = []
        
        with tqdm(total=self.num_samples, desc="Aleatorik Analiz") as pbar:
            for _ in range(self.num_samples):
                pred = self.model(X, training=False)
                predictions.append(pred)
                pbar.update(1)
        
        predictions = np.array(predictions)
        return np.var(predictions, axis=0)
    
    def get_prediction_intervals(self, X: np.ndarray, confidence_level: float = 0.95) -> Dict:
        """Calculate prediction intervals"""
        predictions = self.estimate_uncertainty(X)
        
        z_score = tfp.stats.norm.ppf((1 + confidence_level) / 2)
        std_dev = np.sqrt(predictions['total_uncertainty'])
        
        lower_bound = predictions['prediction'] - z_score * std_dev
        upper_bound = predictions['prediction'] + z_score * std_dev
        
        print(f"\n📈 Tahmin Aralıkları ({confidence_level*100:.0f}% Güven):")
        print(f"Ortalama Aralık Genişliği: {np.mean(upper_bound - lower_bound):.4f}")
        
        return {
            'prediction': predictions['prediction'],
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_level': confidence_level
        }