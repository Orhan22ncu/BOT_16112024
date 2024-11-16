import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple
from .neural_architecture_search import NeuralArchitectureSearch
from .meta_optimizer import MetaOptimizer
from .adaptive_pruning import AdaptivePruning
from .uncertainty_estimator import UncertaintyEstimator
from tqdm import tqdm
import time
from datetime import datetime

class SelfEvolutionEngine:
    def __init__(self, 
                 base_model: tf.keras.Model,
                 evolution_interval: int = 1000,
                 min_improvement: float = 0.01):
        self.base_model = base_model
        self.evolution_interval = evolution_interval
        self.min_improvement = min_improvement
        self.performance_history = []
        
        print("🚀 Self Evolution Engine başlatılıyor...")
        print(f"Model Özeti:")
        self.base_model.summary()
        
        # Alt sistemler
        print("\n🔧 Alt sistemler yükleniyor...")
        start_time = time.time()
        
        self.architecture_search = NeuralArchitectureSearch(
            input_shape=base_model.input_shape[1:]
        )
        self.meta_optimizer = MetaOptimizer(
            model_builder=self._build_model
        )
        self.pruning_engine = AdaptivePruning(base_model)
        self.uncertainty_estimator = UncertaintyEstimator(base_model)
        
        setup_time = time.time() - start_time
        print(f"✅ Alt sistemler yüklendi (Süre: {setup_time:.2f} sn)")
        
    def evolve(self, X: np.ndarray, y: np.ndarray):
        """Sistemin kendi kendini geliştirme döngüsü"""
        print("\n🔄 Evrim döngüsü başlatılıyor...")
        evolution_start = time.time()
        
        print("📊 Mevcut performans değerlendiriliyor...")
        current_performance = self._evaluate_performance(X, y)
        self.performance_history.append(current_performance)
        print(f"Mevcut Performans: {current_performance:.4f}")
        
        if self._should_evolve():
            print("\n🧬 Evrim süreci başlıyor...")
            
            # Mimari optimizasyonu
            print("\n1/4 - 🏗️ Mimari optimizasyonu yapılıyor...")
            arch_start = time.time()
            new_architecture, arch_performance = self.architecture_search.search(X, y)
            print(f"Mimari optimizasyonu tamamlandı ({time.time() - arch_start:.2f} sn)")
            print(f"Yeni mimari performansı: {arch_performance:.4f}")
            
            # Meta-öğrenme
            print("\n2/4 - 🧠 Meta-öğrenme optimizasyonu yapılıyor...")
            meta_start = time.time()
            optimized_params = self.meta_optimizer.optimize(X, y)
            print(f"Meta-öğrenme tamamlandı ({time.time() - meta_start:.2f} sn)")
            
            # Model budama
            print("\n3/4 - ✂️ Model budama işlemi yapılıyor...")
            prune_start = time.time()
            pruned_model = self.pruning_engine.prune_model(X, y)
            print(f"Model budama tamamlandı ({time.time() - prune_start:.2f} sn)")
            
            # Belirsizlik analizi
            print("\n4/4 - 📈 Belirsizlik analizi yapılıyor...")
            uncert_start = time.time()
            uncertainty_metrics = self.uncertainty_estimator.estimate_uncertainty(X)
            print(f"Belirsizlik analizi tamamlandı ({time.time() - uncert_start:.2f} sn)")
            
            # Model seçimi
            print("\n🔍 En iyi model seçiliyor...")
            best_model = self._select_best_model([
                self.base_model,
                new_architecture,
                pruned_model
            ], X, y)
            
            # Model güncelleme
            if self._is_significant_improvement(best_model, X, y):
                print("\n⭐ Önemli gelişme tespit edildi! Model güncelleniyor...")
                self.base_model = best_model
                self._adapt_learning_strategy(uncertainty_metrics)
                print("✅ Model başarıyla güncellendi!")
            else:
                print("\n📌 Önemli bir gelişme tespit edilmedi. Model korunuyor.")
        
        total_time = time.time() - evolution_start
        print(f"\n✨ Evrim döngüsü tamamlandı (Toplam süre: {total_time:.2f} sn)")
        self._save_progress_report()
    
    def _evaluate_performance(self, X: np.ndarray, y: np.ndarray) -> float:
        """Model performansını değerlendir"""
        print("Performans değerlendiriliyor...")
        with tqdm(total=100, desc="Değerlendirme") as pbar:
            metrics = self.base_model.evaluate(X, y, verbose=0)
            pbar.update(100)
        return metrics[0] if isinstance(metrics, list) else metrics
    
    def _should_evolve(self) -> bool:
        """Evrim gerekli mi kontrol et"""
        if len(self.performance_history) < 2:
            return True
            
        recent_improvement = (self.performance_history[-2] - 
                            self.performance_history[-1])
        return recent_improvement < self.min_improvement
    
    def _select_best_model(self, 
                          models: List[tf.keras.Model],
                          X: np.ndarray,
                          y: np.ndarray) -> tf.keras.Model:
        """En iyi performanslı modeli seç"""
        performances = []
        print("\nModel karşılaştırması yapılıyor...")
        for i, model in enumerate(models, 1):
            with tqdm(total=100, desc=f"Model {i}/3") as pbar:
                perf = model.evaluate(X, y, verbose=0)
                performances.append(perf)
                pbar.update(100)
        
        best_idx = np.argmin(performances)
        print(f"En iyi model: Model {best_idx + 1} (Performans: {performances[best_idx]:.4f})")
        return models[best_idx]
    
    def _is_significant_improvement(self,
                                  new_model: tf.keras.Model,
                                  X: np.ndarray,
                                  y: np.ndarray) -> bool:
        """Yeni model önemli bir gelişme sağlıyor mu"""
        base_perf = self._evaluate_performance(X, y)
        new_perf = new_model.evaluate(X, y, verbose=0)
        improvement = base_perf - new_perf
        print(f"İyileştirme: {improvement:.4f} (Minimum gerekli: {self.min_improvement})")
        return improvement > self.min_improvement
    
    def _adapt_learning_strategy(self, uncertainty_metrics: Dict):
        """Öğrenme stratejisini belirsizlik metriklerine göre adapte et"""
        print("\nÖğrenme stratejisi adapte ediliyor...")
        if uncertainty_metrics['epistemic_uncertainty'] > 0.5:
            print("⚠️ Yüksek epistemik belirsizlik tespit edildi - Veri toplama başlatılıyor")
            self._request_more_data()
        
        if uncertainty_metrics['aleatoric_uncertainty'] > 0.5:
            print("⚠️ Yüksek aleatorik belirsizlik tespit edildi - Gürültü azaltma başlatılıyor")
            self._enhance_noise_reduction()
    
    def _request_more_data(self):
        """Veri toplama stratejisi"""
        print("📥 Ek veri toplama talebi oluşturuldu")
        # Veri toplama mantığı buraya eklenecek
    
    def _enhance_noise_reduction(self):
        """Gürültü azaltma stratejilerini güçlendir"""
        print("🔍 Gürültü azaltma stratejileri güçlendiriliyor")
        # Gürültü azaltma mantığı buraya eklenecek
    
    def _build_model(self) -> tf.keras.Model:
        """Yeni model oluştur"""
        return tf.keras.models.clone_model(self.base_model)
    
    def _save_progress_report(self):
        """İlerleme raporunu kaydet"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = {
            "timestamp": timestamp,
            "performance_history": self.performance_history,
            "current_performance": self.performance_history[-1]
        }
        
        print(f"\n📊 İlerleme Raporu ({timestamp})")
        print(f"Son Performans: {report['current_performance']:.4f}")
        print(f"Toplam İyileştirme: {self.performance_history[0] - report['current_performance']:.4f}")
        
        # Raporu JSON olarak kaydet
        import json
        with open(f"evolution_report_{timestamp}.json", "w") as f:
            json.dump(report, f, indent=4)
        print(f"✅ Rapor kaydedildi: evolution_report_{timestamp}.json")