import sys
import logging
import numpy as np
import pandas as pd
from binance.client import Client
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_environment():
    checks = {
        "Python Versiyonu": check_python_version(),
        "Gerekli Kütüphaneler": check_required_libraries(),
        "GPU Kullanılabilirliği": check_gpu_availability(),
        "Binance API": check_binance_api(),
        "Model Dosyaları": check_model_files()
    }
    
    all_passed = all(checks.values())
    
    if all_passed:
        logger.info("✅ Tüm sistem kontrolleri başarılı!")
    else:
        logger.error("❌ Bazı kontroller başarısız!")
        for check, status in checks.items():
            logger.info(f"{check}: {'✅' if status else '❌'}")
    
    return all_passed

def check_python_version():
    required_version = (3, 7)
    current_version = sys.version_info[:2]
    
    if current_version >= required_version:
        logger.info(f"✅ Python versiyonu uyumlu: {sys.version}")
        return True
    else:
        logger.error(f"❌ Python versiyonu uyumsuz. Gerekli: 3.7+, Mevcut: {sys.version}")
        return False

def check_required_libraries():
    required_libraries = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'tensorflow': 'tensorflow',
        'scikit-learn': 'sklearn',
        'python-binance': 'binance'
    }
    
    missing_libraries = []
    
    for lib, import_name in required_libraries.items():
        try:
            __import__(import_name)
            logger.info(f"✅ {lib} yüklü")
        except ImportError:
            missing_libraries.append(lib)
            logger.error(f"❌ {lib} eksik")
    
    return len(missing_libraries) == 0

def check_gpu_availability():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"✅ GPU bulundu: {len(gpus)} adet")
            return True
        else:
            logger.warning("⚠️ GPU bulunamadı, CPU kullanılacak")
            return True
    except:
        logger.error("❌ GPU kontrolünde hata")
        return False

def check_binance_api():
    try:
        client = Client("test", "test")
        client.get_server_time()
        logger.info("✅ Binance API bağlantısı başarılı")
        return True
    except Exception as e:
        logger.error(f"❌ Binance API bağlantı hatası: {e}")
        return False

def check_model_files():
    required_files = [
        'bch_model.h5',
        'trading_agent.h5'
    ]
    
    import os
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
            logger.error(f"❌ {file} bulunamadı")
        else:
            logger.info(f"✅ {file} mevcut")
    
    return len(missing_files) == 0

def test_model_predictions():
    """Model tahminlerini test et"""
    try:
        from model import DeepLearningModel
        from feature_engineering import FeatureEngineer
        
        # Test verisi oluştur
        test_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='1min'),
            'open': np.random.random(100),
            'high': np.random.random(100),
            'low': np.random.random(100),
            'close': np.random.random(100),
            'volume': np.random.random(100)
        })
        
        # Özellikleri hazırla
        feature_engineer = FeatureEngineer()
        features = feature_engineer.create_features(test_data)
        
        # Model yükle ve tahmin yap
        model = DeepLearningModel.load_model('bch_model.h5')
        predictions = model.predict(features)
        
        logger.info("✅ Model tahmin testi başarılı")
        return True
    except Exception as e:
        logger.error(f"❌ Model tahmin testi başarısız: {e}")
        return False

if __name__ == "__main__":
    logger.info("🔄 Sistem kontrolleri başlatılıyor...")
    
    if check_environment():
        logger.info("🔄 Model tahmin testi başlatılıyor...")
        if test_model_predictions():
            logger.info("✅ Tüm testler başarılı! Bot başlatılabilir.")
        else:
            logger.error("❌ Model testi başarısız! Lütfen modeli kontrol edin.")
    else:
        logger.error("❌ Sistem kontrolleri başarısız! Lütfen hataları düzeltin.")