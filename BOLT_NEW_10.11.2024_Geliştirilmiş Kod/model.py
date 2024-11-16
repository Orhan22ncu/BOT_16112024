import tensorflow as tf
import numpy as np
from datetime import datetime
import logging
from tqdm import tqdm
from model import SelfEvolutionEngine

# Loglama ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

class DeepLearningModel:
    def __init__(self, input_shape):
        self.model = self._build_model(input_shape)
        self.evolution_engine = SelfEvolutionEngine(self.model)
    
    def _build_model(self, input_shape):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu',
                                 input_shape=input_shape),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.Huber(),
            metrics=['mae', 'mape']
        )
        return model

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100):
        """Model eğitimi"""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='models/model.keras',
                save_best_only=True,
                monitor='val_loss'
            )
        ]

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        # Her 10 epoch'ta bir evrim kontrolü
        if epochs % 10 == 0:
            self.evolution_engine.evolve(X_train, y_train)

        return history

if __name__ == "__main__":
    try:
        print("🚀 Kripto Trading Bot Başlatılıyor...")
        
        API_KEY = "N7VUikmN3cGw7Lf2Pr4TYstS8L1PFkMFuK8cBWGIaidh1pxL4oZECDEhnLBrV3QG"
        API_SECRET = "9DyevXOspE4PStVm341SheIWGRNJk5G0w61miJC41IYG8YQeK0SXIjeTyfOJvBA5"
        
        # Örnek veri oluştur
        X = np.random.random((1000, 60, 10))  # (örnekler, zaman_adımları, özellikler)
        y = np.random.random((1000, 1))       # hedef değerler
        
        # Model oluştur ve eğit
        model = DeepLearningModel(input_shape=(60, 10))
        history = model.train(X, y, epochs=100)
        
        print("\n✅ İşlem başarıyla tamamlandı!")
        
    except Exception as e:
        logging.error(f"Ana program hatası: {str(e)}")
        print(f"\n❌ Hata: {str(e)}")