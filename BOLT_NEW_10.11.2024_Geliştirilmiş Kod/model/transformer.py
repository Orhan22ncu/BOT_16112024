import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

class TransformerBlock:
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="gelu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

class AdvancedTradingModel:
    def __init__(self, sequence_length, num_features):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.model = self.build_model()

    def build_model(self):
        inputs = Input(shape=(self.sequence_length, self.num_features))
        
        # Transformer Block
        x = TransformerBlock(embed_dim=32, num_heads=4, ff_dim=32)(inputs)
        
        # Hybrid Architecture
        x1 = LSTM(64, return_sequences=True)(x)
        x2 = GRU(64, return_sequences=True)(x)
        
        # Combine LSTM and GRU
        combined = Concatenate()([x1, x2])
        
        # Attention mechanism
        attention = Attention()([combined, combined])
        
        x = GlobalAveragePooling1D()(attention)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='huber_loss',
            metrics=['mae', 'mape']
        )
        
        return model