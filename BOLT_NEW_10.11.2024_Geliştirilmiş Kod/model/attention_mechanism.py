import tensorflow as tf
import numpy as np

class MultiHeadCrossAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    def split_heads(self, x):
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, query, key, value, mask=None, training=True):
        batch_size = tf.shape(query)[0]
        
        # Linear layers
        q = self.wq(query)
        k = self.wk(key)
        v = self.wv(value)
        
        # Split heads
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        # Scaled dot-product attention
        scaled_attention = tf.matmul(q, k, transpose_b=True)
        scaled_attention = scaled_attention / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        
        if mask is not None:
            scaled_attention += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scaled_attention, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)
        
        output = tf.matmul(attention_weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        
        # Final linear layer
        output = self.dense(output)
        output = self.dropout(output, training=training)
        output = self.layer_norm(output + query)
        
        return output, attention_weights