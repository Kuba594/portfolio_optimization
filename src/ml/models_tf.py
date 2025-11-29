# src/ml/models_tf.py
import tensorflow as tf
from tensorflow.keras import layers, models

def build_univariate_lstm(window: int, hidden: int = 32, lr: float = 1e-3):
    model = models.Sequential([
        layers.Input(shape=(window, 1)),
        layers.LSTM(hidden),
        layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse')
    return model

def build_univariate_gru(window: int, hidden: int = 32, lr: float = 1e-3):
    model = models.Sequential([
        layers.Input(shape=(window, 1)),
        layers.GRU(hidden),
        layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse')
    return model
