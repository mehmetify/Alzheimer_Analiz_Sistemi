import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def build_3d_cnn(input_shape):
    """3D CNN modeli - MR görüntü analizi için."""
    inputs = layers.Input(input_shape)
    
    x = layers.Conv3D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPool3D(2)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv3D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPool3D(2)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv3D(128, 3, activation="relu", padding="same")(x)
    x = layers.MaxPool3D(2)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    model = models.Model(inputs, outputs, name="3D_CNN")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def get_ml_model():
    """Random Forest modeli - klinik veri analizi için."""
    return RandomForestClassifier(n_estimators=100, random_state=42)

class HybridModel:
    """Hibrit model - CNN ve RF sonuçlarını birleştirir."""
    
    def __init__(self, cnn_model, ml_model):
        self.cnn = cnn_model
        self.ml = ml_model
        
    def predict(self, X_img, X_tab):
        p_cnn = self.cnn.predict(X_img, verbose=0).flatten()
        p_ml = self.ml.predict_proba(X_tab)[:, 1]
        
        # %40 Görüntü + %60 Klinik ağırlık
        final_prob = (0.4 * p_cnn) + (0.6 * p_ml)
        return (final_prob > 0.5).astype(int)