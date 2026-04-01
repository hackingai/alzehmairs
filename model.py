"""
Alzheimer's classification model.
Architecture: VGG16 (transfer learning) + CNN feature extractor + LSTM temporal layer.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16

NUM_CLASSES = 4
IMG_SIZE = (128, 128, 3)


def build_vgg16_cnn_lstm(num_classes: int = NUM_CLASSES) -> Model:
    """
    VGG16 backbone with added CNN refinement and LSTM for sequential feature modeling.
    Input: single MRI image (224, 224, 3)
    """
    # --- VGG16 backbone (frozen base) ---
    base = VGG16(weights="imagenet", include_top=False, input_shape=IMG_SIZE)
    for layer in base.layers:  # freeze entire VGG16
        layer.trainable = False

    x = base.output  # (7, 7, 512)

    # --- Additional CNN refinement ---
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    # --- Reshape for LSTM: treat spatial rows as time steps ---
    # (7, 7, 128) -> (7, 7*128) = (7, 896)
    shape = x.shape
    x = layers.Reshape((shape[1], shape[2] * shape[3]))(x)

    # --- LSTM layers ---
    x = layers.LSTM(256, return_sequences=True, dropout=0.3)(x)
    x = layers.LSTM(128, dropout=0.3)(x)

    # --- Classification head ---
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=output, name="VGG16_CNN_LSTM_AD")
    return model


if __name__ == "__main__":
    m = build_vgg16_cnn_lstm()
    m.summary()
