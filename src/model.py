"""
Model architectures for Deep-NIDS-TF.

- build_classifier: supervised MLP that labels traffic by attack type.
- build_autoencoder: trained only on benign traffic; flags high
  reconstruction error at inference time as a potential unknown/novel attack.

Both are deliberately small/shallow: tabular flow-level features (already
hand-engineered numeric columns) rarely benefit from very deep networks,
and a compact model keeps inference latency low, which matters far more
than model depth for a real-time NIDS.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_classifier(input_dim: int, num_classes: int, l2_reg: float = 1e-4) -> tf.keras.Model:
    inputs = layers.Input(shape=(input_dim,), name="flow_features")
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="attack_class")(x)

    model = models.Model(inputs, outputs, name="nids_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_autoencoder(input_dim: int, latent_dim: int = 8) -> tf.keras.Model:
    inputs = layers.Input(shape=(input_dim,), name="flow_features")

    # Encoder
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dense(32, activation="relu")(x)
    latent = layers.Dense(latent_dim, activation="relu", name="latent")(x)

    # Decoder
    x = layers.Dense(32, activation="relu")(latent)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(input_dim, activation="linear", name="reconstruction")(x)

    model = models.Model(inputs, outputs, name="nids_autoencoder")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model
