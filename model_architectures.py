"""
Model definitions for baseline MLP, simple CNN, and utilities.
"""
from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers  # pyright: ignore[reportMissingImports]


def create_mlp(input_shape: Tuple[int, ...], l2_reg: float = 1e-4, dropout: float = 0.4) -> tf.keras.Model:
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(l2_reg)),
            layers.Dropout(dropout),
            layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_reg)),
            layers.Dropout(dropout),
            layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2_reg)),
            layers.Dropout(dropout),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return model


def create_simple_cnn(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    l2_reg: float = 1e-4,
    dropout: float = 0.4,
) -> tf.keras.Model:
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(l2_reg)),
            layers.Dropout(dropout),
            layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2_reg)),
            layers.Dropout(dropout),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return model


__all__ = ["create_mlp", "create_simple_cnn"]

