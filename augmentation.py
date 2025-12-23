"""
Keras ImageDataGenerator helpers for augmentation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

from tensorflow.keras.preprocessing.image import ImageDataGenerator  # pyright: ignore[reportMissingImports]


def build_train_datagen() -> ImageDataGenerator:
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
        rescale=1.0 / 255.0,
    )


def build_val_datagen() -> ImageDataGenerator:
    return ImageDataGenerator(rescale=1.0 / 255.0)


def create_generators(
    data_dir: str | Path,
    target_size: Tuple[int, int] = (224, 224),
    batch_size: int = 16,
    validation_split: float = 0.15,
):
    data_dir = Path(data_dir)
    train_datagen = build_train_datagen()
    val_datagen = build_val_datagen()

    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="training",
        shuffle=True,
        seed=42,
        validation_split=validation_split,
    )
    val_gen = val_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="validation",
        shuffle=False,
        seed=42,
        validation_split=validation_split,
    )
    return train_gen, val_gen


__all__ = ["build_train_datagen", "build_val_datagen", "create_generators"]

