"""
Training script for Brain MRI tumor detection.
Includes baseline MLP on HOG features, optional SVM, and simple CNN.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # pyright: ignore[reportMissingImports]

import augmentation
from model_architectures import create_mlp, create_simple_cnn
from preprocessing import ensure_dir, load_dataset, load_dataset_hog

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def train_hog_mlp(
    data_dir: Path,
    model_dir: Path,
    epochs: int = 25,
    batch_size: int = 32,
) -> Dict:
    model_dir = ensure_dir(model_dir)
    features, labels, _ = load_dataset_hog(data_dir)
    X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.15, random_state=SEED, stratify=labels)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    model = create_mlp(input_shape=(X_train.shape[1],))

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss"),
        # Cast to str because Keras expects string paths, not pathlib.Path
        ModelCheckpoint(str(model_dir / "best_mlp.h5"), monitor="val_loss", save_best_only=True),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose="auto",
        callbacks=callbacks,
    )
    assert history is not None, "Model fit returned None"

    test_probs = model.predict(X_test).ravel()
    test_preds = (test_probs >= 0.5).astype(int)
    test_acc = accuracy_score(y_test, test_preds)

    joblib.dump(scaler, model_dir / "hog_scaler.pkl")
    model.save(str(model_dir / "final_mlp.h5"))

    metrics = {
        "val_best_loss": float(np.min(history.history["val_loss"])),
        "val_best_auc": float(np.max(history.history.get("val_auc", [0]))),
        "test_accuracy": float(test_acc),
    }
    with open(model_dir / "mlp_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def train_hog_svm(
    data_dir: Path,
    model_dir: Path,
    kernel: str = "rbf",
    C: float = 5.0,
    gamma: str = "scale",
) -> Dict:
    model_dir = ensure_dir(model_dir)
    features, labels, _ = load_dataset_hog(data_dir)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=SEED, stratify=labels)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    svm = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, class_weight="balanced", random_state=SEED)
    svm.fit(X_train, y_train)
    preds = svm.predict(X_test)
    probs = svm.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, preds)

    joblib.dump(scaler, model_dir / "hog_scaler_svm.pkl")
    joblib.dump(svm, model_dir / "svm_model.pkl")

    metrics = {"test_accuracy": float(test_acc)}
    with open(model_dir / "svm_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def train_cnn(
    data_dir: Path,
    model_dir: Path,
    epochs: int = 20,
    batch_size: int = 16,
    target_size: Tuple[int, int] = (224, 224),
) -> Dict:
    model_dir = ensure_dir(model_dir)
    train_gen, val_gen = augmentation.create_generators(data_dir, target_size=target_size, batch_size=batch_size)

    model = create_simple_cnn(input_shape=(*target_size, 3))
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss"),
        ModelCheckpoint(str(model_dir / "best_cnn.h5"), monitor="val_loss", save_best_only=True),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6),
    ]

    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)
    assert history is not None, "Model fit returned None"
    model.save(str(model_dir / "final_cnn.h5"))

    metrics = {
        "val_best_loss": float(np.min(history.history["val_loss"])),
        "val_best_auc": float(np.max(history.history.get("val_auc", [0]))),
        "val_best_acc": float(np.max(history.history.get("val_accuracy", [0]))),
    }
    with open(model_dir / "cnn_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train brain tumor detection models.")
    parser.add_argument("--data_dir", type=Path, default=Path("data"), help="Dataset root with yes/ and no/ folders")
    parser.add_argument("--model_dir", type=Path, default=Path("models"), help="Output directory for models")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_cnn", action="store_true", help="Also train the CNN model")
    parser.add_argument("--train_svm", action="store_true", help="Also train the SVM model")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.model_dir)
    print("Training MLP on HOG features...")
    mlp_metrics = train_hog_mlp(args.data_dir, args.model_dir, epochs=args.epochs, batch_size=args.batch_size)
    print(f"MLP metrics: {mlp_metrics}")

    if args.train_svm:
        print("Training SVM on HOG features...")
        svm_metrics = train_hog_svm(args.data_dir, args.model_dir)
        print(f"SVM metrics: {svm_metrics}")

    if args.train_cnn:
        print("Training CNN with augmentation...")
        cnn_metrics = train_cnn(args.data_dir, args.model_dir, epochs=args.epochs, batch_size=args.batch_size)
        print(f"CNN metrics: {cnn_metrics}")


if __name__ == "__main__":
    main()

