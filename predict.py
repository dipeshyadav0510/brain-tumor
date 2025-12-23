"""
Predict tumor presence on a single MRI image using the trained MLP (HOG features).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import tensorflow as tf

from preprocessing import preprocess_image

LABELS = {0: "No Tumor Detected", 1: "Tumor Detected"}


def load_artifacts(model_path: Path, scaler_path: Path):
    model = tf.keras.models.load_model(model_path)
    assert model is not None, "Loaded model is None"
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_image(
    image_path: Path,
    model_path: Path = Path("models/best_mlp.h5"),
    scaler_path: Path = Path("models/hog_scaler.pkl"),
) -> Tuple[int, float]:
    model, scaler = load_artifacts(model_path, scaler_path)
    processed = preprocess_image(image_path)
    features = scaler.transform(processed.hog_features.reshape(1, -1))
    prob = float(model.predict(features)[0][0])
    label = int(prob >= 0.5)
    return label, prob


def parse_args():
    parser = argparse.ArgumentParser(description="Predict tumor presence from MRI image.")
    parser.add_argument("--image", type=Path, required=True, help="Path to MRI image")
    parser.add_argument("--model_path", type=Path, default=Path("models/best_mlp.h5"))
    parser.add_argument("--scaler_path", type=Path, default=Path("models/hog_scaler.pkl"))
    return parser.parse_args()


def main():
    args = parse_args()
    label, prob = predict_image(args.image, args.model_path, args.scaler_path)
    print(f"Prediction: {LABELS[label]} (confidence: {prob:.2%})")


if __name__ == "__main__":
    main()

