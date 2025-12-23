"""
Evaluate trained models and generate metrics/plots.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve

from preprocessing import load_dataset_hog


def evaluate_hog_model(
    data_dir: Path,
    model_path: Path,
    scaler_path: Path,
    output_dir: Path,
) -> Tuple[dict, np.ndarray, np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y, _ = load_dataset_hog(data_dir)
    scaler = joblib.load(scaler_path)
    X = scaler.transform(X)

    model = tf.keras.models.load_model(model_path)
    assert model is not None, "Loaded model is None"
    probs = model.predict(X).ravel()
    preds = (probs >= 0.5).astype(int)

    metrics = compute_metrics(y, preds, probs)
    save_plots(y, probs, preds, output_dir)

    with open(output_dir / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics, y, probs


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division="warn")
    roc_auc = roc_auc_score(y_true, y_prob)
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
    }


def save_plots(y_true, y_prob, y_pred, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Tumor", "Tumor"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png")
    plt.close(fig)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label="ROC")
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "roc_curve.png")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained HOG-based model.")
    parser.add_argument("--data_dir", type=Path, default=Path("data"), help="Dataset root")
    parser.add_argument("--model_path", type=Path, default=Path("models/best_mlp.h5"))
    parser.add_argument("--scaler_path", type=Path, default=Path("models/hog_scaler.pkl"))
    parser.add_argument("--output_dir", type=Path, default=Path("reports"))
    return parser.parse_args()


def main():
    args = parse_args()
    metrics, _, _ = evaluate_hog_model(args.data_dir, args.model_path, args.scaler_path, args.output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

