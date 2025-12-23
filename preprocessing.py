"""
Preprocessing utilities for Brain MRI tumor detection.

Pipeline (order matters):
1) Contouring to find brain region
2) Cropping to the largest contour
3) Resizing to a fixed target size
4) Normalization to 0-1
5) HOG feature extraction
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from skimage.feature import hog

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


@dataclass
class PreprocessResult:
    image_rgb: np.ndarray
    mask: np.ndarray
    cropped_rgb: np.ndarray
    resized_rgb: np.ndarray
    normalized_rgb: np.ndarray
    hog_features: np.ndarray
    hog_image: np.ndarray
    filepath: Path


def _read_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours[0]


def find_brain_mask(image_rgb: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)
    # Explicit 3x3 kernel to satisfy type checker and avoid implicit defaults
    kernel: np.ndarray = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.erode(thresh, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    contour = _largest_contour(mask)
    return mask, contour


def crop_to_contour(image_rgb: np.ndarray, contour: Optional[np.ndarray]) -> np.ndarray:
    if contour is None:
        # Fallback to center crop (square)
        h, w = image_rgb.shape[:2]
        side = min(h, w)
        start_x = (w - side) // 2
        start_y = (h - side) // 2
        return image_rgb[start_y : start_y + side, start_x : start_x + side]
    x, y, w, h = cv2.boundingRect(contour)
    return image_rgb[y : y + h, x : x + w]


def resize_image(image_rgb: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    return cv2.resize(image_rgb, target_size, interpolation=cv2.INTER_AREA)


def normalize_image(image_rgb: np.ndarray) -> np.ndarray:
    return image_rgb.astype("float32") / 255.0


def extract_hog_features(
    image_rgb: np.ndarray,
    pixels_per_cell: Tuple[int, int] = (8, 8),
    cells_per_block: Tuple[int, int] = (2, 2),
    orientations: int = 9,
) -> Tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor((image_rgb * 255).astype("uint8"), cv2.COLOR_RGB2GRAY)
    features, hog_image = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        visualize=True,
    )
    return features.astype("float32"), hog_image


def preprocess_image(
    image_path: str | Path,
    target_size: Tuple[int, int] = (224, 224),
    save_dir: Optional[str | Path] = None,
) -> PreprocessResult:
    path = Path(image_path)
    image_rgb = _read_image(path)
    mask, contour = find_brain_mask(image_rgb)
    cropped = crop_to_contour(image_rgb, contour)
    resized = resize_image(cropped, target_size)
    normalized = normalize_image(resized)
    hog_features, hog_image = extract_hog_features(normalized)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_dir / f"{path.stem}_original.png"), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(save_dir / f"{path.stem}_mask.png"), mask)
        cv2.imwrite(str(save_dir / f"{path.stem}_cropped.png"), cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(save_dir / f"{path.stem}_resized.png"), cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(save_dir / f"{path.stem}_hog.png"), hog_image)

    return PreprocessResult(
        image_rgb=image_rgb,
        mask=mask,
        cropped_rgb=cropped,
        resized_rgb=resized,
        normalized_rgb=normalized,
        hog_features=hog_features,
        hog_image=hog_image,
        filepath=path,
    )


def load_dataset(
    data_dir: str | Path,
    target_size: Tuple[int, int] = (224, 224),
    save_dir: Optional[str | Path] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Path]]:
    data_dir = Path(data_dir)
    images: List[np.ndarray] = []
    labels: List[int] = []
    paths: List[Path] = []
    for label_name, label_value in [("no", 0), ("yes", 1)]:
        class_dir = data_dir / label_name
        if not class_dir.exists():
            continue
        for img_path in sorted(class_dir.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            try:
                result = preprocess_image(img_path, target_size, save_dir)
                images.append(result.normalized_rgb)
                labels.append(label_value)
                paths.append(img_path)
            except Exception as exc:
                print(f"Skipping {img_path} due to error: {exc}")
    return np.array(images, dtype="float32"), np.array(labels, dtype="int64"), paths


def load_dataset_hog(
    data_dir: str | Path,
    target_size: Tuple[int, int] = (224, 224),
    save_dir: Optional[str | Path] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Path]]:
    data_dir = Path(data_dir)
    features: List[np.ndarray] = []
    labels: List[int] = []
    paths: List[Path] = []
    for label_name, label_value in [("no", 0), ("yes", 1)]:
        class_dir = data_dir / label_name
        if not class_dir.exists():
            continue
        for img_path in sorted(class_dir.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            try:
                result = preprocess_image(img_path, target_size, save_dir)
                features.append(result.hog_features)
                labels.append(label_value)
                paths.append(img_path)
            except Exception as exc:
                print(f"Skipping {img_path} due to error: {exc}")
    return np.array(features, dtype="float32"), np.array(labels, dtype="int64"), paths


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = [
    "PreprocessResult",
    "preprocess_image",
    "load_dataset",
    "load_dataset_hog",
    "ensure_dir",
]

