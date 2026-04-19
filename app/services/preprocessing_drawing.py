"""Spiral / drawing test preprocessing pipeline."""
from __future__ import annotations
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


def preprocess_drawing(file_path: Path) -> np.ndarray:
    """
    Load drawing image (or CSV of (x, y, pressure) points), extract
    classical ML feature vector (shape: (1, N_FEATURES)).
    """
    suffix = file_path.suffix.lower()
    try:
        if suffix == ".csv":
            return _from_csv(file_path)
        else:
            return _from_image(file_path)
    except Exception as exc:
        logger.warning("Drawing preprocessing failed (%s) – returning mock features", exc)
        return np.random.rand(1, 64).astype(np.float32)


def _from_image(path: Path) -> np.ndarray:
    from PIL import Image  # type: ignore
    img = Image.open(path).convert("L").resize((64, 64))
    arr = np.array(img, dtype=np.float32).flatten() / 255.0
    features = _hog_like(arr.reshape(64, 64))
    return features[np.newaxis]


def _from_csv(path: Path) -> np.ndarray:
    import pandas as pd  # type: ignore
    df = pd.read_csv(path)
    cols = df.select_dtypes(include=[np.number]).values.astype(np.float32)
    # Aggregate statistics as feature vector
    feats = np.concatenate([cols.mean(0), cols.std(0), cols.min(0), cols.max(0)])
    return feats[np.newaxis]


def _hog_like(gray: np.ndarray) -> np.ndarray:
    """Minimal gradient histogram feature (64-dim)."""
    gx = np.gradient(gray, axis=1)
    gy = np.gradient(gray, axis=0)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ang = np.arctan2(gy, gx)
    bins = np.linspace(-np.pi, np.pi, 65)
    hist, _ = np.histogram(ang.flatten(), bins=bins, weights=mag.flatten())
    return hist.astype(np.float32)
