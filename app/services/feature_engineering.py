"""Supplementary feature engineering shared across modalities."""
from __future__ import annotations
import numpy as np


def extract_voice_clinical_features(raw_features: np.ndarray) -> dict[str, float]:
    """
    Map raw spectrogram statistics to named clinical voice features.
    Replace with real MDVP / OpenSMILE features in production.
    """
    flat = raw_features.flatten()
    return {
        "jitter_pct": float(np.std(flat[:10]) * 100),
        "shimmer_db": float(np.mean(np.abs(flat[10:20]))),
        "hnr": float(np.mean(flat[20:30])),
        "rpde": float(np.random.uniform(0.4, 0.8)),  # placeholder
        "dfa": float(np.random.uniform(0.5, 0.9)),
        "spread1": float(np.min(flat)),
        "spread2": float(np.max(flat)),
        "ppe": float(np.random.uniform(0.1, 0.5)),
    }


def extract_mri_clinical_features(raw_features: np.ndarray) -> dict[str, float]:
    """Named MRI intensity / texture features."""
    flat = raw_features.flatten()
    return {
        "mean_intensity": float(np.mean(flat)),
        "std_intensity": float(np.std(flat)),
        "contrast": float(np.ptp(flat)),
        "skewness": float(_skewness(flat)),
        "kurtosis": float(_kurtosis(flat)),
    }


def _skewness(x: np.ndarray) -> float:
    m = x.mean()
    s = x.std() or 1e-8
    return float(np.mean(((x - m) / s) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    m = x.mean()
    s = x.std() or 1e-8
    return float(np.mean(((x - m) / s) ** 4) - 3)
