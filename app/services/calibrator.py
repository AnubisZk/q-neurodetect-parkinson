"""
app/services/calibrator.py
==========================
Phase 5 — Probability calibration.

İki yöntem:
  - Platt Scaling   (LogisticRegression — sigmoid eğrisi)
  - Isotonic        (monoton kalibre — daha esnek)

Kullanım:
  1. Eğitim zamanı: train_calibrator() → pkl kaydet
  2. Inference zamanı: calibrate_score() → kalibre edilmiş olasılık

Modeller mevcut değilse ham skoru olduğu gibi döndürür (graceful fallback).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

CALIBRATOR_DIR = Path("models/fusion")
CALIBRATOR_PATHS = {
    "voice":   CALIBRATOR_DIR / "calibrator_voice.pkl",
    "mri":     CALIBRATOR_DIR / "calibrator_mri.pkl",
    "drawing": CALIBRATOR_DIR / "calibrator_drawing.pkl",
    "fusion":  CALIBRATOR_DIR / "calibrator_fusion.pkl",
}

_cache: dict[str, object] = {}


# ── Yükleme ───────────────────────────────────────────────────────────────────
def load_calibrators() -> None:
    """Uygulama başlangıcında çağrılır."""
    import joblib
    for name, path in CALIBRATOR_PATHS.items():
        if path.exists():
            try:
                _cache[name] = joblib.load(path)
                logger.info("Calibrator yüklendi: %s", name)
            except Exception as exc:
                logger.warning("Calibrator yüklenemedi %s: %s", name, exc)


def calibrate_score(raw_score: float, modality: str = "fusion") -> float:
    """
    Ham olasılığı kalibre et.
    Calibrator yoksa ham skoru döndür.
    """
    cal = _cache.get(modality)
    if cal is None:
        return float(np.clip(raw_score, 0.0, 1.0))
    try:
        x = np.array([[raw_score]])
        prob = cal.predict_proba(x)[0][1]
        return float(np.clip(prob, 0.0, 1.0))
    except Exception as exc:
        logger.warning("Calibration hatası (%s): %s — ham skor kullanılıyor", modality, exc)
        return float(np.clip(raw_score, 0.0, 1.0))


def calibrators_loaded() -> dict[str, bool]:
    return {k: k in _cache for k in CALIBRATOR_PATHS}


# ── Eğitim (notebook'tan çağrılır) ───────────────────────────────────────────
def train_calibrator(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    modality: str = "fusion",
    method: Literal["platt", "isotonic"] = "platt",
    save: bool = True,
) -> object:
    """
    y_prob: (N,) ham model çıktı olasılıkları
    y_true: (N,) gerçek etiketler (0/1)
    method: "platt" (LogisticRegression) | "isotonic"
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    from sklearn.base import BaseEstimator, ClassifierMixin

    if method == "platt":
        # Platt scaling: sigmoid fit
        lr = LogisticRegression(C=1.0, solver="lbfgs")
        lr.fit(y_prob.reshape(-1, 1), y_true)
        calibrator = lr
    else:
        # Isotonic regression
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(y_prob, y_true)

        # Sklearn uyumlu wrapper
        class IsoWrapper(BaseEstimator, ClassifierMixin):
            def __init__(self, iso):
                self.iso = iso
                self.classes_ = np.array([0, 1])
            def fit(self, X, y):
                return self
            def predict_proba(self, X):
                p = self.iso.predict(X[:, 0])
                p = np.clip(p, 0, 1)
                return np.column_stack([1 - p, p])
            def predict(self, X):
                return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

        calibrator = IsoWrapper(iso)

    logger.info("Calibrator eğitildi: %s / %s", modality, method)

    if save:
        import joblib
        CALIBRATOR_DIR.mkdir(parents=True, exist_ok=True)
        path = CALIBRATOR_PATHS.get(modality, CALIBRATOR_DIR / f"calibrator_{modality}.pkl")
        joblib.dump(calibrator, path)
        logger.info("Calibrator kaydedildi: %s", path)

    _cache[modality] = calibrator
    return calibrator


# ── Reliability Diagram (görselleştirme) ──────────────────────────────────────
def plot_reliability(y_prob: np.ndarray, y_true: np.ndarray,
                     modality: str = "fusion", n_bins: int = 10):
    """Calibration curve — eğitimden sonra çağrılabilir."""
    try:
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve

        fig, ax = plt.subplots(figsize=(5, 4))
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        ax.plot(mean_pred, frac_pos, "s-", label="Model")
        ax.plot([0, 1], [0, 1], "k--", label="Mükemmel kalibrasyon")
        ax.set_xlabel("Tahmin edilen olasılık")
        ax.set_ylabel("Gerçek oran")
        ax.set_title(f"Reliability Diagram — {modality}")
        ax.legend()
        fig.tight_layout()
        out = CALIBRATOR_DIR / f"reliability_{modality}.png"
        CALIBRATOR_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120)
        plt.close()
        logger.info("Reliability diagram: %s", out)
        return str(out)
    except Exception as exc:
        logger.warning("Reliability diagram hatası: %s", exc)
        return None
