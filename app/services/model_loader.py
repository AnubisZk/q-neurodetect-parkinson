"""Lazy model loader – .h5, .pkl ve QSVM bundle destekler."""
from __future__ import annotations
import logging
import pickle
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)
_registry: dict[str, Any] = {}


def _load_h5(path: Path) -> Optional[Any]:
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(str(path))
        logger.info("Loaded Keras model: %s", path.name)
        return model
    except Exception as exc:
        logger.warning("Could not load .h5 at %s: %s (mock mode)", path.name, exc)
        return None


def _load_pkl(path: Path) -> Optional[Any]:
    try:
        import joblib
        model = joblib.load(path)
        logger.info("Loaded pkl model: %s", path.name)
        return model
    except Exception:
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
            logger.info("Loaded pickle model: %s", path.name)
            return model
        except Exception as exc:
            logger.warning("Could not load .pkl at %s: %s (mock mode)", path.name, exc)
            return None


def load_all_models(cfg) -> None:
    # Ses: .pkl tercih edilir (sklearn pipeline), .h5 fallback
    voice_pkl = cfg.MODELS_DIR / "voice" / "voice_model.pkl"
    if voice_pkl.exists():
        _registry["voice"] = _load_pkl(voice_pkl)
    else:
        _registry["voice"] = _load_h5(cfg.VOICE_MODEL_PATH)

    # MRI: .h5 keras modeli
    _registry["mri"] = _load_h5(cfg.MRI_MODEL_PATH)

    # Çizim: sklearn pipeline
    _registry["drawing"] = _load_pkl(cfg.DRAWING_MODEL_PATH)

    # Fusion: sklearn meta-classifier
    _registry["fusion"] = _load_pkl(cfg.FUSION_MODEL_PATH)

    # QSVM bundle
    qsvm_path = cfg.MODELS_DIR / "voice" / "qsvm_model.pkl"
    _registry["qsvm"] = _load_pkl(qsvm_path) if qsvm_path.exists() else None


def get_model(name: str) -> Optional[Any]:
    return _registry.get(name)


def models_loaded_status() -> dict[str, bool]:
    return {k: v is not None for k, v in _registry.items()}
