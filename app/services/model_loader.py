"""
Model Loader — Hugging Face Hub entegrasyonu
Railway deployment için modelleri HF'den çeker.
"""
from __future__ import annotations
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

HF_REPO = "AnubisZk/q-neurodetect-models"
MODELS_DIR = Path("models")
_models: dict = {}


def load_all_models(settings=None):
    try:
        from huggingface_hub import hf_hub_download
        logger.info("HF Hub'dan modeller indiriliyor...")

        # Voice
        try:
            path = hf_hub_download(HF_REPO, "voice_model.pkl")
            import joblib
            _models["voice"] = joblib.load(path)
            logger.info("✅ voice_model yüklendi")
        except Exception as e:
            logger.warning("voice_model yüklenemedi: %s", e)

        # MRI
        try:
            path = hf_hub_download(HF_REPO, "mri_model.h5")
            import tensorflow as tf
            _models["mri"] = tf.keras.models.load_model(path)
            logger.info("✅ mri_model yüklendi")
        except Exception as e:
            logger.warning("mri_model yüklenemedi: %s", e)

        # Drawing
        try:
            path = hf_hub_download(HF_REPO, "drawing_model.pkl")
            import joblib
            _models["drawing"] = joblib.load(path)
            logger.info("✅ drawing_model yüklendi")
        except Exception as e:
            logger.warning("drawing_model yüklenemedi: %s", e)

    except ImportError:
        logger.warning("huggingface_hub yok — local modeller deneniyor")
        _load_local()


def _load_local():
    """Local models/ klasöründen yükle (fallback)."""
    import joblib

    voice_pkl = MODELS_DIR / "voice" / "voice_model.pkl"
    mri_h5    = MODELS_DIR / "mri" / "mri_model.h5"
    draw_pkl  = MODELS_DIR / "drawing" / "drawing_model.pkl"

    if voice_pkl.exists():
        _models["voice"] = joblib.load(voice_pkl)
        logger.info("✅ voice_model (local)")
    if mri_h5.exists():
        import tensorflow as tf
        _models["mri"] = tf.keras.models.load_model(str(mri_h5))
        logger.info("✅ mri_model (local)")
    if draw_pkl.exists():
        _models["drawing"] = joblib.load(draw_pkl)
        logger.info("✅ drawing_model (local)")


def get_model(modality: str):
    return _models.get(modality)


def models_loaded_status() -> dict[str, bool]:
    return {
        "voice":   "voice"   in _models,
        "mri":     "mri"     in _models,
        "drawing": "drawing" in _models,
        "fusion":  "fusion"  in _models,
        "qsvm":    "qsvm"    in _models,
    }
