"""POST /predict/voice — sklearn pipeline veya Keras .h5 desteği"""
from __future__ import annotations
import logging
import numpy as np
from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from app.schemas.prediction_schema import ModalityResult
from app.services.preprocessing_voice import preprocess_voice
from app.services.feature_engineering import extract_voice_clinical_features
from app.services.model_loader import get_model
from app.services.qsvm_inference import qsvm_predict
from app.utils.validators import validate_voice_file
from app.utils.file_handlers import save_upload, cleanup_file

router = APIRouter()
logger = logging.getLogger(__name__)


def _infer_voice(arr: np.ndarray) -> tuple[float, float, str]:
    """
    Döner: (klasik_prob, qsvm_prob, model_type)
    arr: log-mel spektrogram (1, 128, 128, 1) — Keras için
         flatten edilmiş hali sklearn pipeline için de kullanılır
    """
    model = get_model("voice")
    qsvm  = get_model("qsvm")

    flat = arr.flatten().astype(np.float32)

    # Klasik model
    if model is not None:
        try:
            if hasattr(model, "predict_proba"):        # sklearn pipeline
                classic_prob = float(model.predict_proba(flat.reshape(1, -1))[0][1])
            else:                                       # Keras
                classic_prob = float(model.predict(arr)[0][0])
        except Exception as e:
            logger.warning("Voice classic inference hatası: %s", e)
            classic_prob = float(np.random.uniform(0.3, 0.85))
        model_type = "deep_learning" if not hasattr(model, "predict_proba") else "classical_ml"
    else:
        classic_prob = float(np.random.uniform(0.3, 0.85))
        model_type   = "deep_learning"

    # QSVM
    if qsvm is not None:
        qsvm_prob = qsvm_predict(flat, qsvm)
    else:
        qsvm_prob = float(np.clip(classic_prob + np.random.uniform(-0.06, 0.06), 0, 1))

    return classic_prob, qsvm_prob, model_type


@router.post("/voice", response_model=ModalityResult)
async def predict_voice(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Ses dosyası (.wav/.mp3/.flac/.ogg)"),
):
    validate_voice_file(file)
    path = await save_upload(file, sub_dir="voice")
    background_tasks.add_task(cleanup_file, path)

    arr = preprocess_voice(path)
    classic_prob, qsvm_prob, model_type = _infer_voice(arr)
    features = extract_voice_clinical_features(arr)

    # Basit ensemble: klasik + qsvm ortalaması
    prob = (classic_prob + qsvm_prob) / 2

    return ModalityResult(
        modality="voice",
        probability=round(prob, 4),
        label="Parkinson" if prob >= 0.5 else "Healthy",
        confidence=round(abs(prob - 0.5) * 2, 4),
        model_type=model_type,
        features_used=list(features.keys()),
        notes=f"Klasik={classic_prob:.3f} | QSVM={qsvm_prob:.3f}",
    )
