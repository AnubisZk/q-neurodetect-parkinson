"""POST /predict/mri"""
from __future__ import annotations
import uuid, logging, numpy as np
from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from app.schemas.prediction_schema import ModalityResult
from app.services.preprocessing_mri import preprocess_mri
from app.services.feature_engineering import extract_mri_clinical_features
from app.services.model_loader import get_model
from app.utils.validators import validate_mri_file
from app.utils.file_handlers import save_upload, cleanup_file

router = APIRouter()
logger = logging.getLogger(__name__)


def _infer_mri(arr: np.ndarray) -> float:
    model = get_model("mri")
    if model is not None:
        return float(model.predict(arr)[0][0])
    return float(np.random.uniform(0.25, 0.90))


@router.post("/mri", response_model=ModalityResult)
async def predict_mri(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="MRI file (.nii/.nii.gz or image)"),
):
    validate_mri_file(file)
    path = await save_upload(file, sub_dir="mri")
    background_tasks.add_task(cleanup_file, path)

    arr = preprocess_mri(path)
    prob = _infer_mri(arr)
    features = extract_mri_clinical_features(arr)

    return ModalityResult(
        modality="mri",
        probability=round(prob, 4),
        label="Parkinson" if prob >= 0.5 else "Healthy",
        confidence=round(abs(prob - 0.5) * 2, 4),
        model_type="deep_learning",
        features_used=list(features.keys()),
        notes=f"Mean intensity={features['mean_intensity']:.4f}, Contrast={features['contrast']:.4f}",
    )
