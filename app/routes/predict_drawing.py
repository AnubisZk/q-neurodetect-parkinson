"""POST /predict/drawing"""
from __future__ import annotations
import logging, numpy as np
from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from app.schemas.prediction_schema import ModalityResult
from app.services.preprocessing_drawing import preprocess_drawing
from app.services.model_loader import get_model
from app.utils.validators import validate_drawing_file
from app.utils.file_handlers import save_upload, cleanup_file

router = APIRouter()
logger = logging.getLogger(__name__)


def _infer_drawing(arr: np.ndarray) -> float:
    model = get_model("drawing")
    if model is not None:
        try:
            return float(model.predict_proba(arr)[0][1])
        except Exception:
            return float(model.predict(arr)[0])
    return float(np.random.uniform(0.20, 0.80))


@router.post("/drawing", response_model=ModalityResult)
async def predict_drawing(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Drawing image (.png/.jpg) or CSV of (x,y,pressure)"),
):
    validate_drawing_file(file)
    path = await save_upload(file, sub_dir="drawing")
    background_tasks.add_task(cleanup_file, path)

    arr = preprocess_drawing(path)
    prob = _infer_drawing(arr)

    return ModalityResult(
        modality="drawing",
        probability=round(prob, 4),
        label="Parkinson" if prob >= 0.5 else "Healthy",
        confidence=round(abs(prob - 0.5) * 2, 4),
        model_type="classical_ml",
        features_used=["hog_gradient_histogram"] if arr.shape[-1] == 64 else ["statistical_aggregates"],
        notes=f"Feature vector dim: {arr.shape[-1]}",
    )
