"""Q-NeuroDetect Parkinson – FastAPI entry point."""
from __future__ import annotations
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.utils.config import settings
from app.services.model_loader import load_all_models, models_loaded_status
from app.services.calibrator import load_calibrators
from app.schemas.prediction_schema import HealthResponse
from app.routes import predict_voice, predict_mri, predict_drawing, predict_fusion

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading models…")
    load_all_models(settings)
    load_calibrators()
    logger.info("Models status: %s", models_loaded_status())
    yield
    logger.info("Shutting down Q-NeuroDetect.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description=(
        "Çok modlu Parkinson karar destek sistemi. "
        "Ses, MRI ve çizim verilerini analiz eder ve weighted/meta-classifier füzyonuyla risk skoru döndürür."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for generated reports
reports_dir = settings.BASE_DIR / "data" / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)
app.mount("/reports", StaticFiles(directory=str(reports_dir)), name="reports")

# Routers
app.include_router(predict_voice.router, prefix="/predict", tags=["Prediction"])
app.include_router(predict_mri.router, prefix="/predict", tags=["Prediction"])
app.include_router(predict_drawing.router, prefix="/predict", tags=["Prediction"])
app.include_router(predict_fusion.router, prefix="/predict", tags=["Prediction"])


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(
        status="ok",
        version=settings.VERSION,
        models_loaded=models_loaded_status(),
    )
