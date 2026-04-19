"""Central configuration for Q-NeuroDetect Parkinson."""
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "Q-NeuroDetect Parkinson"
    VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Model paths
    BASE_DIR: Path = Path(__file__).resolve().parents[3]
    MODELS_DIR: Path = BASE_DIR / "models"
    VOICE_MODEL_PATH: Path = MODELS_DIR / "voice" / "voice_model.h5"
    MRI_MODEL_PATH: Path = MODELS_DIR / "mri" / "mri_model.h5"
    DRAWING_MODEL_PATH: Path = MODELS_DIR / "drawing" / "drawing_model.pkl"
    FUSION_MODEL_PATH: Path = MODELS_DIR / "fusion" / "fusion_model.pkl"

    # Upload
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
    MAX_FILE_SIZE_MB: int = 50

    # Fusion weights (voice, mri, drawing)
    FUSION_WEIGHTS: list[float] = [0.35, 0.40, 0.25]

    # Risk thresholds
    HIGH_RISK_THRESHOLD: float = 0.70
    MODERATE_RISK_THRESHOLD: float = 0.40

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
