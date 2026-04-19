"""Input validation utilities."""
from pathlib import Path
from fastapi import UploadFile, HTTPException

ALLOWED_VOICE_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".csv"}
ALLOWED_MRI_EXTS = {".nii", ".nii.gz", ".png", ".jpg", ".jpeg", ".dcm"}
ALLOWED_DRAWING_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".csv"}
MAX_BYTES = 50 * 1024 * 1024  # 50 MB


def _check_size(file: UploadFile) -> None:
    # FastAPI does not expose size directly; validated after read
    pass


def validate_voice_file(file: UploadFile) -> None:
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_VOICE_EXTS:
        raise HTTPException(
            status_code=422,
            detail=f"Voice file must be one of {ALLOWED_VOICE_EXTS}, got '{ext}'",
        )


def validate_mri_file(file: UploadFile) -> None:
    name = file.filename.lower()
    ext = ".nii.gz" if name.endswith(".nii.gz") else Path(name).suffix
    if ext not in ALLOWED_MRI_EXTS:
        raise HTTPException(
            status_code=422,
            detail=f"MRI file must be one of {ALLOWED_MRI_EXTS}, got '{ext}'",
        )


def validate_drawing_file(file: UploadFile) -> None:
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_DRAWING_EXTS:
        raise HTTPException(
            status_code=422,
            detail=f"Drawing file must be one of {ALLOWED_DRAWING_EXTS}, got '{ext}'",
        )
