"""Async file I/O helpers."""
import uuid
import aiofiles
from pathlib import Path
from fastapi import UploadFile
from app.utils.config import settings


async def save_upload(file: UploadFile, sub_dir: str = "") -> Path:
    """Save uploaded file to UPLOAD_DIR/<sub_dir>/<uuid>_<filename> and return path."""
    dest_dir: Path = settings.UPLOAD_DIR / sub_dir
    dest_dir.mkdir(parents=True, exist_ok=True)

    unique_name = f"{uuid.uuid4().hex}_{Path(file.filename).name}"
    dest_path = dest_dir / unique_name

    async with aiofiles.open(dest_path, "wb") as out:
        while chunk := await file.read(1024 * 256):  # 256 KB chunks
            await out.write(chunk)

    return dest_path


def cleanup_file(path: Path) -> None:
    """Remove a temporary uploaded file."""
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass
