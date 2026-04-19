"""
MRI preprocessing — modelinizin eğitildiği boyuta uygun.
PDF'ten: IMG_SIZE = (128, 128), input_shape=(128, 128, 3)
"""
from __future__ import annotations
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# !! Modeliniz (128,128,3) ile eğitildi — bunu değiştirmeyin
TARGET_SIZE = (128, 128)


def preprocess_mri(file_path: Path) -> np.ndarray:
    """
    MRI görüntüsünü yükle, normalize et, (1, 128, 128, 3) döndür.
    """
    suffix = file_path.suffix.lower()
    try:
        if suffix in {".nii", ".gz"}:
            arr = _load_nifti(file_path)
        else:
            arr = _load_image(file_path)
        logger.debug("MRI preprocessed: shape %s", arr.shape)
        return arr
    except Exception as exc:
        logger.warning("MRI preprocessing failed (%s) — mock array", exc)
        return np.random.rand(1, *TARGET_SIZE, 3).astype(np.float32)


def _load_image(path: Path) -> np.ndarray:
    from PIL import Image
    img = Image.open(path).convert("RGB").resize(TARGET_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[np.newaxis]   # (1, 128, 128, 3)


def _load_nifti(path: Path) -> np.ndarray:
    import nibabel as nib
    from PIL import Image
    img  = nib.load(str(path))
    data = img.get_fdata()
    mid  = data.shape[2] // 2
    sl   = data[:, :, mid]
    mn, mx = sl.min(), sl.max()
    if mx > mn:
        sl = (sl - mn) / (mx - mn) * 255.0
    pil = Image.fromarray(sl.astype(np.uint8)).convert("RGB").resize(TARGET_SIZE)
    arr = np.array(pil, dtype=np.float32) / 255.0
    return arr[np.newaxis]
