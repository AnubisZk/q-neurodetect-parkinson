"""
Ses preprocessing — iki mod desteklenir:

MOD A: UCI/MDVP CSV dosyası yükleme (PDF'teki modelinizle uyumlu)
  - parkinsons.csv gibi dosyalar
  - 22 MDVP özelliği içeren CSV
  - Gradient Boosting modeliyle kullanılır

MOD B: WAV dosyası → librosa MFCC (gelecek ses modeli için)
  - Model yoksa mock döner

Sistem hangi dosyanın yüklendiğini uzantıdan otomatik anlar.
"""
from __future__ import annotations
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# UCI Parkinson veri setindeki MDVP özellik kolonları (name ve status hariç)
MDVP_COLUMNS = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ",
    "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
    "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
]


def preprocess_voice(file_path: Path) -> np.ndarray:
    """
    Dosya tipine göre otomatik mod seç.
    CSV → MDVP özellik vektörü (1, 22)
    WAV → librosa MFCC (1, 193) veya mock
    """
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return _from_csv(file_path)
    else:
        return _from_wav(file_path)


def _from_csv(path: Path) -> np.ndarray:
    """
    UCI formatı CSV → (1, 22) float32 vektör.
    Tek satır veya çok satır varsa ilk kaydı kullanır.
    name ve status kolonları varsa otomatik düşer.
    """
    import pandas as pd
    df = pd.read_csv(path)

    # name ve status kolonlarını düş
    drop_cols = [c for c in df.columns if c.lower() in ("name", "status")]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Sadece sayısal kolonlar
    df = df.select_dtypes(include=[np.number])

    if df.empty:
        logger.warning("CSV boş veya sayısal kolon yok — mock döndürülüyor")
        return np.random.rand(1, 22).astype(np.float32)

    # İlk satırı al (tek kayıt bekleniyor)
    arr = df.iloc[0].values.astype(np.float32).reshape(1, -1)
    logger.debug("CSV ses özellikleri: shape %s", arr.shape)
    return arr


def _from_wav(path: Path) -> np.ndarray:
    """WAV → librosa MFCC (1, 193)"""
    try:
        import librosa
        y, sr = librosa.load(str(path), sr=22050, mono=True, duration=10.0)
        y, _  = librosa.effects.trim(y, top_db=20)
        if len(y) == 0:
            return np.zeros((1, 193), dtype=np.float32)

        mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        zcr    = librosa.feature.zero_crossing_rate(y)
        rms    = librosa.feature.rms(y=y)

        feats = np.concatenate([
            mfcc.mean(axis=1), mfcc.std(axis=1),
            chroma.mean(axis=1), chroma.std(axis=1),
            librosa.power_to_db(mel).mean(axis=1)[:87],
            [zcr.mean()], [rms.mean()],
        ])
        target = 193
        if len(feats) >= target:
            feats = feats[:target]
        else:
            feats = np.pad(feats, (0, target - len(feats)))

        return feats.astype(np.float32).reshape(1, -1)

    except Exception as exc:
        logger.warning("WAV preprocessing hatası (%s) — mock", exc)
        return np.random.rand(1, 193).astype(np.float32)
