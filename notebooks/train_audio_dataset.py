"""
notebooks/train_audio_dataset.py
=================================
Parkinson Multi Model DATASET → Ses modeli eğitimi (WAV dosyaları)

Veri yapısı:
    Healthy/AUDIO 1 HEALTHY/   *.wav (veya diğer ses formatları)
    Healthy/AUDIO 2 HEALTHY/   *.wav
    Unhealthy/AUDIO 1 UNHEALTHY/  *.wav
    Unhealthy/AUDIO 2 UNHEALTHY/  *.wav

Çalıştırma:
    cd ~/Desktop/parkinson_multimodal_system
    python notebooks/train_audio_dataset.py \
        --dataset "/Users/zafersavaskivilcim/Desktop/Parkinson Multi Model DATASET"
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

RANDOM_STATE = 42
SR           = 22050
DURATION     = 10.0
N_MFCC       = 40
OUT_DIR      = Path("models/voice")
OUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff"}


# ── Özellik Çıkarımı ──────────────────────────────────────────────────────────
def extract_features(path: Path) -> np.ndarray | None:
    try:
        import librosa
        y, sr = librosa.load(str(path), sr=SR, mono=True, duration=DURATION)
        y, _  = librosa.effects.trim(y, top_db=20)
        if len(y) < 1000:
            return None

        mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        zcr    = librosa.feature.zero_crossing_rate(y)
        rms    = librosa.feature.rms(y=y)
        sc     = librosa.feature.spectral_centroid(y=y, sr=sr)
        sb     = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        sr_feat = librosa.feature.spectral_rolloff(y=y, sr=sr)

        feats = np.concatenate([
            mfcc.mean(axis=1), mfcc.std(axis=1),       # 80
            chroma.mean(axis=1), chroma.std(axis=1),    # 24
            [zcr.mean(), zcr.std()],                    # 2
            [rms.mean(), rms.std()],                    # 2
            [sc.mean(), sc.std()],                      # 2
            [sb.mean(), sb.std()],                      # 2
            [sr_feat.mean(), sr_feat.std()],            # 2
        ])   # toplam: 114
        return feats.astype(np.float32)

    except Exception as e:
        print(f"  ⚠️  {path.name}: {e}")
        return None


# ── Veri Yükleme ───────────────────────────────────────────────────────────────
def load_dataset(dataset_root: Path):
    healthy_dirs = [
        dataset_root / "Healthy" / "AUDIO 1 HEALTHY",
        dataset_root / "Healthy" / "AUDIO 2 HEALTHY",
    ]
    unhealthy_dirs = [
        dataset_root / "Unhealthy" / "AUDIO 1 UNHEALTHY",
        dataset_root / "Unhealthy" / "AUDIO 2 UNHEALTHY",
    ]

    X, y = [], []

    for d in healthy_dirs:
        if not d.exists():
            print(f"  ⚠️  Klasör yok: {d}")
            continue
        files = [f for f in d.iterdir() if f.suffix.lower() in AUDIO_EXTS]
        print(f"  {d.name}: {len(files)} dosya")
        for p in sorted(files):
            feats = extract_features(p)
            if feats is not None:
                X.append(feats); y.append(0)

    for d in unhealthy_dirs:
        if not d.exists():
            print(f"  ⚠️  Klasör yok: {d}")
            continue
        files = [f for f in d.iterdir() if f.suffix.lower() in AUDIO_EXTS]
        print(f"  {d.name}: {len(files)} dosya")
        for p in sorted(files):
            feats = extract_features(p)
            if feats is not None:
                X.append(feats); y.append(1)

    if not X:
        print("❌ Hiç ses dosyası yüklenemedi.")
        print("   Ses dosyası formatları:", AUDIO_EXTS)
        sys.exit(1)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"\n✅ Ses veri seti: {len(X)} kayıt, {X.shape[1]} özellik")
    print(f"   Healthy={(y==0).sum()} | Parkinson={y.sum()}")
    return X, y


# ── Model ──────────────────────────────────────────────────────────────────────
def build_model():
    gbm = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                      learning_rate=0.05,
                                      random_state=RANDOM_STATE)
    rf  = RandomForestClassifier(n_estimators=200, max_depth=12,
                                  random_state=RANDOM_STATE, n_jobs=-1)
    svm = SVC(kernel="rbf", C=10, gamma="scale",
              probability=True, random_state=RANDOM_STATE)
    voter = VotingClassifier(
        estimators=[("gbm", gbm), ("rf", rf), ("svm", svm)],
        voting="soft"
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    voter),
    ])


# ── Eğitim ─────────────────────────────────────────────────────────────────────
def train(X, y):
    model = build_model()
    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    aucs  = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    accs  = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    print(f"\n📊 5-Fold CV:")
    print(f"   AUC : {aucs.mean():.4f} ± {aucs.std():.4f}")
    print(f"   ACC : {accs.mean():.4f} ± {accs.std():.4f}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    print(f"\n📋 Test seti:")
    print(classification_report(y_te, y_pred,
                                 target_names=["Healthy","Parkinson"]))
    try:
        print(f"   AUC: {roc_auc_score(y_te, y_prob):.4f}")
    except Exception:
        pass

    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy','Parkinson'],
                yticklabels=['Healthy','Parkinson'], ax=ax)
    ax.set_title('Audio Model — Confusion Matrix')
    fig.tight_layout()
    fig.savefig(OUT_DIR / "audio_confusion.png", dpi=120)
    print(f"✅ Grafik: {OUT_DIR}/audio_confusion.png")

    model.fit(X, y)
    return model


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="/Users/zafersavaskivilcim/Desktop/Parkinson Multi Model DATASET",
    )
    args = parser.parse_args()
    root = Path(args.dataset)

    print("=" * 55)
    print("  Q-NeuroDetect — Ses Modeli Eğitimi")
    print(f"  Dataset: {root}")
    print("=" * 55)

    if not root.exists():
        print(f"❌ Klasör bulunamadı: {root}")
        sys.exit(1)

    X, y  = load_dataset(root)
    model = train(X, y)

    out = OUT_DIR / "voice_model.pkl"
    joblib.dump(model, out)
    print(f"\n✅ Model kaydedildi: {out}")
    print(f"   Input shape: (1, {X.shape[1]})")
    print("   Backend yeniden başlatınca otomatik yüklenecek.")


if __name__ == "__main__":
    main()
