"""
notebooks/train_mri_dataset.py
==============================
Parkinson Multi Model DATASET → MRI eğitimi

Veri yapısı:
    <DATASET_ROOT>/Healthy/MRI 1 HEALTHY/   *.png
    <DATASET_ROOT>/Healthy/MRI 2 HEALTHY/   *.png
    <DATASET_ROOT>/Unhealthy/MRI 1 UNHEALTHY/  *.png
    <DATASET_ROOT>/Unhealthy/MRI 2 UNHEALTHY/  *.png

Çalıştırma:
    cd ~/Desktop/parkinson_multimodal_system
    python notebooks/train_mri_dataset.py \
        --dataset "/Users/zafersavaskivilcim/Desktop/Parkinson Multi Model DATASET"
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

IMG_SIZE     = (128, 128)   # Modeliniz 128x128 ile eğitildi
BATCH_SIZE   = 8
EPOCHS       = 15
RANDOM_STATE = 42
OUT_DIR      = Path("models/mri")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Veri Yükleme ───────────────────────────────────────────────────────────────
def load_dataset(dataset_root: Path):
    healthy_dirs   = [
        dataset_root / "Healthy" / "MRI 1 HEALTHY",
        dataset_root / "Healthy" / "MRI 2 HEALTHY",
    ]
    unhealthy_dirs = [
        dataset_root / "Unhealthy" / "MRI 1 UNHEALTHY",
        dataset_root / "Unhealthy" / "MRI 2 UNHEALTHY",
    ]

    X, y = [], []

    for d in healthy_dirs:
        for p in sorted(d.glob("*.png")) + sorted(d.glob("*.jpg")):
            arr = _load(p)
            if arr is not None:
                X.append(arr); y.append(0)

    for d in unhealthy_dirs:
        for p in sorted(d.glob("*.png")) + sorted(d.glob("*.jpg")):
            arr = _load(p)
            if arr is not None:
                X.append(arr); y.append(1)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"✅ MRI veri seti: {len(X)} görüntü")
    print(f"   Healthy={( y==0).sum()} | Parkinson={y.sum()}")
    return X, y


def _load(path: Path):
    try:
        img = Image.open(path).convert("RGB").resize(IMG_SIZE)
        return np.array(img, dtype=np.float32) / 255.0
    except Exception as e:
        print(f"  ⚠️  {path.name}: {e}")
        return None


# ── Model ──────────────────────────────────────────────────────────────────────
def build_model():
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Conv2D(64, (3,3), activation='relu', input_shape=(*IMG_SIZE, 3)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(256, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# ── Eğitim ─────────────────────────────────────────────────────────────────────
def train(X, y):
    import tensorflow as tf

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    model = build_model()
    model.summary()

    n_pos = y_tr.sum()
    n_neg = len(y_tr) - n_pos
    class_weight = {0: 1.0, 1: n_neg / (n_pos + 1e-8)}

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=4,
            restore_best_weights=True, verbose=1),
    ]

    print(f"\n🏋️  Eğitim: {EPOCHS} epoch, batch={BATCH_SIZE}")
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_te, y_te),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # Değerlendirme
    loss, acc = model.evaluate(X_te, y_te, verbose=0)
    y_prob = model.predict(X_te, batch_size=BATCH_SIZE).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    print(f"\n📊 Test — Accuracy: {acc:.4f}")
    print(classification_report(y_te, y_pred,
                                 target_names=["Healthy","Parkinson"]))
    try:
        print(f"   AUC: {roc_auc_score(y_te, y_prob):.4f}")
    except Exception:
        pass

    # Grafik
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Val')
    axes[0].set_title('Accuracy'); axes[0].legend()

    cm = confusion_matrix(y_te, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy','Parkinson'],
                yticklabels=['Healthy','Parkinson'], ax=axes[1])
    axes[1].set_title('Confusion Matrix')
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mri_training.png", dpi=120)
    print(f"✅ Grafik: {OUT_DIR}/mri_training.png")

    return model


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="/Users/zafersavaskivilcim/Desktop/Parkinson Multi Model DATASET",
        help="Dataset ana klasör yolu"
    )
    args   = parser.parse_args()
    root   = Path(args.dataset)

    print("=" * 55)
    print("  Q-NeuroDetect — MRI Model Eğitimi")
    print(f"  Dataset: {root}")
    print("=" * 55)

    if not root.exists():
        print(f"❌ Klasör bulunamadı: {root}")
        sys.exit(1)

    X, y   = load_dataset(root)
    model  = train(X, y)

    out = OUT_DIR / "mri_model.h5"
    model.save(str(out))
    print(f"\n✅ Model kaydedildi: {out}")
    print("   Backend yeniden başlatınca otomatik yüklenecek.")


if __name__ == "__main__":
    main()
