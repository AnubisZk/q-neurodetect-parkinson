"""
notebooks/train_mri.py
======================
MRI PNG klasörü → MobileNetV2 fine-tune → model kayıt (CPU uyumlu)

Beklenen veri yapısı (iki format desteklenir):

  Format A — klasör bazlı etiket:
    data/raw/mri/
        healthy/
            img001.png
            img002.png
        parkinson/
            img101.png

  Format B — düz klasör + etiket CSV:
    data/raw/mri/
        images/
            img001.png
        labels.csv     ← sütunlar: filename, label (0/1)

Çalıştırma:
    cd parkinson_multimodal_system
    python notebooks/train_mri.py
"""

import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data/raw/mri")
OUT_DIR  = Path("models/mri")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE     = (224, 224)
BATCH_SIZE   = 8       # CPU için küçük batch
EPOCHS       = 15
RANDOM_STATE = 42


# ── Veri Yükleme ───────────────────────────────────────────────────────────────
def load_dataset():
    from PIL import Image

    X, y = [], []

    # Format A: healthy/ parkinson/ klasörleri
    healthy_dir   = DATA_DIR / "healthy"
    parkinson_dir = DATA_DIR / "parkinson"
    if healthy_dir.exists() and parkinson_dir.exists():
        print("📂 Format A: klasör bazlı etiket")
        for img_path in sorted(healthy_dir.glob("*.png")) + \
                         sorted(healthy_dir.glob("*.jpg")):
            arr = _load_img(img_path)
            if arr is not None:
                X.append(arr); y.append(0)
        for img_path in sorted(parkinson_dir.glob("*.png")) + \
                         sorted(parkinson_dir.glob("*.jpg")):
            arr = _load_img(img_path)
            if arr is not None:
                X.append(arr); y.append(1)

    # Format B: images/ + labels.csv
    else:
        import pandas as pd
        labels_csv = DATA_DIR / "labels.csv"
        images_dir = DATA_DIR / "images"
        if not labels_csv.exists():
            raise FileNotFoundError(
                f"Veri bulunamadı.\n"
                f"Beklenen: {healthy_dir} / {parkinson_dir}\n"
                f"Ya da: {labels_csv} + {images_dir}"
            )
        print("📂 Format B: CSV etiket")
        df = pd.read_csv(labels_csv)
        for _, row in df.iterrows():
            img_path = images_dir / row["filename"]
            arr = _load_img(img_path)
            if arr is not None:
                X.append(arr); y.append(int(row["label"]))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"✅ Veri seti: {len(X)} görüntü | "
          f"Parkinson={y.sum()} Healthy={(y==0).sum()}")
    return X, y


def _load_img(path: Path):
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB").resize(IMG_SIZE)
        return np.array(img, dtype=np.float32) / 255.0
    except Exception as e:
        print(f"  ⚠️  {path.name} atlandı: {e}")
        return None


# ── Model ──────────────────────────────────────────────────────────────────────
def build_keras_model():
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import MobileNetV2

    base = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    # İlk 100 katmanı dondur, son katmanları fine-tune et
    for layer in base.layers[:100]:
        layer.trainable = False
    for layer in base.layers[100:]:
        layer.trainable = True

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base.input, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


# ── Sklearn wrapper için özellik çıkarımı (SHAP uyumlu) ───────────────────────
def extract_mobilenet_features(X: np.ndarray) -> np.ndarray:
    """MobileNetV2 feature extractor — son FC öncesi embedding."""
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras import Model

    base  = MobileNetV2(input_shape=(*IMG_SIZE, 3),
                        include_top=False, weights="imagenet")
    pool  = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    feat_model = Model(inputs=base.input, outputs=pool)
    feats = feat_model.predict(X, batch_size=BATCH_SIZE, verbose=0)
    return feats.astype(np.float32)


# ── Eğitim ─────────────────────────────────────────────────────────────────────
def train(X, y):
    import tensorflow as tf

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    model = build_keras_model()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=4, mode="max",
            restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, verbose=1),
    ]

    # Sınıf ağırlıkları (dengesiz veri için)
    n_pos = y_tr.sum()
    n_neg = len(y_tr) - n_pos
    class_weight = {0: 1.0, 1: n_neg / (n_pos + 1e-8)}

    print(f"\n🏋️  Eğitim başlıyor — {EPOCHS} epoch, batch={BATCH_SIZE}")
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_te, y_te),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Değerlendirme ──────────────────────────────────────────────────────────
    y_prob = model.predict(X_te, batch_size=BATCH_SIZE).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    print("\n📋 Classification Report:")
    print(classification_report(y_te, y_pred,
                                 target_names=["Healthy", "Parkinson"]))
    print(f"   Test AUC: {roc_auc_score(y_te, y_prob):.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_te, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Healthy","Parkinson"],
                yticklabels=["Healthy","Parkinson"], ax=axes[0])
    axes[0].set_title("MRI Model — Confusion Matrix")

    axes[1].plot(history.history["auc"],     label="Train AUC")
    axes[1].plot(history.history["val_auc"], label="Val AUC")
    axes[1].set_title("AUC per Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "training_curves.png", dpi=120)
    print(f"✅ Grafikler kaydedildi: {OUT_DIR}/training_curves.png")

    return model, X_tr, X_te, y_tr, y_te


# ── SHAP ───────────────────────────────────────────────────────────────────────
def run_shap(keras_model, X_sample: np.ndarray):
    try:
        import shap
        print("\n🔍 SHAP GradientExplainer çalıştırılıyor…")
        import tensorflow as tf
        background = X_sample[:10]
        test_imgs  = X_sample[10:20]
        explainer  = shap.GradientExplainer(keras_model, background)
        shap_vals  = explainer.shap_values(test_imgs)
        sv = shap_vals[0] if isinstance(shap_vals, list) else shap_vals

        shap.image_plot(sv, test_imgs, show=False)
        plt.savefig(OUT_DIR / "shap_image.png", dpi=100, bbox_inches="tight")
        plt.close()
        print(f"✅ SHAP image plot kaydedildi: {OUT_DIR}/shap_image.png")
    except Exception as e:
        print(f"⚠️  SHAP atlandı: {e}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Q-NeuroDetect Parkinson — MRI Modeli Eğitimi")
    print("=" * 55)

    X, y = load_dataset()
    model, X_tr, X_te, y_tr, y_te = train(X, y)
    run_shap(model, X_te)

    # Keras H5 formatında kaydet
    out_h5 = OUT_DIR / "mri_model.h5"
    model.save(str(out_h5))
    print(f"\n✅ Model kaydedildi: {out_h5}")
    print("   Backend, bu modeli otomatik olarak yükleyecek.")


if __name__ == "__main__":
    main()
