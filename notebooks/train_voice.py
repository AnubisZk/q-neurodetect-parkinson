"""
notebooks/train_voice.py
========================
Ses verisi → MFCC özellik çıkarımı → SVM + RandomForest eğitimi → model kayıt

Beklenen veri yapısı:
    data/raw/voice/
        labels.csv          ← sütunlar: filename, label  (0=Healthy, 1=Parkinson)
        audio/
            hasta001.wav
            hasta002.wav
            ...

Çalıştırma:
    cd parkinson_multimodal_system
    python notebooks/train_voice.py
"""

import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data/raw/voice")
AUDIO_DIR  = DATA_DIR / "audio"
LABELS_CSV = DATA_DIR / "labels.csv"
OUT_DIR    = Path("models/voice")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_MFCC   = 40
SR       = 22050
DURATION = 10.0   # saniye — daha uzun dosyalar kırpılır
RANDOM_STATE = 42


# ── Feature Extraction ─────────────────────────────────────────────────────────
def extract_features(wav_path: Path) -> np.ndarray:
    """
    Bir WAV dosyasından 193 boyutlu özellik vektörü çıkar:
      - MFCC (40) : mean + std
      - Chroma (12): mean + std
      - Mel-spectrogram (128): mean
      - ZCR (1): mean
      - RMS (1): mean
    """
    import librosa
    y, sr = librosa.load(str(wav_path), sr=SR, mono=True, duration=DURATION)
    y, _ = librosa.effects.trim(y, top_db=20)
    if len(y) == 0:
        return np.zeros(193)

    mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    chroma   = librosa.feature.chroma_stft(y=y, sr=sr)
    mel      = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    zcr      = librosa.feature.zero_crossing_rate(y)
    rms      = librosa.feature.rms(y=y)

    feats = np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),       # 80
        chroma.mean(axis=1), chroma.std(axis=1),    # 24
        librosa.power_to_db(mel).mean(axis=1),      # 128 → truncate to 87 to hit 193?
        [zcr.mean()],                               # 1
        [rms.mean()],                               # 1
    ])
    # Sabit 193 boyut garantisi
    target = 193
    if len(feats) >= target:
        return feats[:target]
    return np.pad(feats, (0, target - len(feats)))


def build_dataset():
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"Etiket dosyası bulunamadı: {LABELS_CSV}")

    df = pd.read_csv(LABELS_CSV)
    assert "filename" in df.columns and "label" in df.columns, \
        "labels.csv sütunları: filename, label"

    X, y, skipped = [], [], []
    for _, row in df.iterrows():
        wav = AUDIO_DIR / row["filename"]
        if not wav.exists():
            skipped.append(row["filename"])
            continue
        feats = extract_features(wav)
        X.append(feats)
        y.append(int(row["label"]))

    if skipped:
        print(f"⚠️  Atlandı ({len(skipped)} dosya): {skipped[:5]}")

    print(f"✅ Veri seti: {len(X)} örnek | "
          f"Parkinson={sum(y)} Healthy={len(y)-sum(y)}")
    return np.array(X, dtype=np.float32), np.array(y)


# ── Model ──────────────────────────────────────────────────────────────────────
def build_model() -> Pipeline:
    svm = SVC(kernel="rbf", C=10, gamma="scale",
              probability=True, random_state=RANDOM_STATE)
    rf  = RandomForestClassifier(n_estimators=200, max_depth=12,
                                  random_state=RANDOM_STATE, n_jobs=-1)
    voter = VotingClassifier(
        estimators=[("svm", svm), ("rf", rf)],
        voting="soft"
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    voter),
    ])


# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate(model, X, y):
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    aucs = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    accs = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"\n📊 5-Fold CV Sonuçları:")
    print(f"   AUC  : {aucs.mean():.4f} ± {aucs.std():.4f}")
    print(f"   ACC  : {accs.mean():.4f} ± {accs.std():.4f}")

    # Son split üzerinde tam rapor
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    print("\n📋 Classification Report:")
    print(classification_report(y_te, y_pred,
                                 target_names=["Healthy", "Parkinson"]))
    print(f"   Test AUC: {roc_auc_score(y_te, y_prob):.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Healthy","Parkinson"],
                yticklabels=["Healthy","Parkinson"], ax=ax)
    ax.set_title("Voice Model — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=120)
    print(f"✅ Confusion matrix kaydedildi: {OUT_DIR}/confusion_matrix.png")
    return model


# ── SHAP ───────────────────────────────────────────────────────────────────────
def run_shap(model, X):
    try:
        import shap
        print("\n🔍 SHAP analizi çalıştırılıyor…")
        # Sadece RF kolu için SHAP (TreeExplainer)
        rf_step = model.named_steps["clf"].estimators_[1]  # rf
        scaler  = model.named_steps["scaler"]
        X_scaled = scaler.transform(X[:100])  # ilk 100 örnek yeterli

        explainer  = shap.TreeExplainer(rf_step)
        shap_vals  = explainer.shap_values(X_scaled)
        sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

        feat_names = [f"MFCC_{i}" for i in range(80)] + \
                     [f"Chroma_{i}" for i in range(24)] + \
                     [f"Mel_{i}" for i in range(87)] + \
                     ["ZCR", "RMS"]

        shap.summary_plot(sv, X_scaled, feature_names=feat_names[:X_scaled.shape[1]],
                          show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "shap_summary.png", dpi=120, bbox_inches="tight")
        plt.close()
        print(f"✅ SHAP summary kaydedildi: {OUT_DIR}/shap_summary.png")
    except Exception as e:
        print(f"⚠️  SHAP atlandı: {e}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Q-NeuroDetect Parkinson — Ses Modeli Eğitimi")
    print("=" * 55)

    X, y = build_dataset()

    model = build_model()
    model = evaluate(model, X, y)

    # Tüm veriyle final fit
    model.fit(X, y)
    run_shap(model, X)

    # Kaydet
    out_path = OUT_DIR / "voice_model.pkl"
    joblib.dump(model, out_path)
    print(f"\n✅ Model kaydedildi: {out_path}")
    print("   Backend otomatik olarak bu modeli yükleyecek.")


if __name__ == "__main__":
    main()
