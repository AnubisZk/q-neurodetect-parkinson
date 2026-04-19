"""
notebooks/train_drawing_dataset.py
===================================
Parkinson Multi Model DATASET → Çizim (Spiral/Wave) modeli eğitimi

Veri yapısı:
    Healthy/DRAWING 1 HEALTHY (S)/   *.png  ← Spiral
    Healthy/DRAWING 1 HEALTHY (W)/   *.png  ← Wave
    Healthy/DRAWING 2 HEALTHY (S)/   *.png
    Healthy/DRAWING 2 HEALTHY (W)/   *.png
    Unhealthy/DRAWING 1 UNHEALTHY (S)/  *.png
    ... vb.

Çalıştırma:
    cd ~/Desktop/parkinson_multimodal_system
    python notebooks/train_drawing_dataset.py \
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
from PIL import Image
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

IMG_SIZE     = (64, 64)
RANDOM_STATE = 42
OUT_DIR      = Path("models/drawing")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Veri Yükleme ───────────────────────────────────────────────────────────────
def load_dataset(dataset_root: Path, drawing_type: str = "both"):
    """
    drawing_type: "S" (spiral), "W" (wave), "both"
    """
    healthy_dirs, unhealthy_dirs = [], []

    for suffix in ["(S)", "(W)"]:
        if drawing_type == "S" and suffix == "(W)": continue
        if drawing_type == "W" and suffix == "(S)": continue
        for i in ["1", "2"]:
            h = dataset_root / "Healthy"   / f"DRAWING {i} HEALTHY {suffix}"
            u = dataset_root / "Unhealthy" / f"DRAWING {i} UNHEALTHY {suffix}"
            if h.exists(): healthy_dirs.append(h)
            if u.exists(): unhealthy_dirs.append(u)

    X, y = [], []

    for d in healthy_dirs:
        for p in sorted(d.glob("*.png")) + sorted(d.glob("*.jpg")):
            feats = _extract(p)
            if feats is not None:
                X.append(feats); y.append(0)

    for d in unhealthy_dirs:
        for p in sorted(d.glob("*.png")) + sorted(d.glob("*.jpg")):
            feats = _extract(p)
            if feats is not None:
                X.append(feats); y.append(1)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"✅ Çizim veri seti: {len(X)} görüntü, {X.shape[1]} özellik")
    print(f"   Healthy={( y==0).sum()} | Parkinson={y.sum()}")
    return X, y


def _extract(path: Path):
    """HOG-benzeri gradient histogram özellik çıkarımı."""
    try:
        img  = Image.open(path).convert("L").resize(IMG_SIZE)
        gray = np.array(img, dtype=np.float32) / 255.0
        gx   = np.gradient(gray, axis=1)
        gy   = np.gradient(gray, axis=0)
        mag  = np.sqrt(gx**2 + gy**2)
        ang  = np.arctan2(gy, gx)

        # 64 bin gradient histogram
        hist, _ = np.histogram(ang.flatten(), bins=64,
                               range=(-np.pi, np.pi),
                               weights=mag.flatten())
        hist = hist / (hist.sum() + 1e-8)

        # Ek istatistikler
        stats = np.array([
            gray.mean(), gray.std(),
            mag.mean(), mag.std(),
            gx.mean(), gy.mean(),
        ])
        return np.concatenate([hist, stats])
    except Exception as e:
        print(f"  ⚠️  {path.name}: {e}")
        return None


# ── Model ──────────────────────────────────────────────────────────────────────
def build_model():
    rf  = RandomForestClassifier(n_estimators=200, max_depth=12,
                                  random_state=RANDOM_STATE, n_jobs=-1)
    gbm = GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                      learning_rate=0.05,
                                      random_state=RANDOM_STATE)
    svm = SVC(kernel="rbf", C=5, gamma="scale",
              probability=True, random_state=RANDOM_STATE)
    voter = VotingClassifier(
        estimators=[("rf", rf), ("gbm", gbm), ("svm", svm)],
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
    print(f"   AUC: {roc_auc_score(y_te, y_prob):.4f}")

    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Healthy','Parkinson'],
                yticklabels=['Healthy','Parkinson'], ax=ax)
    ax.set_title('Drawing Model — Confusion Matrix')
    fig.tight_layout()
    fig.savefig(OUT_DIR / "drawing_confusion.png", dpi=120)
    print(f"✅ Grafik: {OUT_DIR}/drawing_confusion.png")

    model.fit(X, y)   # final fit tüm veriyle
    return model


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="/Users/zafersavaskivilcim/Desktop/Parkinson Multi Model DATASET",
    )
    parser.add_argument("--type", default="both",
                        choices=["S", "W", "both"],
                        help="Spiral (S), Wave (W) veya ikisi (both)")
    args = parser.parse_args()
    root = Path(args.dataset)

    print("=" * 55)
    print("  Q-NeuroDetect — Çizim Modeli Eğitimi")
    print(f"  Dataset: {root}")
    print(f"  Tip: {args.type}")
    print("=" * 55)

    if not root.exists():
        print(f"❌ Klasör bulunamadı: {root}")
        sys.exit(1)

    X, y  = load_dataset(root, args.type)
    model = train(X, y)

    out = OUT_DIR / "drawing_model.pkl"
    joblib.dump(model, out)
    print(f"\n✅ Model kaydedildi: {out}")


if __name__ == "__main__":
    main()
