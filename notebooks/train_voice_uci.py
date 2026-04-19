"""
notebooks/train_voice_uci.py
============================
UCI Parkinson CSV verisi → Gradient Boosting + SVM ensemble → pkl kayıt

PDF'teki kodunuzla aynı veri seti (parkinsons.csv) kullanır.
Gradient Boosting en iyi AUC'u verdi (0.917) → onu kullanıyoruz.

Çalıştırma:
    cd parkinson_multimodal_system
    python notebooks/train_voice_uci.py --csv data/raw/voice/parkinsons.csv

parkinsons.csv kolonları:
    name, MDVP:Fo(Hz), ..., status
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

OUT_DIR      = Path("models/voice")
RANDOM_STATE = 42
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_uci_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(csv_path)
    print(f"✅ CSV yüklendi: {len(df)} satır, {len(df.columns)} kolon")

    # name ve status düş
    drop = [c for c in df.columns if c.lower() in ("name", "status")]
    target_col = next((c for c in df.columns if c.lower() == "status"), None)
    if target_col is None:
        raise ValueError("'status' kolonu bulunamadı!")

    y = df[target_col].astype(int).values
    X_df = df.drop(columns=drop + [target_col] if target_col not in drop else drop,
                   errors="ignore")
    X_df = X_df.select_dtypes(include=[np.number])

    print(f"   Özellik sayısı: {X_df.shape[1]}")
    print(f"   Parkinson={y.sum()} | Healthy={(y==0).sum()}")
    return X_df.values.astype(np.float32), y, list(X_df.columns)


def build_model():
    gbm = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=RANDOM_STATE)
    rf  = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        random_state=RANDOM_STATE, n_jobs=-1)
    svm = SVC(kernel="rbf", C=10, gamma="scale",
              probability=True, random_state=RANDOM_STATE)

    voter = VotingClassifier(
        estimators=[("gbm", gbm), ("rf", rf), ("svm", svm)],
        voting="soft",
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    voter),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/raw/voice/parkinsons.csv",
                        help="UCI parkinsons.csv yolu")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        # Demo: sentetik veri
        print(f"⚠️  {csv_path} bulunamadı — demo veri üretiliyor")
        rng = np.random.default_rng(42)
        X   = rng.random((195, 22)).astype(np.float32)
        y   = (X[:, 0] > 0.25).astype(int)  # ~%75 Parkinson (UCI oranı)
        feat_names = [f"feat_{i}" for i in range(22)]
    else:
        X, y, feat_names = load_uci_csv(csv_path)

    # ── Cross-validation ──────────────────────────────────────────────────────
    model = build_model()
    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    aucs  = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    accs  = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    print(f"\n📊 5-Fold CV:")
    print(f"   AUC : {aucs.mean():.4f} ± {aucs.std():.4f}")
    print(f"   ACC : {accs.mean():.4f} ± {accs.std():.4f}")

    # ── Test seti değerlendirmesi ─────────────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    print(f"\n📋 Test seti:")
    print(classification_report(y_te, y_pred, target_names=["Healthy", "Parkinson"]))
    print(f"   AUC: {roc_auc_score(y_te, y_prob):.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Healthy","Parkinson"],
                yticklabels=["Healthy","Parkinson"], ax=ax)
    ax.set_title("Voice UCI — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "voice_uci_confusion.png", dpi=120)
    print(f"✅ Grafik: {OUT_DIR}/voice_uci_confusion.png")

    # ── SHAP (özellik önemi) ──────────────────────────────────────────────────
    try:
        import shap
        scaler  = model.named_steps["scaler"]
        gbm_clf = model.named_steps["clf"].estimators_[0]  # gbm
        X_sc    = scaler.transform(X_te)
        exp     = shap.TreeExplainer(gbm_clf)
        sv      = exp.shap_values(X_sc)
        shap.summary_plot(sv, X_sc, feature_names=feat_names, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "voice_uci_shap.png", dpi=120, bbox_inches="tight")
        plt.close()
        print(f"✅ SHAP: {OUT_DIR}/voice_uci_shap.png")
    except Exception as e:
        print(f"⚠️  SHAP atlandı: {e}")

    # ── Final fit ve kaydet ───────────────────────────────────────────────────
    model.fit(X, y)
    out = OUT_DIR / "voice_model.pkl"
    joblib.dump(model, out)
    print(f"\n✅ Model kaydedildi: {out}")
    print("   Backend yeniden başlatıldığında otomatik yüklenecek.")
    print(f"   Beklenen input shape: (1, {X.shape[1]})")


if __name__ == "__main__":
    main()
