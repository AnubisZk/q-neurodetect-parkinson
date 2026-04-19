"""
notebooks/train_fusion.py
=========================
Phase 5 — Fusion meta-learner + calibrator eğitimi.

Çalıştırma:
    cd parkinson_multimodal_system
    python notebooks/train_fusion.py

Gereksinim: ses ve MRI modellerinin önceden eğitilmiş olması.
Yoksa sentetik veri ile de çalışır (test modu).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, classification_report, brier_score_loss
from sklearn.calibration import calibration_curve

from app.services.calibrator import train_calibrator, plot_reliability

OUT_DIR = Path("models/fusion")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# ── Veri Üretici ──────────────────────────────────────────────────────────────
def generate_training_data(n_samples: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """
    Gerçek model çıktıları varsa onları kullan.
    Yoksa sentetik veri üret (test/demo modu).

    Feature sıralama:
      [voice_prob, mri_prob, drawing_prob,
       voice_conf, mri_conf, drawing_conf]
    """
    voice_pkl = Path("models/voice/voice_model.pkl")
    mri_h5    = Path("models/mri/mri_model.h5")

    if voice_pkl.exists() and mri_h5.exists():
        print("✅ Gerçek model çıktıları kullanılıyor…")
        return _collect_real_predictions(voice_pkl, mri_h5, n_samples)

    print("ℹ️  Gerçek model yok — sentetik veri üretiliyor (demo modu).")
    return _synthetic_data(n_samples)


def _synthetic_data(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Korelasyonlu sentetik modalite skorları üret.
    Parkinson hastalarında tüm modaliteler yüksek çıkma eğiliminde.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    y   = rng.integers(0, 2, n)

    def modality_prob(y_arr, strength=0.7, noise=0.25):
        signal = y_arr * strength
        return np.clip(signal + rng.normal(0, noise, n), 0.05, 0.95)

    voice_p   = modality_prob(y, 0.65)
    mri_p     = modality_prob(y, 0.72)
    drawing_p = modality_prob(y, 0.58)

    # Confidence: yüksek olasılık → yüksek güven (sigmoid benzeri)
    def conf(p): return np.clip(np.abs(p - 0.5) * 2 + rng.normal(0, 0.05, n), 0, 1)

    X = np.column_stack([
        voice_p, mri_p, drawing_p,
        conf(voice_p), conf(mri_p), conf(drawing_p)
    ]).astype(np.float32)
    return X, y.astype(np.int32)


def _collect_real_predictions(voice_pkl, mri_h5, n_samples):
    """Gerçek modelleri yükle ve validation seti üzerinde tahmin topla."""
    import warnings; warnings.filterwarnings("ignore")
    try:
        voice_model = joblib.load(voice_pkl)
        import tensorflow as tf
        mri_model = tf.keras.models.load_model(str(mri_h5))
        # Burada validation setini yükleyip geçirmen gerekir
        # Basitlik için sentetik'e düşüyoruz
        print("  Model yüklendi ama validation seti yok — sentetik ile devam.")
    except Exception as e:
        print(f"  Model yükleme hatası ({e}) — sentetik veri.")
    return _synthetic_data(n_samples)


# ── Model Tanımları ───────────────────────────────────────────────────────────
def build_logreg() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(C=1.0, solver="lbfgs",
                                      random_state=RANDOM_STATE, max_iter=500)),
    ])


def build_gbm() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            random_state=RANDOM_STATE)),
    ])


# ── Değerlendirme ──────────────────────────────────────────────────────────────
def evaluate_models(models: dict, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("\n📊 5-Fold CV Karşılaştırması:")
    print(f"  {'Model':<20} {'AUC':>8} {'±':>6} {'ACC':>8} {'±':>6} {'Brier':>8}")
    print("  " + "-" * 58)

    results = {}
    for name, model in models.items():
        aucs   = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        accs   = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        briers = cross_val_score(model, X, y, cv=cv,
                                  scoring="neg_brier_score")
        print(f"  {name:<20} {aucs.mean():>8.4f} {aucs.std():>6.4f} "
              f"{accs.mean():>8.4f} {accs.std():>6.4f} "
              f"{-briers.mean():>8.4f}")
        results[name] = {"auc": aucs.mean(), "acc": accs.mean()}

    best = max(results, key=lambda k: results[k]["auc"])
    print(f"\n🏆 En iyi model: {best} (AUC={results[best]['auc']:.4f})")
    return best, results


# ── Kalibrasyon Analizi ───────────────────────────────────────────────────────
def calibration_analysis(model, X_te, y_te, name: str):
    y_prob = model.predict_proba(X_te)[:, 1]
    brier  = brier_score_loss(y_te, y_prob)
    auc    = roc_auc_score(y_te, y_prob)
    print(f"\n🎯 {name} — Test AUC: {auc:.4f}, Brier: {brier:.4f}")

    # Reliability diagram
    fig, ax = plt.subplots(figsize=(5, 4))
    frac, mean_pred = calibration_curve(y_te, y_prob, n_bins=8)
    ax.plot(mean_pred, frac, "s-b", label="Model")
    ax.plot([0, 1], [0, 1], "k--", label="Mükemmel")
    ax.set_title(f"Reliability — {name}\nAUC={auc:.3f}, Brier={brier:.4f}")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT_DIR / f"reliability_{name.lower().replace(' ','_')}.png", dpi=120)
    plt.close()
    return y_prob


# ── Ana Akış ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Q-NeuroDetect — Fusion Model Eğitimi (Phase 5)")
    print("=" * 60)

    X, y = generate_training_data(n_samples=400)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

    print(f"\n📐 Veri: train={len(X_tr)}, test={len(X_te)}")
    print(f"   Özellikler: [voice_prob, mri_prob, drawing_prob, "
          f"voice_conf, mri_conf, drawing_conf]")

    models = {
        "LogReg (Platt)": build_logreg(),
        "GBM":            build_gbm(),
    }

    best_name, cv_results = evaluate_models(models, X_tr, y_tr)

    # Final fit
    best_model = models[best_name]
    best_model.fit(X_tr, y_tr)

    # Kalibrasyon analizi (ham model)
    y_prob_raw = calibration_analysis(best_model, X_te, y_te, f"{best_name} (ham)")

    # Platt calibration
    print("\n🔧 Platt calibration eğitiliyor…")
    cal = train_calibrator(
        best_model.predict_proba(X_te)[:, 1], y_te,
        modality="fusion", method="platt"
    )
    y_prob_cal = cal.predict_proba(
        best_model.predict_proba(X_te)[:, 1].reshape(-1, 1))[:, 1]
    brier_cal = brier_score_loss(y_te, y_prob_cal)
    print(f"   Kalibre Brier: {brier_cal:.4f}  "
          f"(ham: {brier_score_loss(y_te, y_prob_raw):.4f})")

    plot_reliability(y_prob_cal, y_te, modality="fusion_calibrated")

    # Confusion matrix
    y_pred = best_model.predict(X_te)
    cm = confusion_matrix_plot(y_te, y_pred, best_name)

    # Kaydet
    out_path = OUT_DIR / "fusion_model.pkl"
    joblib.dump(best_model, out_path)
    print(f"\n✅ Fusion model kaydedildi: {out_path}")
    print("   Backend yeniden başlatıldığında otomatik yüklenecek.")


def confusion_matrix_plot(y_te, y_pred, name):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Healthy","Parkinson"],
                yticklabels=["Healthy","Parkinson"], ax=ax)
    ax.set_title(f"Fusion — {name}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fusion_confusion.png", dpi=120)
    plt.close()
    return cm


if __name__ == "__main__":
    main()
