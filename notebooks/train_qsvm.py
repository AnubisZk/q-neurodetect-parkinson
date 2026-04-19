"""
notebooks/train_qsvm.py
=======================
Kuantum SVM (QSVM) — PennyLane ZZFeatureMap + klasik SVM kernel
Ses verisi üzerinde çalışır (MFCC özellik vektörü, PCA ile 8 boyuta indirgenir).

Çalıştırma:
    cd parkinson_multimodal_system
    python notebooks/train_qsvm.py

Gereksinim:
    pip install pennylane scikit-learn joblib
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import seaborn as sns

OUT_DIR = Path("models/voice")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_QUBITS     = 8     # PCA boyutu = qubit sayısı
RANDOM_STATE = 42


# ── Kuantum Kernel ─────────────────────────────────────────────────────────────
def build_quantum_kernel(n_qubits: int = N_QUBITS):
    """
    ZZFeatureMap benzeri kuantum devresi ile kernel matrisi hesaplar.
    Her veri noktası için bağımsız devre çalıştırır.
    """
    import pennylane as qml

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x1, x2):
        # ZZFeatureMap: iki kez uygula (x1 state hazırlama)
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
            qml.RZ(2.0 * x1[i], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
            qml.RZ(2.0 * (np.pi - x1[i]) * (np.pi - x1[i + 1]), wires=i + 1)
            qml.CNOT(wires=[i, i + 1])

        # x2 için adjoint (inner product hesabı)
        for i in range(n_qubits - 2, -1, -1):
            qml.CNOT(wires=[i, i + 1])
            qml.RZ(-2.0 * (np.pi - x2[i]) * (np.pi - x2[i + 1]), wires=i + 1)
            qml.CNOT(wires=[i, i + 1])
        for i in range(n_qubits - 1, -1, -1):
            qml.RZ(-2.0 * x2[i], wires=i)
            qml.Hadamard(wires=i)

        return qml.probs(wires=range(n_qubits))

    def kernel(x1, x2):
        probs = circuit(x1, x2)
        return float(probs[0])  # |0...0⟩ durumunun olasılığı = kernel değeri

    return kernel


def build_kernel_matrix(kernel_fn, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """Kernel matrisi hesapla — O(n²) çağrı."""
    n, m = len(X1), len(X2)
    K = np.zeros((n, m))
    total = n * m
    for i in range(n):
        for j in range(m):
            K[i, j] = kernel_fn(X1[i], X2[j])
        if (i + 1) % 5 == 0:
            print(f"  Kernel: {(i+1)*m}/{total} tamamlandı…", end="\r")
    print()
    return K


# ── Veri Yükleme (ses eğitimi kodunu yeniden kullan) ──────────────────────────
def load_voice_features():
    """
    Önce eğitilmiş ses modelinden scaler'ı kullan.
    Yoksa ham veriden çıkar.
    """
    voice_model_path = Path("models/voice/voice_model.pkl")
    if voice_model_path.exists():
        print("✅ Mevcut ses modeli pipeline'ından özellikler alınıyor…")
        model = joblib.load(voice_model_path)
        # Eğitim verisi cache'i yok, ham veriden çıkarmak gerekiyor
    
    # Ham veri
    from notebooks.train_voice import build_dataset
    X, y = build_dataset()
    return X, y


# ── Ana Eğitim ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Q-NeuroDetect — QSVM Eğitimi (PennyLane)")
    print("=" * 55)

    try:
        X, y = load_voice_features()
    except Exception as e:
        print(f"⚠️  Ses verisi yüklenemedi: {e}")
        print("   Mock veri ile devam ediliyor (test modu)…")
        np.random.seed(RANDOM_STATE)
        X = np.random.randn(80, 193).astype(np.float32)
        y = np.random.randint(0, 2, 80)

    # ── Ön işleme: scale + PCA → N_QUBITS boyut ───────────────────────────────
    scaler = StandardScaler()
    pca    = PCA(n_components=N_QUBITS, random_state=RANDOM_STATE)
    X_sc   = scaler.fit_transform(X)
    X_pca  = pca.fit_transform(X_sc)

    # [-π, π] aralığına normalize et (kuantum devresi için)
    X_norm = np.pi * (X_pca - X_pca.min(0)) / (X_pca.ptp(0) + 1e-8) - np.pi / 2

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_norm, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # Küçük alt küme — kernel O(n²) zaman karmaşıklığında
    MAX_TRAIN = 60
    if len(X_tr) > MAX_TRAIN:
        print(f"ℹ️  CPU uyumu için eğitim seti {MAX_TRAIN} örneğe kırpıldı.")
        idx = np.random.choice(len(X_tr), MAX_TRAIN, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]

    print(f"📐 Veri: train={len(X_tr)}, test={len(X_te)}, qubit={N_QUBITS}")

    # ── Kuantum kernel matrisi ─────────────────────────────────────────────────
    print("\n⚛️  Kuantum kernel matrisi hesaplanıyor…")
    kernel_fn = build_quantum_kernel(N_QUBITS)
    K_train = build_kernel_matrix(kernel_fn, X_tr, X_tr)
    K_test  = build_kernel_matrix(kernel_fn, X_te, X_tr)

    # ── SVM precomputed kernel ile ─────────────────────────────────────────────
    print("\n🤖 SVM eğitimi (precomputed kernel)…")
    clf = SVC(kernel="precomputed", C=1.0, probability=True,
              random_state=RANDOM_STATE)
    clf.fit(K_train, y_tr)

    y_prob = clf.predict_proba(K_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    print("\n📋 QSVM Classification Report:")
    print(classification_report(y_te, y_pred,
                                 target_names=["Healthy", "Parkinson"]))
    auc = roc_auc_score(y_te, y_prob)
    print(f"   Test AUC: {auc:.4f}")

    # Confusion matrix
    cm  = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=["Healthy","Parkinson"],
                yticklabels=["Healthy","Parkinson"], ax=ax)
    ax.set_title(f"QSVM — Confusion Matrix  (AUC={auc:.3f})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "qsvm_confusion.png", dpi=120)
    print(f"✅ Grafik: {OUT_DIR}/qsvm_confusion.png")

    # ── Kaydet ────────────────────────────────────────────────────────────────
    qsvm_bundle = {
        "clf": clf,
        "scaler": scaler,
        "pca": pca,
        "K_train": K_train,
        "X_tr_norm": X_tr,
        "n_qubits": N_QUBITS,
    }
    out_path = OUT_DIR / "qsvm_model.pkl"
    joblib.dump(qsvm_bundle, out_path)
    print(f"✅ QSVM bundle kaydedildi: {out_path}")
    print("\nNot: QSVM inference için kernel matrisinin yeniden hesaplanması gerekir.")
    print("     Bu nedenle backend'de ayrı bir /predict/qsvm endpoint'i kullanılır.")


if __name__ == "__main__":
    main()
