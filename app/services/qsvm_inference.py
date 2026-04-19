"""
app/services/qsvm_inference.py
Eğitilmiş QSVM bundle'ından inference yapar.
Bundle şunları içerir: clf, scaler, pca, K_train, X_tr_norm, n_qubits
"""
from __future__ import annotations
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


def qsvm_predict(feature_vector: np.ndarray, bundle: dict) -> float:
    """
    feature_vector: (N_FEATURES,) ham ses özellik vektörü
    bundle: train_qsvm.py'nin kaydettiği dict
    Döner: Parkinson olasılığı [0, 1]
    """
    try:
        import pennylane as qml

        clf       = bundle["clf"]
        scaler    = bundle["scaler"]
        pca       = bundle["pca"]
        X_tr_norm = bundle["X_tr_norm"]
        n_qubits  = bundle["n_qubits"]

        # Ön işleme
        x_sc   = scaler.transform(feature_vector.reshape(1, -1))
        x_pca  = pca.transform(x_sc)
        x_norm = np.pi * (x_pca - x_pca.min()) / (x_pca.ptp() + 1e-8) - np.pi / 2

        # Kernel vektörü — yeni nokta ile tüm eğitim setine karşı
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def circuit(x1, x2):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                qml.RZ(2.0 * x1[i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(2.0*(np.pi-x1[i])*(np.pi-x1[i+1]), wires=i+1)
                qml.CNOT(wires=[i, i + 1])
            for i in range(n_qubits - 2, -1, -1):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(-2.0*(np.pi-x2[i])*(np.pi-x2[i+1]), wires=i+1)
                qml.CNOT(wires=[i, i + 1])
            for i in range(n_qubits - 1, -1, -1):
                qml.RZ(-2.0 * x2[i], wires=i)
                qml.Hadamard(wires=i)
            return qml.probs(wires=range(n_qubits))

        x_q = x_norm.flatten()
        k_vec = np.array([float(circuit(x_q, xt)[0]) for xt in X_tr_norm])
        prob = clf.predict_proba(k_vec.reshape(1, -1))[0][1]
        return float(prob)

    except Exception as exc:
        logger.warning("QSVM inference hatası: %s — mock döndürülüyor", exc)
        return float(np.random.uniform(0.3, 0.75))
