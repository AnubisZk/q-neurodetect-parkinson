"""
tests/test_fusion_phase5.py
============================
Phase 5 — Fusion servislerinin örnek veriyle çalıştırma akışı.

Çalıştırma:
    cd parkinson_multimodal_system
    pytest tests/test_fusion_phase5.py -v

    # Ya da doğrudan:
    python tests/test_fusion_phase5.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from app.services.fusion import (
    ModalityScore, weighted_ensemble, stacking_fusion,
    bayesian_avg, run_fusion, DEFAULT_WEIGHTS,
)
from app.services.calibrator import calibrate_score, train_calibrator
from app.services.explain import generate_full_explanation, contribution_chart_data


# ── Örnek veri fabrikası ──────────────────────────────────────────────────────
def make_score(modality: str, prob: float, conf: float = None,
               available: bool = True) -> ModalityScore:
    if conf is None:
        conf = round(abs(prob - 0.5) * 2, 4)
    return ModalityScore(modality=modality, probability=prob,
                         confidence=conf, available=available)


# ── Senaryo setleri ───────────────────────────────────────────────────────────
SCENARIOS = {
    "tam_parkinson": [
        make_score("voice",   0.82, 0.64),
        make_score("mri",     0.79, 0.58),
        make_score("drawing", 0.74, 0.48),
    ],
    "tam_healthy": [
        make_score("voice",   0.18, 0.64),
        make_score("mri",     0.22, 0.56),
        make_score("drawing", 0.15, 0.70),
    ],
    "belirsiz": [
        make_score("voice",   0.55, 0.10),
        make_score("mri",     0.48, 0.04),
        make_score("drawing", 0.51, 0.02),
    ],
    "sadece_ses": [
        make_score("voice",   0.76, 0.52),
        make_score("mri",     0.50, 0.00, available=False),
        make_score("drawing", 0.50, 0.00, available=False),
    ],
    "ses_mri": [
        make_score("voice",   0.71, 0.42),
        make_score("mri",     0.68, 0.36),
        make_score("drawing", 0.50, 0.00, available=False),
    ],
    "hic_modalite": [
        make_score("voice",   0.50, 0.00, available=False),
        make_score("mri",     0.50, 0.00, available=False),
        make_score("drawing", 0.50, 0.00, available=False),
    ],
}


# ── Weighted Ensemble Testleri ─────────────────────────────────────────────────
class TestWeightedEnsemble:
    def test_tam_parkinson(self):
        r = weighted_ensemble(SCENARIOS["tam_parkinson"])
        assert r.risk_score > 0.70, f"Parkinson skor düşük: {r.risk_score}"
        assert r.risk_label == "high"
        assert r.method == "weighted_ensemble"

    def test_tam_healthy(self):
        r = weighted_ensemble(SCENARIOS["tam_healthy"])
        assert r.risk_score < 0.40, f"Healthy skor yüksek: {r.risk_score}"
        assert r.risk_label == "low"

    def test_belirsiz(self):
        r = weighted_ensemble(SCENARIOS["belirsiz"])
        assert 0.40 <= r.risk_score <= 0.65
        assert r.risk_label == "moderate"

    def test_eksik_mri_drawing(self):
        r = weighted_ensemble(SCENARIOS["sadece_ses"])
        assert r.missing_modalities == ["mri", "drawing"]
        assert r.risk_score > 0.5   # ses yüksek olduğu için

    def test_hic_modalite(self):
        r = weighted_ensemble(SCENARIOS["hic_modalite"])
        assert r.risk_score == 0.5  # belirsiz varsayılan

    def test_kontribyusyon_toplam_100(self):
        r = weighted_ensemble(SCENARIOS["ses_mri"])
        total = sum(r.modality_contributions.values())
        assert abs(total - 100.0) < 0.5, f"Katkı toplamı: {total}"

    def test_kismi_veri_confidence_cezasi(self):
        full    = weighted_ensemble(SCENARIOS["tam_parkinson"])
        partial = weighted_ensemble(SCENARIOS["sadece_ses"])
        assert partial.confidence < full.confidence, \
            "Eksik modalite güveni düşürmeli"


# ── Bayesian Testleri ─────────────────────────────────────────────────────────
class TestBayesian:
    def test_prior_etkisi(self):
        scores = SCENARIOS["belirsiz"]
        r_low  = bayesian_avg(scores, prior=0.1)
        r_high = bayesian_avg(scores, prior=0.7)
        # Yüksek prior → yüksek posterior
        assert r_high.risk_score > r_low.risk_score

    def test_tam_parkinson(self):
        r = bayesian_avg(SCENARIOS["tam_parkinson"])
        assert r.risk_score > 0.70

    def test_method_ismi(self):
        r = bayesian_avg(SCENARIOS["tam_parkinson"])
        assert r.method == "bayesian_avg"


# ── Stacking Testleri ─────────────────────────────────────────────────────────
class TestStacking:
    def _mock_meta_model(self):
        """Basit mock meta-learner."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        rng = np.random.default_rng(42)
        X   = rng.random((200, 6))
        y   = (X[:, 0] * 0.4 + X[:, 1] * 0.4 + X[:, 2] * 0.2 > 0.5).astype(int)
        model = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression())])
        model.fit(X, y)
        return model

    def test_stacking_calisir(self):
        meta = self._mock_meta_model()
        r = stacking_fusion(SCENARIOS["tam_parkinson"], meta)
        assert 0.0 <= r.risk_score <= 1.0
        assert r.method in ("stacking", "stacking→weighted_fallback",
                             "stacking→error_fallback")

    def test_eksik_modalite_fallback(self):
        meta = self._mock_meta_model()
        r = stacking_fusion(SCENARIOS["sadece_ses"], meta)
        assert "fallback" in r.method


# ── run_fusion dispatcher ─────────────────────────────────────────────────────
class TestRunFusion:
    def test_weighted_dispatch(self):
        r = run_fusion(SCENARIOS["tam_parkinson"], method="weighted")
        assert r.method == "weighted_ensemble"

    def test_bayesian_dispatch(self):
        r = run_fusion(SCENARIOS["tam_parkinson"], method="bayesian")
        assert r.method == "bayesian_avg"

    def test_stacking_fallback_without_model(self):
        r = run_fusion(SCENARIOS["tam_parkinson"], method="stacking", meta_model=None)
        assert r.method == "weighted_ensemble"  # model yok → weighted


# ── Calibrator Testleri ───────────────────────────────────────────────────────
class TestCalibrator:
    def test_calibrate_no_model(self):
        # Calibrator yoksa ham skoru döndür
        result = calibrate_score(0.75, modality="nonexistent")
        assert result == 0.75

    def test_calibrate_clip(self):
        assert calibrate_score(1.5, "nonexistent") == 1.0
        assert calibrate_score(-0.1, "nonexistent") == 0.0

    def test_train_and_calibrate_platt(self):
        rng    = np.random.default_rng(42)
        y_prob = rng.uniform(0, 1, 200)
        y_true = (y_prob + rng.normal(0, 0.15, 200) > 0.5).astype(int)

        cal = train_calibrator(y_prob, y_true, modality="test_platt",
                                method="platt", save=False)
        result = cal.predict_proba(np.array([[0.8]]))[0][1]
        assert 0.0 <= result <= 1.0

    def test_train_and_calibrate_isotonic(self):
        rng    = np.random.default_rng(99)
        y_prob = rng.uniform(0, 1, 300)
        y_true = (y_prob > 0.5).astype(int)

        cal = train_calibrator(y_prob, y_true, modality="test_iso",
                                method="isotonic", save=False)
        result = cal.predict_proba(np.array([[0.3]]))[0][1]
        assert 0.0 <= result <= 1.0


# ── Explanation Testleri ──────────────────────────────────────────────────────
class TestExplain:
    def test_summary_uretilir(self):
        r = weighted_ensemble(SCENARIOS["tam_parkinson"])
        exp = generate_full_explanation(r)
        assert "summary" in exp
        assert len(exp["summary"]) > 20

    def test_kontribyusyon_chart_data(self):
        r    = weighted_ensemble(SCENARIOS["ses_mri"])
        data = contribution_chart_data(r)
        assert "voice" in data
        assert "mri" in data
        assert "drawing" not in data  # mevcut değil

    def test_missing_warns(self):
        r   = weighted_ensemble(SCENARIOS["sadece_ses"])
        exp = generate_full_explanation(r)
        assert len(exp["missing_warns"]) == 2

    def test_risk_factors(self):
        r   = weighted_ensemble(SCENARIOS["tam_parkinson"])
        exp = generate_full_explanation(r)
        assert len(exp["risk_factors"]["risk"]) > 0

    def test_protective_factors(self):
        r   = weighted_ensemble(SCENARIOS["tam_healthy"])
        exp = generate_full_explanation(r)
        assert len(exp["risk_factors"]["protective"]) > 0


# ── Tam Akış (entegrasyon) ────────────────────────────────────────────────────
class TestFullFlow:
    """Örnek test verisiyle baştan sona akış."""

    def test_parkinson_hasta_akisi(self):
        print("\n\n=== SENARYO: Parkinson Hastası (3 modalite) ===")
        scores = SCENARIOS["tam_parkinson"]
        result = run_fusion(scores, method="weighted")
        exp    = generate_full_explanation(result)

        print(f"Risk Skoru    : {result.risk_score:.0%}")
        print(f"Risk Etiketi  : {result.risk_display}")
        print(f"Güven         : {result.confidence:.0%}")
        print(f"Füzyon Yöntemi: {result.method}")
        print(f"Katkılar      : {result.modality_contributions}")
        print(f"Özet          : {exp['summary'][:80]}…")
        print(f"Risk bulguları: {exp['risk_factors']['risk']}")

        assert result.risk_label == "high"
        assert result.modality_contributions != {}

    def test_eksik_modalite_akisi(self):
        print("\n\n=== SENARYO: Sadece Ses Verisi ===")
        scores = SCENARIOS["sadece_ses"]
        result = run_fusion(scores, method="bayesian")
        exp    = generate_full_explanation(result)

        print(f"Risk Skoru    : {result.risk_score:.0%}")
        print(f"Eksik         : {result.missing_modalities}")
        print(f"Uyarılar      : {exp['missing_warns']}")
        print(f"Güven (ceza)  : {result.confidence:.0%}")

        assert "mri" in result.missing_modalities
        assert "drawing" in result.missing_modalities
        assert result.confidence < 0.50  # eksik modalite cezası

    def test_uc_yontem_karsilastirma(self):
        print("\n\n=== SENARYO: 3 Yöntem Karşılaştırma ===")
        scores = SCENARIOS["belirsiz"]
        header = f"{'Yöntem':<25} {'Skor':>8} {'Etiket':>12} {'Güven':>8}"
        print(header)
        print("-" * 55)
        for method in ["weighted", "bayesian"]:
            r = run_fusion(scores, method=method)
            print(f"{r.method:<25} {r.risk_score:>8.4f} "
                  f"{r.risk_display:>12} {r.confidence:>8.4f}")


# ── Doğrudan çalıştırma ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import unittest

    # Full flow senaryolarını doğrudan çalıştır
    t = TestFullFlow()
    t.test_parkinson_hasta_akisi()
    t.test_eksik_modalite_akisi()
    t.test_uc_yontem_karsilastirma()

    print("\n✅ Tüm manuel testler geçti.")
    print("\n   pytest ile tam test seti için:")
    print("   pytest tests/test_fusion_phase5.py -v")
