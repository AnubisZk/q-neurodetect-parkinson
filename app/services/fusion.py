"""
app/services/fusion.py
======================
Phase 5 — Gerçek fusion implementasyonu.

Desteklenen yöntemler:
  1. weighted_ensemble  — dinamik ağırlık, eksik modalite toleranslı
  2. stacking           — eğitilmiş meta-learner (LogReg veya XGBoost)
  3. bayesian_avg       — posterior olasılık ortalaması (prior + likelihood)

Tüm yöntemler ModalityScore listesi alır, FusionResult döndürür.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

# ── Varsayılan modalite ağırlıkları (toplam 1.0) ──────────────────────────────
DEFAULT_WEIGHTS: dict[str, float] = {
    "voice":   0.35,
    "mri":     0.40,
    "drawing": 0.25,
}

RISK_THRESHOLDS = {"high": 0.65, "moderate": 0.40}


# ── Veri Yapıları ─────────────────────────────────────────────────────────────
@dataclass
class ModalityScore:
    modality: str               # "voice" | "mri" | "drawing"
    probability: float          # [0, 1] ham model çıktısı
    confidence: float           # [0, 1] modelin kendi güven skoru
    available: bool = True      # modalite yüklendi mi?
    raw_features: np.ndarray | None = field(default=None, repr=False)


@dataclass
class FusionResult:
    method: str                          # hangi yöntem kullanıldı
    risk_score: float                    # [0, 1] nihai Parkinson olasılığı
    risk_label: Literal["high", "moderate", "low"]
    risk_display: str                    # "Yüksek Risk" vb.
    confidence: float                    # [0, 1] füzyon güven skoru
    modality_contributions: dict[str, float]   # her modalite katkı yüzdesi
    modality_scores: list[ModalityScore]
    missing_modalities: list[str]
    calibrated: bool = False


# ── Yardımcı Fonksiyonlar ─────────────────────────────────────────────────────
def _risk_label(score: float) -> tuple[str, str]:
    if score >= RISK_THRESHOLDS["high"]:
        return "high", "🔴 Yüksek Risk"
    if score >= RISK_THRESHOLDS["moderate"]:
        return "moderate", "🟡 Orta Risk"
    return "low", "🟢 Düşük Risk"


def _contribution_pct(weights: dict[str, float], scores: list[ModalityScore]) -> dict[str, float]:
    """Her modalite katkısını yüzde olarak hesapla."""
    total_w = sum(weights.get(s.modality, 0) for s in scores if s.available)
    if total_w == 0:
        return {}
    contribs = {}
    for s in scores:
        if s.available:
            w = weights.get(s.modality, 0)
            contribs[s.modality] = round((w / total_w) * 100, 1)
    return contribs


def _fusion_confidence(scores: list[ModalityScore], weights: dict[str, float]) -> float:
    """
    Füzyon güven skoru:
      - Modalite güven değerlerinin ağırlıklı ortalaması
      - Eksik modalite sayısıyla ceza
    """
    available = [s for s in scores if s.available]
    if not available:
        return 0.0
    total_w = sum(weights.get(s.modality, 0) for s in available)
    if total_w == 0:
        return 0.0
    weighted_conf = sum(
        weights.get(s.modality, 0) * s.confidence for s in available
    ) / total_w
    # Eksik modalite cezası: her eksik modalite %10 güven düşürür
    n_missing = sum(1 for s in scores if not s.available)
    penalty = 0.10 * n_missing
    return round(max(0.0, weighted_conf - penalty), 4)


# ── Yöntem 1: Weighted Ensemble ───────────────────────────────────────────────
def weighted_ensemble(
    scores: list[ModalityScore],
    weights: dict[str, float] | None = None,
) -> FusionResult:
    """
    Dinamik ağırlıklı ortalama.
    Eksik modalitelerin ağırlığı kalan modalitelere normalize edilerek dağıtılır.
    """
    w = weights or DEFAULT_WEIGHTS
    available = [s for s in scores if s.available]
    missing   = [s.modality for s in scores if not s.available]

    if not available:
        # Hiç modalite yok — varsayılan belirsiz skor
        risk_score = 0.5
    else:
        total_w = sum(w.get(s.modality, 0.0) for s in available)
        if total_w == 0:
            total_w = 1.0
        risk_score = sum(
            (w.get(s.modality, 0.0) / total_w) * s.probability
            for s in available
        )

    risk_score = float(np.clip(risk_score, 0.0, 1.0))
    label, display = _risk_label(risk_score)

    return FusionResult(
        method="weighted_ensemble",
        risk_score=round(risk_score, 4),
        risk_label=label,
        risk_display=display,
        confidence=_fusion_confidence(scores, w),
        modality_contributions=_contribution_pct(w, scores),
        modality_scores=scores,
        missing_modalities=missing,
    )


# ── Yöntem 2: Stacking (Meta-Learner) ─────────────────────────────────────────
def stacking_fusion(
    scores: list[ModalityScore],
    meta_model,                  # eğitilmiş sklearn pipeline
    weights: dict[str, float] | None = None,
) -> FusionResult:
    """
    Eğitilmiş meta-learner ile fusion.
    Eksik modaliteler için weighted_ensemble'a düşer.
    """
    w = weights or DEFAULT_WEIGHTS
    available = {s.modality: s for s in scores if s.available}
    missing   = [s.modality for s in scores if not s.available]

    # Tüm modaliteler mevcut değilse — weighted'a düş
    required = {"voice", "mri", "drawing"}
    if not required.issubset(available.keys()):
        logger.info("Stacking: eksik modalite(%s) — weighted_ensemble'a düşülüyor", missing)
        result = weighted_ensemble(scores, w)
        result.method = "stacking→weighted_fallback"
        return result

    try:
        feature_vec = np.array([[
            available["voice"].probability,
            available["mri"].probability,
            available["drawing"].probability,
            available["voice"].confidence,
            available["mri"].confidence,
            available["drawing"].confidence,
        ]], dtype=np.float32)

        risk_score = float(meta_model.predict_proba(feature_vec)[0][1])
        risk_score = float(np.clip(risk_score, 0.0, 1.0))
        label, display = _risk_label(risk_score)

        return FusionResult(
            method="stacking",
            risk_score=round(risk_score, 4),
            risk_label=label,
            risk_display=display,
            confidence=_fusion_confidence(scores, w),
            modality_contributions=_contribution_pct(w, scores),
            modality_scores=scores,
            missing_modalities=missing,
        )
    except Exception as exc:
        logger.warning("Stacking inference hatası: %s — weighted_ensemble'a düşülüyor", exc)
        result = weighted_ensemble(scores, w)
        result.method = "stacking→error_fallback"
        return result


# ── Yöntem 3: Bayesian Model Averaging ────────────────────────────────────────
def bayesian_avg(
    scores: list[ModalityScore],
    prior: float = 0.3,          # Parkinson prevalans tahmini
    weights: dict[str, float] | None = None,
) -> FusionResult:
    """
    Posterior = prior × Π likelihood_i (Naive Bayes tarzı)
    Her modalite skoru likelihood olarak kullanılır.
    Log-space'de hesaplanır (numerik stabilite).
    """
    w = weights or DEFAULT_WEIGHTS
    available = [s for s in scores if s.available]
    missing   = [s.modality for s in scores if not s.available]

    if not available:
        risk_score = prior
    else:
        eps = 1e-7
        log_posterior_pos = np.log(prior + eps)
        log_posterior_neg = np.log(1 - prior + eps)

        for s in available:
            p = float(np.clip(s.probability, eps, 1 - eps))
            log_posterior_pos += np.log(p)
            log_posterior_neg += np.log(1 - p)

        # Softmax normalizasyon
        max_log = max(log_posterior_pos, log_posterior_neg)
        exp_pos = np.exp(log_posterior_pos - max_log)
        exp_neg = np.exp(log_posterior_neg - max_log)
        risk_score = float(exp_pos / (exp_pos + exp_neg))

    risk_score = float(np.clip(risk_score, 0.0, 1.0))
    label, display = _risk_label(risk_score)

    return FusionResult(
        method="bayesian_avg",
        risk_score=round(risk_score, 4),
        risk_label=label,
        risk_display=display,
        confidence=_fusion_confidence(scores, w),
        modality_contributions=_contribution_pct(w, scores),
        modality_scores=scores,
        missing_modalities=missing,
    )


# ── Ana Dispatcher ────────────────────────────────────────────────────────────
def run_fusion(
    scores: list[ModalityScore],
    method: str = "weighted",
    meta_model=None,
    weights: dict[str, float] | None = None,
    prior: float = 0.3,
) -> FusionResult:
    """
    method: "weighted" | "stacking" | "bayesian"
    meta_model: stacking için eğitilmiş sklearn pipeline (opsiyonel)
    """
    if method == "stacking" and meta_model is not None:
        return stacking_fusion(scores, meta_model, weights)
    elif method == "bayesian":
        return bayesian_avg(scores, prior, weights)
    else:
        return weighted_ensemble(scores, weights)
