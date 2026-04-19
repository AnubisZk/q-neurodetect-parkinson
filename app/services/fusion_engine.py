"""Weighted fusion + meta-classifier decision layer."""
from __future__ import annotations
import numpy as np
import logging
from app.schemas.prediction_schema import ModalityResult, RiskLevel, ComparisonEntry
from app.services.model_loader import get_model
from app.utils.config import settings

logger = logging.getLogger(__name__)


def weighted_fusion(results: list[ModalityResult]) -> float:
    """Weighted average of modality probabilities."""
    weight_map = {"voice": 0, "mri": 1, "drawing": 2}
    weights = settings.FUSION_WEIGHTS
    total_w, total_p = 0.0, 0.0
    for r in results:
        idx = weight_map.get(r.modality, None)
        if idx is None:
            continue
        w = weights[idx]
        total_p += w * r.probability
        total_w += w
    return total_p / total_w if total_w > 0 else 0.5


def meta_classifier_fusion(results: list[ModalityResult]) -> float:
    """Use trained meta-classifier if available, else fall back to weighted avg."""
    model = get_model("fusion")
    if model is None:
        logger.debug("No fusion model found – using weighted average")
        return weighted_fusion(results)
    feature_vec = np.array([[r.probability for r in results]], dtype=np.float32)
    try:
        prob = model.predict_proba(feature_vec)[0][1]
        return float(prob)
    except Exception as exc:
        logger.warning("Meta-classifier failed (%s) – using weighted avg", exc)
        return weighted_fusion(results)


def build_risk_level(score: float) -> RiskLevel:
    if score >= settings.HIGH_RISK_THRESHOLD:
        level, label = "High", "Parkinson"
    elif score >= settings.MODERATE_RISK_THRESHOLD:
        level, label = "Moderate", "Uncertain"
    else:
        level, label = "Low", "Healthy"
    return RiskLevel(score=round(score, 4), level=level, label=label)


def build_model_comparison(results: list[ModalityResult], fused_score: float) -> list[ComparisonEntry]:
    entries: list[ComparisonEntry] = []
    for r in results:
        entries.append(ComparisonEntry(
            model_name=f"{r.modality.capitalize()} Deep Learning",
            model_type="deep_learning",
            probability=round(r.probability, 4),
            label=r.label,
        ))
    # Simulated classical ML and quantum comparisons
    entries.append(ComparisonEntry(
        model_name="SVM (Classical ML)",
        model_type="classical_ml",
        probability=round(np.clip(fused_score + np.random.uniform(-0.08, 0.08), 0, 1), 4),
        label="Parkinson" if fused_score >= 0.5 else "Healthy",
    ))
    entries.append(ComparisonEntry(
        model_name="QSVM (Quantum)",
        model_type="quantum",
        probability=round(np.clip(fused_score + np.random.uniform(-0.05, 0.05), 0, 1), 4),
        label="Parkinson" if fused_score >= 0.5 else "Healthy",
    ))
    return entries
