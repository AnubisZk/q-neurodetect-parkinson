"""
app/services/explain.py
=======================
Phase 5 — Açıklanabilirlik katmanı.

Ürettiği çıktılar:
  1. Modalite katkı yüzdeleri (hangi modalite kararı ne kadar etkiledi)
  2. Risk açıklama metni (Türkçe, kural tabanlı)
  3. Özellik önemi özeti (SHAP varsa)
  4. Eksik modalite uyarısı
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.services.fusion import FusionResult, ModalityScore

logger = logging.getLogger(__name__)


# ── Türkçe şablonlar ──────────────────────────────────────────────────────────
_RISK_TEMPLATES = {
    "high": (
        "Çok modlu analiz **yüksek Parkinson riski** tespit etti "
        "(risk skoru: {score:.0%}, güven: {conf:.0%}). "
        "{dominant_modality} baskın belirleyici olarak öne çıkmaktadır. "
        "Nöroloji uzmanına başvurulması önerilir."
    ),
    "moderate": (
        "Analiz **orta düzey Parkinson riski** tespit etti "
        "(risk skoru: {score:.0%}, güven: {conf:.0%}). "
        "Modaliteler arasında karışık sinyaller gözlemlenmektedir. "
        "Klinik değerlendirme ve takip önerilir."
    ),
    "low": (
        "Analiz **düşük Parkinson riski** tespit etti "
        "(risk skoru: {score:.0%}, güven: {conf:.0%}). "
        "Mevcut veriler sağlıklı örüntülerle uyumludur. "
        "Rutin takip önerilir."
    ),
}

_MODALITY_DETAIL = {
    "voice": {
        True:  "🎙️ Ses: Jitter/shimmer yüksek, vokal instabilite saptandı.",
        False: "🎙️ Ses: Vokal parametreler normal aralıkta.",
    },
    "mri": {
        True:  "🧲 MRI: Substansiya nigra bölgesinde yoğunluk azalması gözlemlendi.",
        False: "🧲 MRI: Görüntüleme bulguları normal sınırlarda.",
    },
    "drawing": {
        True:  "✏️ Çizim: Motor titreme ve spiral düzensizliği tespit edildi.",
        False: "✏️ Çizim: Motor kontrol parametreleri normal.",
    },
}

_MISSING_WARNING = {
    "voice":   "🎙️ Ses verisi yüklenmedi — bu modalite analize dahil edilmedi.",
    "mri":     "🧲 MRI verisi yüklenmedi — bu modalite analize dahil edilmedi.",
    "drawing": "✏️ Çizim verisi yüklenmedi — bu modalite analize dahil edilmedi.",
}


# ── Ana Açıklama Üretici ──────────────────────────────────────────────────────
def generate_full_explanation(result: FusionResult) -> dict[str, Any]:
    """
    FusionResult'tan tam açıklama paketi üretir.

    Dönen dict:
      summary       : kısa Türkçe özet
      modality_lines: her modalite için detay satırı
      missing_warns : eksik modalite uyarıları
      contributions : modalite katkı yüzdeleri (dict)
      dominant      : en yüksek katkılı modalite adı
      risk_factors  : olumlu / olumsuz bulgu listesi
    """
    available = [s for s in result.modality_scores if s.available]
    contribs  = result.modality_contributions

    # Baskın modalite
    dominant = max(contribs, key=contribs.get) if contribs else "—"
    dominant_display = {
        "voice": "Ses analizi", "mri": "MRI görüntülemesi", "drawing": "Çizim testi"
    }.get(dominant, dominant)

    # Özet metin
    summary = _RISK_TEMPLATES[result.risk_label].format(
        score=result.risk_score,
        conf=result.confidence,
        dominant_modality=dominant_display,
    )

    # Modalite detay satırları
    modality_lines = []
    for s in available:
        is_pk = s.probability >= 0.5
        line  = _MODALITY_DETAIL.get(s.modality, {}).get(is_pk, "")
        contrib_pct = contribs.get(s.modality, 0)
        if line:
            modality_lines.append(
                f"{line} (olasılık: {s.probability:.0%}, katkı: {contrib_pct:.0f}%)"
            )

    # Eksik modalite uyarıları
    missing_warns = [
        _MISSING_WARNING[m] for m in result.missing_modalities
        if m in _MISSING_WARNING
    ]

    # Risk faktörleri
    risk_factors = _extract_risk_factors(result.modality_scores)

    return {
        "summary":        summary,
        "modality_lines": modality_lines,
        "missing_warns":  missing_warns,
        "contributions":  contribs,
        "dominant":       dominant,
        "risk_factors":   risk_factors,
        "method_used":    result.method,
        "calibrated":     result.calibrated,
    }


def _extract_risk_factors(scores: list[ModalityScore]) -> dict[str, list[str]]:
    """Olumlu (protective) ve olumsuz (risk) bulguları ayır."""
    positive, negative = [], []

    for s in scores:
        if not s.available:
            continue
        name = {"voice": "Ses", "mri": "MRI", "drawing": "Çizim"}.get(s.modality, s.modality)
        p = s.probability

        if p >= 0.70:
            negative.append(f"{name} skoru yüksek ({p:.0%}) — güçlü Parkinson sinyali")
        elif p >= 0.50:
            negative.append(f"{name} skoru sınırda ({p:.0%}) — hafif anormallik")
        elif p >= 0.30:
            positive.append(f"{name} skoru normal aralıkta ({p:.0%})")
        else:
            positive.append(f"{name} skoru düşük ({p:.0%}) — sağlıklı örüntü")

    return {"risk": negative, "protective": positive}


# ── Katkı Pasta Verisi (Streamlit için) ───────────────────────────────────────
def contribution_chart_data(result: FusionResult) -> dict[str, float]:
    """
    Streamlit st.bar_chart / st.plotly_chart için hazır dict.
    Sadece mevcut modaliteleri içerir.
    """
    return {
        k: v for k, v in result.modality_contributions.items()
        if any(s.modality == k and s.available for s in result.modality_scores)
    }


# ── SHAP Entegrasyonu (opsiyonel) ─────────────────────────────────────────────
def get_shap_summary(modality: str, feature_vector: np.ndarray | None) -> dict[str, float] | None:
    """
    Ses modeli için SHAP değerleri — model yüklüyse hesaplar.
    Diğer modaliteler için None döndürür (Phase 4'te genişletilebilir).
    """
    if modality != "voice" or feature_vector is None:
        return None
    try:
        from app.services.explainability import compute_shap_voice
        return compute_shap_voice(feature_vector)
    except Exception as exc:
        logger.warning("SHAP hesaplama hatası: %s", exc)
        return None
