"""
Açıklanabilirlik servisi.
- Kural tabanlı Türkçe metin (her zaman çalışır)
- SHAP tabanlı özellik önemi (model mevcut ise)
"""
from __future__ import annotations
import logging
import numpy as np
from app.schemas.prediction_schema import ModalityResult, RiskLevel

logger = logging.getLogger(__name__)

_TEMPLATES = {
    "High": (
        "Çok modlu analiz yüksek Parkinson riski tespit etti (risk skoru: {score:.0%}). "
        "Ses analizi titreme ve vokal kırılma örüntüleri gösterirken MRI verisi substansiya nigra "
        "bölgesinde anormallik işaretleri sunmaktadır. Çizim testi motor koordinasyon güçlüğüne "
        "işaret etmektedir. Nöroloji uzmanına başvurulması önerilir."
    ),
    "Moderate": (
        "Analiz orta düzey Parkinson riski tespit etti (risk skoru: {score:.0%}). "
        "Modaliteler arasında karışık sinyaller gözlemlenmektedir; bazı özellikler erken evre "
        "motor bozukluğuna işaret edebilir. Klinik değerlendirme ve takip önerilir."
    ),
    "Low": (
        "Analiz düşük Parkinson riski tespit etti (risk skoru: {score:.0%}). "
        "Ses, MRI ve çizim verileri sağlıklı örüntülerle uyumludur. "
        "Rutin takip önerilir; semptomlar gelişirse uzman görüşü alınız."
    ),
}

_MODALITY_NOTES = {
    "voice": {
        True:  "Ses: Jitter/shimmer değerleri yüksek – vokal instabilite mevcut.",
        False: "Ses: Vokal parametreler normal aralıkta.",
    },
    "mri": {
        True:  "MRI: Substansiya nigra yoğunluk kaybı gözlemlendi.",
        False: "MRI: Görüntüleme bulguları normal sınırlarda.",
    },
    "drawing": {
        True:  "Çizim: Motor titreme ve spiral bozulması saptandı.",
        False: "Çizim: Motor kontrol parametreleri normal.",
    },
}


def generate_explanation(modalities: list[ModalityResult], risk: RiskLevel) -> str:
    main_text = _TEMPLATES[risk.level].format(score=risk.score)
    detail_lines = []
    for m in modalities:
        is_parkinson = m.label == "Parkinson"
        note = _MODALITY_NOTES.get(m.modality, {}).get(is_parkinson, "")
        if note:
            detail_lines.append(note)
    if detail_lines:
        return main_text + "\n\nModalite Detayları:\n" + "\n".join(f"• {l}" for l in detail_lines)
    return main_text


def compute_shap_voice(feature_vector: np.ndarray) -> dict[str, float] | None:
    """
    Eğitilmiş ses modeli varsa SHAP değerlerini döndür.
    Yoksa None döndür (UI mock gösterir).
    """
    try:
        import shap
        from app.services.model_loader import get_model

        model = get_model("voice")
        if model is None or not hasattr(model, "predict_proba"):
            return None

        scaler = model.named_steps.get("scaler")
        clf    = model.named_steps.get("clf")
        if scaler is None or clf is None:
            return None

        x_sc = scaler.transform(feature_vector.reshape(1, -1))

        # VotingClassifier içindeki RF için TreeExplainer
        rf = None
        if hasattr(clf, "estimators_"):
            for name, est in clf.estimators_:
                if "rf" in name or "forest" in name.lower():
                    rf = est
                    break

        if rf is None:
            return None

        explainer = shap.TreeExplainer(rf)
        sv = explainer.shap_values(x_sc)
        sv_class1 = sv[1][0] if isinstance(sv, list) else sv[0]

        feat_names = (
            [f"MFCC_mean_{i}" for i in range(40)] +
            [f"MFCC_std_{i}"  for i in range(40)] +
            [f"Chroma_mean_{i}" for i in range(12)] +
            [f"Chroma_std_{i}"  for i in range(12)] +
            [f"Mel_{i}"         for i in range(87)] +
            ["ZCR", "RMS"]
        )[:len(sv_class1)]

        # En etkili 10 özellik
        idx = np.argsort(np.abs(sv_class1))[::-1][:10]
        return {feat_names[i]: round(float(sv_class1[i]), 5) for i in idx}

    except Exception as exc:
        logger.warning("SHAP hesaplama hatası: %s", exc)
        return None
