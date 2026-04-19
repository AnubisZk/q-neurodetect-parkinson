"""Response schemas — Phase 5 güncellemesi."""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class ModalityResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    modality: str
    probability: float = Field(..., ge=0.0, le=1.0)
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_type: str
    features_used: list[str] = []
    notes: Optional[str] = None


class RiskLevel(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    level: str   # "High" | "Moderate" | "Low"
    label: str   # "Parkinson" | "Healthy"


class ComparisonEntry(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name: str
    model_type: str
    probability: float
    label: str


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    request_id:   str
    modalities:   list[ModalityResult]
    fusion:       RiskLevel
    model_comparison: list[ComparisonEntry]
    explanation:  str

    # Phase 5 — yeni alanlar
    modality_contributions: dict[str, float] = {}   # {"voice": 38.9, "mri": 44.4, ...}
    fusion_method:     str = "weighted"              # hangi yöntem kullanıldı
    missing_modalities: list[str] = []              # yüklenmeyen modaliteler
    risk_factors:      dict[str, list[str]] = {}    # {"risk": [...], "protective": [...]}
    shap_values:       Optional[dict[str, float]] = None
    report_url:        Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: dict[str, bool]
