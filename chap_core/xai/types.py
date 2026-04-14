"""
Data types for XAI explanations.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExplanationMethod(StrEnum):
    PERMUTATION_IMPORTANCE = "permutation_importance"
    CORRELATION = "correlation"
    OCCLUSION = "occlusion"
    LIME = "lime"
    SHAP = "shap"


class FeatureAttribution(BaseModel):
    feature_name: str
    importance: float
    direction: str | None = None
    baseline_value: float | None = None
    actual_value: float | None = None

    model_config = ConfigDict(populate_by_name=True)


class GlobalExplanation(BaseModel):
    method: ExplanationMethod
    top_features: list[FeatureAttribution]
    computed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    n_samples: int = 0
    stability_score: float | None = None

    def to_meta_dict(self) -> dict[str, Any]:
        return {
            "xai": {
                "global": {
                    "method": self.method.value,
                    "topFeatures": [f.model_dump() for f in self.top_features],
                    "computedAt": self.computed_at.isoformat(),
                    "nSamples": self.n_samples,
                    "stabilityScore": self.stability_score,
                }
            }
        }

    @classmethod
    def from_meta_dict(cls, meta: dict[str, Any]) -> "GlobalExplanation | None":
        if not meta or "xai" not in meta or "global" not in meta.get("xai", {}):
            return None
        g = meta["xai"]["global"]
        return cls(
            method=ExplanationMethod(g["method"]),
            top_features=[FeatureAttribution(**f) for f in g.get("topFeatures", [])],
            computed_at=datetime.fromisoformat(g["computedAt"]),
            n_samples=g.get("nSamples", 0),
            stability_score=g.get("stabilityScore"),
        )


class LocalExplanation(BaseModel):
    prediction_id: int
    org_unit: str
    period: str
    method: ExplanationMethod
    output_statistic: str = "median"
    feature_attributions: list[FeatureAttribution]
    baseline_prediction: float
    actual_prediction: float
    computed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
