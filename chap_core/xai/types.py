"""
Data types for XAI explanations.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field


class ExplanationMethod(str, Enum):
    PERMUTATION_IMPORTANCE = "permutation_importance"
    OCCLUSION = "occlusion"


class FeatureAttribution(BaseModel):
    feature_name: str
    importance: float
    direction: Optional[str] = None
    baseline_value: Optional[float] = None
    actual_value: Optional[float] = None

    class Config:
        populate_by_name = True


class GlobalExplanation(BaseModel):
    method: ExplanationMethod
    top_features: List[FeatureAttribution]
    computed_at: datetime = Field(default_factory=datetime.utcnow)
    n_samples: int = 0
    stability_score: Optional[float] = None

    def to_meta_dict(self) -> Dict[str, Any]:
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
    def from_meta_dict(cls, meta: Dict[str, Any]) -> Optional["GlobalExplanation"]:
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
    feature_attributions: List[FeatureAttribution]
    baseline_prediction: float
    actual_prediction: float
    computed_at: datetime = Field(default_factory=datetime.utcnow)
