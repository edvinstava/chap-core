"""
Data types for XAI explanations.
"""

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field


class FeatureAttribution(BaseModel):
    feature_name: str
    importance: float
    direction: str | None = None
    baseline_value: float | None = None
    actual_value: float | None = None

    model_config = ConfigDict(populate_by_name=True)


class GlobalExplanation(BaseModel):
    top_features: list[FeatureAttribution]
    computed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    n_samples: int = 0
    stability_score: float | None = None


class LocalExplanation(BaseModel):
    prediction_id: int
    org_unit: str
    period: str
    output_statistic: str = "median"
    feature_attributions: list[FeatureAttribution]
    baseline_prediction: float
    actual_prediction: float
    computed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
