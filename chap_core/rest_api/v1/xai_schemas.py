from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from chap_core.database.base_tables import DBModel


class GlobalExplanationResponse(DBModel):
    method: str
    top_features: list[dict[str, Any]]
    computed_at: datetime | None = None
    n_samples: int = 0
    stability_score: float | None = None
    available: bool = True
    surrogate_quality: dict[str, Any] | None = None


class LocalExplanationRequest(BaseModel):
    org_unit: str = Field(..., alias="orgUnit")
    period: str
    output_statistic: str = Field("median", alias="outputStatistic")
    xai_method: str = Field("shap_auto", alias="xaiMethod")
    top_k: int = Field(10, alias="topK")
    force: bool = False

    model_config = ConfigDict(populate_by_name=True)


class LocalExplanationResponse(DBModel):
    id: int | None = None
    prediction_id: int
    org_unit: str
    period: str
    method: str
    output_statistic: str
    feature_attributions: list[dict]
    baseline_prediction: float
    actual_prediction: float
    computed_at: str | None = None
    status: str = "completed"
    surrogate_quality: dict[str, Any] | None = None
    covariate_provenance: dict[str, Any] | None = None


class RunExplanationsRequest(BaseModel):
    xai_method: str = Field("shap_auto", alias="xaiMethodName")
    output_statistic: str = Field("median", alias="outputStatistic")
    top_k: int = Field(10, alias="topK")

    model_config = ConfigDict(populate_by_name=True)


class ShapBeeswarmPoint(DBModel):
    feature_name: str
    shap_value: float
    feature_value: float
    org_unit: str
    period: str


class ShapBeeswarmResponse(DBModel):
    prediction_id: int
    output_statistic: str
    feature_names: list[str]
    points: list[ShapBeeswarmPoint]
    surrogate_quality: dict[str, Any] | None = None


class HorizonFeatureImportance(DBModel):
    feature_name: str
    importance: float
    direction: str


class HorizonStepSummary(DBModel):
    period: str
    target_period: str
    forecast_step: int
    data_source: dict[str, Any] | None = None
    feature_importances: list[HorizonFeatureImportance]
    actual_prediction: float | None = None


class AverageImportance(DBModel):
    feature_name: str
    mean_abs_importance: float
    mean_signed_importance: float
    direction: str


class HorizonSummaryResponse(DBModel):
    prediction_id: int
    org_unit: str
    method: str
    output_statistic: str
    steps: list[HorizonStepSummary]
    average_importance: list[AverageImportance]
    surrogate_quality: dict[str, Any] | None = None


class XaiMethodRead(DBModel):
    id: int
    name: str
    display_name: str
    description: str
    method_type: str
    source_url: str | None = None
    author: str
    archived: bool
    supported_visualizations: list[str]


class SurrogateQualityRead(DBModel):
    r_squared: float | None = Field(None, alias="rSquared")
    mae: float | None = None
    mape: float | None = None
    n_samples: int = Field(0, alias="nSamples")
    n_unique_rows: int = Field(0, alias="nUniqueRows")
    constant_features: list[str] = Field(default_factory=list, alias="constantFeatures")
    imputation_rates: dict[str, float] = Field(default_factory=dict, alias="imputationRates")
    removed_features: list[str] = Field(default_factory=list, alias="removedFeatures")
    selected_model_type: str | None = Field(None, alias="selectedModelType")
    selected_model_display_name: str | None = Field(None, alias="selectedModelDisplayName")
    n_groups: int | None = Field(None, alias="nGroups")
    fidelity_tier: str | None = Field(None, alias="fidelityTier")
    residual_mean: float | None = Field(None, alias="residualMean")
    residual_std: float | None = Field(None, alias="residualStd")
    target_transformed: bool = Field(False, alias="targetTransformed")
    target_transform_method: str | None = Field(None, alias="targetTransformMethod")
    permutation_removed_features: list[str] = Field(default_factory=list, alias="permutationRemovedFeatures")
    r_squared_train: float | None = Field(None, alias="rSquaredTrain")
    generalization_gap: float | None = Field(None, alias="generalizationGap")

    model_config = ConfigDict(populate_by_name=True)
