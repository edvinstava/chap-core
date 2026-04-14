"""
XAI (Explainable AI) API endpoints for CHAP.

Provides endpoints for retrieving and computing explanations for predictions.
"""

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm.attributes import flag_modified
from sqlmodel import Session, select

from chap_core.database.base_tables import DBModel
from chap_core.database.database import SessionWrapper
from chap_core.database.tables import Prediction
from chap_core.database.xai_tables import PredictionExplanation
from chap_core.log_config import get_status_logger
from chap_core.rest_api.celery_tasks import JOB_NAME_KW, JOB_TYPE_KW, CeleryPool
from chap_core.rest_api.data_models import JobResponse
from chap_core.xai.covariate_fallback import resolve_covariate_row
from chap_core.xai.forecast_matching import find_forecast_row_index
from chap_core.xai.method_registry import XAI_METHODS as XAI_METHOD_DEFINITIONS
from chap_core.xai.types import (
    GlobalExplanation,
)

from .dependencies import get_database_url, get_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/xai", tags=["xai"])
worker: CeleryPool[Any] = CeleryPool()

_surrogate_cache: dict[tuple, Any] = {}
_surrogate_cache_lock = threading.Lock()
_SURROGATE_CACHE_MAX = 20


def _get_cached_surrogate(key: tuple) -> Any | None:
    with _surrogate_cache_lock:
        return _surrogate_cache.get(key)


def _put_cached_surrogate(key: tuple, explainer: Any) -> None:
    with _surrogate_cache_lock:
        if len(_surrogate_cache) >= _SURROGATE_CACHE_MAX:
            oldest = next(iter(_surrogate_cache))
            del _surrogate_cache[oldest]
        _surrogate_cache[key] = explainer


# ---------------------------------------------------------------------------
# Response / request models
# ---------------------------------------------------------------------------


class GlobalExplanationResponse(DBModel):
    method: str
    top_features: list[dict]
    computed_at: str | None = None
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
    xai_method_name: str | None = None
    output_statistic: str
    feature_attributions: list[dict]
    baseline_prediction: float
    actual_prediction: float
    computed_at: str | None = None
    status: str = "completed"
    surrogate_quality: dict[str, Any] | None = None
    covariate_provenance: dict[str, Any] | None = None


class RunExplanationsRequest(BaseModel):
    xai_method_name: str = Field("shap_auto", alias="xaiMethodName")
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
    fidelity_warning: str | None = Field(None, alias="fidelityWarning")
    residual_mean: float | None = Field(None, alias="residualMean")
    residual_std: float | None = Field(None, alias="residualStd")
    target_transformed: bool = Field(False, alias="targetTransformed")
    target_transform_method: str | None = Field(None, alias="targetTransformMethod")
    permutation_removed_features: list[str] = Field(default_factory=list, alias="permutationRemovedFeatures")
    r_squared_train: float | None = Field(None, alias="rSquaredTrain")

    model_config = ConfigDict(populate_by_name=True)


@dataclass
class ForecastLookupRow:
    org_unit: str
    period: str


_XAI_METHODS = [XaiMethodRead(**definition) for definition in XAI_METHOD_DEFINITIONS]
_XAI_METHOD_BY_NAME = {m.name: m for m in _XAI_METHODS}

# Maps xai_method name -> surrogate model_type for SurrogateSHAPExplainer
_METHOD_TO_MODEL_TYPE: dict[str, str] = {
    "shap_auto": "auto",
    "shap_xgboost": "xgboost",
    "shap_lightgbm": "lightgbm",
    "shap_hist_gradient_boosting": "hist_gradient_boosting",
    "shap_gradient_boosting": "gradient_boosting",
    "shap_random_forest": "random_forest",
    "shap_extra_trees": "extra_trees",
    "lime": "auto",
    "lime_auto": "auto",
}


# ---------------------------------------------------------------------------
# Quality dict helper
# ---------------------------------------------------------------------------


def _quality_response_dict(quality: dict[str, Any] | None) -> dict[str, Any] | None:
    if not quality:
        return None
    return SurrogateQualityRead.model_validate(quality).model_dump(by_alias=True)


# ---------------------------------------------------------------------------
# Surrogate helpers
# ---------------------------------------------------------------------------


def _build_surrogate_data(
    forecasts: list,
    dataset: Any,
    feature_names: list[str],
    output_statistic: str = "median",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float], list[dict[str, Any]]]:
    """Build (X, y, groups, imputation_rates) for surrogate training.

    One row per forecast, feature values matched from dataset by (org_unit, period).
    Period ids like ``YYYYMM_k`` resolve to the calendar month at horizon *k*
    when the exact id is missing from the dataset.
    groups is an int array of org_unit indices for grouped cross-validation.
    """
    df = dataset.to_pandas()
    has_location = "location" in df.columns
    period_col = next((c for c in ["time_period", "period", "date"] if c in df.columns), None)

    rows: list[dict] = []
    org_units: list[str] = []
    forecast_value_lists: list[list[float]] = []
    covariate_provenance_rows: list[dict[str, Any]] = []

    for fc in forecasts:
        loc_df = df[df["location"] == fc.org_unit] if has_location else df

        pcol = period_col if period_col is not None and period_col in loc_df.columns else None
        row, prov = resolve_covariate_row(
            loc_df,
            pcol or "",
            feature_names,
            fc.period,
            fc.org_unit,
            df,
        )
        rows.append(row)
        covariate_provenance_rows.append(prov)
        org_units.append(fc.org_unit)
        forecast_value_lists.append(list(fc.values))

    # Build X with median imputation
    columns = []
    imputation_rates: dict[str, float] = {}
    for name in feature_names:
        col = np.array([r.get(name, np.nan) for r in rows], dtype=float)
        n_nan = int(np.sum(np.isnan(col)))
        imputation_rates[name] = n_nan / len(col) if len(col) > 0 else 0.0
        if np.any(np.isnan(col)):
            fill = float(np.nanmedian(col)) if not np.all(np.isnan(col)) else 0.0
            col = np.where(np.isnan(col), fill, col)
        columns.append(col)
    X = np.column_stack(columns) if columns else np.zeros((len(rows), 1))

    # Build y
    if output_statistic == "mean":
        y = np.array([np.mean(v) for v in forecast_value_lists], dtype=float)
    elif output_statistic.startswith("q"):
        try:
            q = float(output_statistic[1:]) / 100.0
        except ValueError:
            q = 0.5
        y = np.array([np.quantile(v, q) for v in forecast_value_lists], dtype=float)
    else:
        y = np.array([np.median(v) for v in forecast_value_lists], dtype=float)

    unique_orgs = list(dict.fromkeys(org_units))
    org_to_idx = {org: i for i, org in enumerate(unique_orgs)}
    groups = np.array([org_to_idx[org] for org in org_units])

    return X, y, groups, imputation_rates, covariate_provenance_rows


def _fit_surrogate(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_type: str,
    feature_names: list[str],
    imputation_rates: dict[str, float],
    xai_method_name: str = "",
    cache_key: tuple | None = None,
) -> Any:
    """Fit a surrogate explainer with optional Optuna hyperparameter tuning.

    Pipeline order: filter features -> model selection -> tuning -> fit.
    All steps operate on the same filtered feature matrix.

    When cache_key is provided, returns a cached explainer if one exists for
    that key and stores the newly fitted explainer in the cache otherwise.
    """
    from chap_core.xai.shap_explainer import (
        MIN_SAMPLES_FOR_TUNING,
        SurrogateLIMEExplainer,
        SurrogateSHAPExplainer,
        filter_features,
        tune_hyperparameters,
    )
    from chap_core.xai.surrogate_model import auto_select_best_model_type, select_and_tune_best_model_type

    if cache_key is not None:
        cached = _get_cached_surrogate(cache_key)
        if cached is not None:
            logger.info("Returning cached surrogate for key %s", cache_key)
            return cached

    status_logger = get_status_logger()
    is_lime = xai_method_name in ("lime", "lime_auto")

    # Step 1: Filter features (constant removal, imputation removal, permutation selection)
    fr = filter_features(
        X,
        y,
        feature_names,
        imputation_rates,
        model_type=model_type,
    )
    X_filtered = fr.X_filtered
    kept_feature_names = fr.kept_feature_names

    status_logger.info("Building surrogate on %d forecasts, features: %s", len(X_filtered), kept_feature_names)

    n_rows = len(X_filtered)
    if n_rows >= 200:
        n_trials = 300
    elif n_rows >= 100:
        n_trials = 200
    elif n_rows >= 30:
        n_trials = 100
    else:
        n_trials = max(15, min(40, n_rows))

    # Step 2: Model selection + tuning on filtered features
    effective_model_type = model_type
    hyperparams: dict = {}
    if model_type == "auto" and len(X_filtered) >= 4:
        if n_rows >= MIN_SAMPLES_FOR_TUNING:
            status_logger.info(
                "Selecting and tuning best surrogate model (LOO + Optuna, %d trials total)...",
                n_trials,
            )
            try:
                effective_model_type, hyperparams = select_and_tune_best_model_type(
                    X_filtered, y, groups=groups, n_trials=n_trials
                )
                status_logger.info("Selected surrogate: %s (tuned CV)", effective_model_type)
                logger.info("Tuned hyperparameters: %s", hyperparams)
            except Exception as e:
                logger.warning("Select+tune failed, falling back to LOO selection: %s", e)
                top = auto_select_best_model_type(X_filtered, y, groups=groups)
                effective_model_type = top[0]
                status_logger.info("Auto-selected surrogate: %s (LOO fallback)", effective_model_type)
        else:
            top = auto_select_best_model_type(X_filtered, y, groups=groups)
            effective_model_type = top[0]
            status_logger.info("Auto-selected surrogate: %s (LOO, dataset too small for tuning)", effective_model_type)
    elif n_rows >= MIN_SAMPLES_FOR_TUNING:
        try:
            status_logger.info(
                "Tuning hyperparameters for %s (Optuna, %d trials)...",
                effective_model_type,
                n_trials,
            )
            hyperparams = tune_hyperparameters(
                X_filtered, y, model_type=effective_model_type, groups=groups, n_trials=n_trials
            )
            logger.info("Tuned hyperparameters: %s", hyperparams)
        except Exception as e:
            logger.warning("Hyperparameter tuning failed, using defaults: %s", e)

    # Step 4: Fit on filtered features (skip re-filtering via filter_result)
    status_logger.info("Fitting surrogate model on %d samples...", len(X_filtered))
    explainer_cls = SurrogateLIMEExplainer if is_lime else SurrogateSHAPExplainer
    explainer = explainer_cls(
        feature_names=feature_names,
        model_config={"model_type": effective_model_type},
        hyperparams=hyperparams,
        imputation_rates=imputation_rates,
    )
    explainer.fit(X, y, groups=groups, filter_result=fr)

    if cache_key is not None:
        _put_cached_surrogate(cache_key, explainer)

    return explainer


# ---------------------------------------------------------------------------
# Native SHAP helpers
# ---------------------------------------------------------------------------


def _has_native_shap(prediction: Any) -> bool:
    """Return True if the prediction has stored native SHAP values."""
    return bool((prediction.meta_data or {}).get("native_shap"))


def _native_shap_global_response(
    prediction_id: int, prediction: Any, xai_method: str
) -> GlobalExplanationResponse | None:
    """Return a GlobalExplanationResponse from stored native SHAP metadata, or None."""
    entry = (prediction.meta_data or {}).get("xai", {}).get("global_by_method", {}).get(xai_method)
    if entry is None:
        return None
    return GlobalExplanationResponse(
        method=xai_method,
        top_features=entry.get("topFeatures", []),
        computed_at=entry.get("computedAt"),
        n_samples=entry.get("nSamples", 0),
        stability_score=entry.get("stabilityScore"),
        available=True,
        surrogate_quality=None,
    )


def _native_shap_local_response(
    prediction_id: int,
    org_unit: str,
    period: str,
    output_statistic: str,
    prediction: Any,
    session: Any,
) -> "LocalExplanationResponse | None":
    """Return a LocalExplanationResponse from stored native SHAP data, or None."""
    native_shap = (prediction.meta_data or {}).get("native_shap")
    if not native_shap:
        return None

    feature_names = native_shap.get("feature_names", [])
    values = native_shap.get("values", [])
    shap_rows = [
        ForecastLookupRow(org_unit=v.get("location", ""), period=str(v.get("time_period", ""))) for v in values
    ]
    idx = find_forecast_row_index(shap_rows, org_unit, period)
    entry = values[idx] if idx is not None and 0 <= idx < len(values) else None

    if entry is None:
        return None

    shap_vals = entry["shap_values"]
    feature_values = entry.get("feature_values") or {}
    expected_value = float(entry.get("expected_value", native_shap.get("expected_value", 0.0)))
    actual_prediction = expected_value + float(np.sum(shap_vals))
    feature_attributions = [
        {
            "feature_name": fn,
            "importance": float(shap_vals[i]),
            "direction": "positive" if shap_vals[i] >= 0 else "negative",
            "baseline_value": None,
            "actual_value": (
                float(raw_feature_value) if (raw_feature_value := feature_values.get(fn)) is not None else None
            ),
        }
        for i, fn in enumerate(feature_names)
    ]
    return LocalExplanationResponse(
        prediction_id=prediction_id,
        org_unit=org_unit,
        period=period,
        method="native_shap",
        xai_method_name="native_shap",
        output_statistic=output_statistic,
        feature_attributions=feature_attributions,
        baseline_prediction=expected_value,
        actual_prediction=actual_prediction,
        surrogate_quality=None,
        covariate_provenance=None,
    )


def _native_shap_beeswarm(
    prediction_id: int,
    output_statistic: str,
    prediction: Any,
    dataset: Any,
) -> ShapBeeswarmResponse | None:
    """Build a beeswarm response directly from native SHAP metadata."""
    native_shap = (prediction.meta_data or {}).get("native_shap")
    if not native_shap:
        return None
    feature_names = native_shap.get("feature_names", [])
    df = dataset.to_pandas()
    has_location = "location" in df.columns
    period_col = next((c for c in ["time_period", "period", "date"] if c in df.columns), None)
    points: list[ShapBeeswarmPoint] = []
    for entry in native_shap.get("values", []):
        shap_vals = entry["shap_values"]
        feature_values = entry.get("feature_values") or {}
        org_unit = str(entry.get("location", ""))
        period = str(entry.get("time_period", ""))
        loc_df = df[df["location"] == org_unit] if has_location else df
        row, _ = resolve_covariate_row(
            loc_df,
            period_col or "",
            feature_names,
            period,
            org_unit,
            df,
        )
        for i, fn in enumerate(feature_names):
            if fn in feature_values and feature_values.get(fn) is not None:
                feature_value = float(feature_values[fn])
            else:
                raw_value = row.get(fn, 0.0)
                feature_value = float(raw_value) if raw_value is not None and not pd.isna(raw_value) else 0.0
            points.append(
                ShapBeeswarmPoint(
                    feature_name=fn,
                    shap_value=float(shap_vals[i]),
                    feature_value=feature_value,
                    org_unit=org_unit,
                    period=period,
                )
            )
    return ShapBeeswarmResponse(
        prediction_id=prediction_id,
        output_statistic=output_statistic,
        feature_names=feature_names,
        points=points,
        surrogate_quality=None,
    )


# ---------------------------------------------------------------------------
# /methods endpoints
# ---------------------------------------------------------------------------


@router.get("/methods", response_model=list[XaiMethodRead], response_model_by_alias=True)
async def list_xai_methods(
    include_archived: bool = Query(False, alias="includeArchived"),
    prediction_id: int | None = Query(None, alias="predictionId"),
    session: Session = Depends(get_session),
):
    methods = _XAI_METHODS if include_archived else [m for m in _XAI_METHODS if not m.archived]

    if prediction_id is not None:
        prediction = session.get(Prediction, prediction_id)
        has_native = prediction is not None and _has_native_shap(prediction)
        if not has_native:
            methods = [m for m in methods if m.name != "native_shap"]

    return methods


@router.get("/methods/{name}", response_model=XaiMethodRead, response_model_by_alias=True)
async def get_xai_method(name: str):
    method = _XAI_METHOD_BY_NAME.get(name)
    if method is None:
        raise HTTPException(status_code=404, detail=f"XAI method '{name}' not found")
    return method


# ---------------------------------------------------------------------------
# Batch explanation run (async Celery job)
# ---------------------------------------------------------------------------


def _run_explanations_task(
    prediction_id: int,
    xai_method_name: str,
    output_statistic: str,
    top_k: int,
    session: SessionWrapper,
):
    """Celery task: compute global + all local explanations for a prediction."""
    status_logger = get_status_logger()
    status_logger.info(
        "Starting XAI explanations (prediction=%d, method=%s, statistic=%s)",
        prediction_id,
        xai_method_name,
        output_statistic,
    )
    prediction = session.session.get(Prediction, prediction_id)
    if prediction is None:
        raise ValueError(f"Prediction {prediction_id} not found")

    forecasts = prediction.forecasts
    if not forecasts:
        raise ValueError(f"No forecasts found for prediction {prediction_id}")

    # Fast-path: native SHAP — explanations were pre-populated at prediction time
    if xai_method_name == "native_shap":
        if not _has_native_shap(prediction):
            raise ValueError(f"Prediction {prediction_id} has no native SHAP data")
        status_logger.info("Native SHAP: explanations already stored, skipping surrogate fitting")
        return

    dataset = session.get_dataset(prediction.dataset_id)

    try:
        feature_names = [f for f in dataset.field_names() if f != "time_period"]
    except Exception:
        feature_names = []
    if not feature_names:
        feature_names = ["disease_cases", "rainfall", "mean_temperature", "population"]

    model_type = _METHOD_TO_MODEL_TYPE.get(xai_method_name, "auto")
    X, y, groups, imputation_rates, covariate_provenance_rows = _build_surrogate_data(
        forecasts, dataset, feature_names, output_statistic
    )
    explainer = _fit_surrogate(X, y, groups, model_type, feature_names, imputation_rates, xai_method_name)
    quality = explainer.quality_dict()

    if quality:
        status_logger.info(
            "Surrogate ready: type=%s, LOO-R²=%s, train-R²=%s, fidelity=%s, n_samples=%d",
            quality.get("selected_model_type"),
            quality.get("r_squared"),
            quality.get("r_squared_train"),
            quality.get("fidelity_tier"),
            quality.get("n_samples", 0),
        )

    # Global explanation
    status_logger.info("Computing global explanation...")
    global_exp = explainer.explain_global(X, top_k=top_k)
    meta_data = dict(prediction.meta_data) if prediction.meta_data else {}
    meta_data.setdefault("xai", {}).setdefault("global_by_method", {})[xai_method_name] = {
        "topFeatures": [f.model_dump() for f in global_exp.top_features],
        "computedAt": global_exp.computed_at.isoformat(),
        "nSamples": global_exp.n_samples,
        "stabilityScore": global_exp.stability_score,
        "surrogateQuality": _quality_response_dict(quality),
    }
    prediction.meta_data = meta_data
    flag_modified(prediction, "meta_data")
    session.session.add(prediction)

    status_logger.info("Computing %d local explanations (method=%s)...", len(forecasts), xai_method_name)
    for idx, fc in enumerate(forecasts):
        samples = np.array(fc.values, dtype=float)
        actual_value = float(np.mean(samples) if output_statistic == "mean" else np.median(samples))
        feature_actual_values = {name: float(X[idx, i]) for i, name in enumerate(feature_names)}

        local_exp = explainer.explain_local(
            X=X,
            instance_idx=idx,
            prediction_id=prediction_id,
            org_unit=fc.org_unit,
            period=fc.period,
            feature_actual_values=feature_actual_values,
            top_k=top_k,
            output_statistic=output_statistic,
            actual_forecast_value=actual_value,
        )

        explanation = PredictionExplanation(
            prediction_id=prediction_id,
            org_unit=fc.org_unit,
            period=fc.period,
            method=xai_method_name,
            output_statistic=output_statistic,
            params={"top_k": top_k},
            result={
                "feature_attributions": [f.model_dump() for f in local_exp.feature_attributions],
                "baseline_prediction": local_exp.baseline_prediction,
                "actual_prediction": local_exp.actual_prediction,
                "xai_method_name": xai_method_name,
                "surrogate_quality": _quality_response_dict(quality),
                "covariate_provenance": covariate_provenance_rows[idx],
            },
            status="completed",
        )
        session.session.add(explanation)

    session.session.commit()
    status_logger.info(
        "XAI explanations complete: %d explanations saved (prediction=%d)",
        len(forecasts),
        prediction_id,
    )


@router.post(
    "/predictions/{predictionId}/explanations/run",
    response_model=JobResponse,
    response_model_by_alias=True,
)
async def run_explanations(
    prediction_id: Annotated[int, Path(alias="predictionId")],
    request: RunExplanationsRequest,
    database_url: str = Depends(get_database_url),
):
    job = worker.queue_db(
        _run_explanations_task,
        prediction_id,
        request.xai_method_name,
        request.output_statistic,
        request.top_k,
        database_url=database_url,
        **{JOB_TYPE_KW: "xai_explanations", JOB_NAME_KW: f"xai_{prediction_id}"},
    )
    return JobResponse(id=job.id)


# ---------------------------------------------------------------------------
# Global explanation endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/predictions/{predictionId}/global",
    response_model=GlobalExplanationResponse,
    response_model_by_alias=True,
)
async def get_global_explanation(
    prediction_id: Annotated[int, Path(alias="predictionId")],
    xai_method: str | None = Query(None, alias="xaiMethod"),
    session: Session = Depends(get_session),
):
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    meta = prediction.meta_data or {}

    # Look up per-method cache first
    if xai_method:
        entry = meta.get("xai", {}).get("global_by_method", {}).get(xai_method)
        if entry:
            return GlobalExplanationResponse(
                method=xai_method,
                top_features=entry.get("topFeatures", []),
                computed_at=entry.get("computedAt"),
                n_samples=entry.get("nSamples", 0),
                stability_score=entry.get("stabilityScore"),
                available=True,
                surrogate_quality=entry.get("surrogateQuality"),
            )

    # Fall back to legacy single-method storage
    global_exp = GlobalExplanation.from_meta_dict(meta)
    if global_exp is None:
        return GlobalExplanationResponse(method="none", top_features=[], available=False, n_samples=0)

    return GlobalExplanationResponse(
        method=global_exp.method.value,
        top_features=[f.model_dump() for f in global_exp.top_features],
        computed_at=global_exp.computed_at.isoformat(),
        n_samples=global_exp.n_samples,
        stability_score=global_exp.stability_score,
        available=True,
    )


@router.post(
    "/predictions/{predictionId}/global",
    response_model=GlobalExplanationResponse,
    response_model_by_alias=True,
)
async def compute_global_explanation(
    prediction_id: Annotated[int, Path(alias="predictionId")],
    top_k: int = Query(10, alias="topK"),
    xai_method: str = Query("shap_auto", alias="xaiMethod"),
    output_statistic: str = Query("median", alias="outputStatistic"),
    force: bool = Query(False),
    session: Session = Depends(get_session),
):
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    if not force:
        entry = (prediction.meta_data or {}).get("xai", {}).get("global_by_method", {}).get(xai_method)
        if entry:
            return GlobalExplanationResponse(
                method=xai_method,
                top_features=entry.get("topFeatures", []),
                computed_at=entry.get("computedAt"),
                n_samples=entry.get("nSamples", 0),
                stability_score=entry.get("stabilityScore"),
                available=True,
                surrogate_quality=entry.get("surrogateQuality"),
            )

    forecasts = prediction.forecasts
    if not forecasts:
        raise HTTPException(status_code=400, detail="No forecasts found for prediction")

    session_wrapper = SessionWrapper(session=session)
    dataset = session_wrapper.get_dataset(prediction.dataset_id)

    try:
        feature_names = [f for f in dataset.field_names() if f != "time_period"]
    except Exception:
        feature_names = []
    if not feature_names:
        feature_names = ["disease_cases", "rainfall", "mean_temperature", "population"]

    try:
        if xai_method == "native_shap":
            resp = _native_shap_global_response(prediction_id, prediction, xai_method)
            if resp is None:
                raise HTTPException(status_code=404, detail="No native SHAP data for this prediction")
            return resp

        if xai_method == "occlusion":
            return _compute_global_occlusion(prediction, forecasts, dataset, feature_names, top_k, session)

        model_type = _METHOD_TO_MODEL_TYPE.get(xai_method, "auto")
        X, y, groups, imputation_rates, _ = _build_surrogate_data(forecasts, dataset, feature_names, output_statistic)
        cache_key = (prediction_id, xai_method, output_statistic)
        explainer = _fit_surrogate(X, y, groups, model_type, feature_names, imputation_rates, cache_key=cache_key)
        global_exp = explainer.explain_global(X, top_k=top_k)
        quality = explainer.quality_dict()

        # Persist per-method in meta_data
        meta_data = dict(prediction.meta_data) if prediction.meta_data else {}
        meta_data.setdefault("xai", {}).setdefault("global_by_method", {})[xai_method] = {
            "topFeatures": [f.model_dump() for f in global_exp.top_features],
            "computedAt": global_exp.computed_at.isoformat(),
            "nSamples": global_exp.n_samples,
            "stabilityScore": global_exp.stability_score,
            "surrogateQuality": quality or None,
        }
        prediction.meta_data = meta_data
        flag_modified(prediction, "meta_data")
        session.add(prediction)
        session.commit()

        return GlobalExplanationResponse(
            method=xai_method,
            top_features=[f.model_dump() for f in global_exp.top_features],
            computed_at=global_exp.computed_at.isoformat(),
            n_samples=global_exp.n_samples,
            stability_score=global_exp.stability_score,
            available=True,
            surrogate_quality=quality or None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error computing global explanation: %s", e)
        raise HTTPException(status_code=500, detail=f"Error computing explanation: {e!s}") from e


def _compute_global_occlusion(
    prediction: Any,
    forecasts: list,
    dataset: Any,
    feature_names: list[str],
    top_k: int,
    session: Session,
) -> GlobalExplanationResponse:
    """Legacy PerturbationExplainer path for the occlusion method."""
    from chap_core.xai import PerturbationExplainer

    all_values = np.array([f.values for f in forecasts])
    median_values = np.median(all_values, axis=-1)

    df = dataset.to_pandas()
    X: dict = {}
    for feature in feature_names:
        if feature in df.columns:
            feature_data = df[feature].dropna().values
            if len(feature_data) >= len(forecasts):
                X[feature] = feature_data[: len(forecasts)].astype(float)
            else:
                padded = np.zeros(len(forecasts))
                padded[: len(feature_data)] = feature_data.astype(float)
                X[feature] = padded
        else:
            X[feature] = np.random.randn(len(forecasts))

    feature_means = {f: np.mean(X[f]) for f in feature_names}
    feature_stds = {f: np.std(X[f]) or 1.0 for f in feature_names}
    original_normalized = {f: (X[f] - feature_means[f]) / feature_stds[f] for f in feature_names}
    raw_weights = {
        f: (float(np.abs(np.corrcoef(X[f], median_values)[0, 1])) if feature_stds[f] > 0 else 0.0)
        for f in feature_names
    }
    raw_weights = {f: v if not np.isnan(v) else 0.1 for f, v in raw_weights.items()}
    total = sum(raw_weights.values()) or 1.0
    feature_weights = {f: v / total for f, v in raw_weights.items()}
    base_std = float(np.std(median_values)) or 1.0

    def predict_fn(features: dict) -> np.ndarray:
        n = len(median_values)
        result = median_values.copy()
        for f in feature_names:
            if f not in features:
                continue
            f_vals = features[f]
            f_mean = feature_means[f]
            if len(f_vals) < n:
                f_vals = np.pad(f_vals, (0, n - len(f_vals)), constant_values=f_mean)
            elif len(f_vals) > n:
                f_vals = f_vals[:n]
            diff = (f_vals - f_mean) / feature_stds[f] - original_normalized[f]
            result = result + diff * feature_weights[f] * base_std
        return np.asarray(result)

    explainer = PerturbationExplainer(predict_fn=predict_fn, feature_names=feature_names, n_repeats=5)
    global_exp = explainer.explain_global(X, top_k=top_k)

    meta_data = prediction.meta_data or {}
    meta_data.update(global_exp.to_meta_dict())
    meta_data.setdefault("xai", {}).setdefault("global_by_method", {})["occlusion"] = {
        "topFeatures": [f.model_dump() for f in global_exp.top_features],
        "computedAt": global_exp.computed_at.isoformat(),
        "nSamples": global_exp.n_samples,
        "stabilityScore": global_exp.stability_score,
        "surrogateQuality": None,
    }
    prediction.meta_data = meta_data
    session.add(prediction)
    session.commit()

    return GlobalExplanationResponse(
        method="occlusion",
        top_features=[f.model_dump() for f in global_exp.top_features],
        computed_at=global_exp.computed_at.isoformat(),
        n_samples=global_exp.n_samples,
        stability_score=global_exp.stability_score,
        available=True,
    )


# ---------------------------------------------------------------------------
# Local explanation endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/predictions/{predictionId}/local",
    response_model=list[LocalExplanationResponse],
    response_model_by_alias=True,
)
async def list_local_explanations(
    prediction_id: Annotated[int, Path(alias="predictionId")],
    org_unit: str | None = Query(None, alias="orgUnit"),
    period: str | None = None,
    xai_method: str | None = Query(None, alias="xaiMethod"),
    session: Session = Depends(get_session),
):
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    query = select(PredictionExplanation).where(PredictionExplanation.prediction_id == prediction_id)
    if org_unit:
        query = query.where(PredictionExplanation.org_unit == org_unit)
    if period:
        # Resolve horizon-step period (e.g. "202405_1") to the canonical period stored by the
        # POST endpoint.  Without this, the GET never finds explanations that were stored under
        # the calendar period derived from the horizon-step id.
        canonical_period = period
        if xai_method != "native_shap" and "_" in period and org_unit and prediction.forecasts:
            idx = find_forecast_row_index(prediction.forecasts, org_unit, period)
            if idx is not None:
                canonical_period = prediction.forecasts[idx].period
        query = query.where(PredictionExplanation.period == canonical_period)
    if xai_method:
        query = query.where(PredictionExplanation.method == xai_method)

    explanations = session.exec(query).all()

    if not explanations and xai_method == "native_shap":
        if org_unit and period:
            resp = _native_shap_local_response(
                prediction_id=prediction_id,
                org_unit=org_unit,
                period=period,
                output_statistic="median",
                prediction=prediction,
                session=session,
            )
            if resp is not None:
                return [resp]
        native_shap = (prediction.meta_data or {}).get("native_shap")
        if native_shap:
            feature_names = native_shap.get("feature_names", [])
            items: list[LocalExplanationResponse] = []
            for entry in native_shap.get("values", []):
                feature_values = entry.get("feature_values") or {}
                entry_org_unit = str(entry.get("location", ""))
                entry_period = str(entry.get("time_period", ""))
                if org_unit and entry_org_unit != org_unit:
                    continue
                if period and entry_period != period:
                    continue
                shap_vals = entry.get("shap_values", [])
                expected_value = float(entry.get("expected_value", native_shap.get("expected_value", 0.0)))
                actual_prediction = expected_value + float(np.sum(shap_vals))
                items.append(
                    LocalExplanationResponse(
                        prediction_id=prediction_id,
                        org_unit=entry_org_unit,
                        period=entry_period,
                        method="native_shap",
                        xai_method_name="native_shap",
                        output_statistic="median",
                        feature_attributions=[
                            {
                                "feature_name": fn,
                                "importance": float(shap_vals[i]),
                                "direction": "positive" if shap_vals[i] >= 0 else "negative",
                                "baseline_value": None,
                                "actual_value": (
                                    float(raw_feature_value)
                                    if (raw_feature_value := feature_values.get(fn)) is not None
                                    else None
                                ),
                            }
                            for i, fn in enumerate(feature_names)
                        ],
                        baseline_prediction=expected_value,
                        actual_prediction=actual_prediction,
                        surrogate_quality=None,
                        covariate_provenance=None,
                    )
                )
            if items:
                return items

    return [_explanation_to_response(exp) for exp in explanations]


@router.post(
    "/predictions/{predictionId}/local",
    response_model=LocalExplanationResponse,
    response_model_by_alias=True,
)
async def compute_local_explanation(
    prediction_id: Annotated[int, Path(alias="predictionId")],
    request: LocalExplanationRequest,
    session: Session = Depends(get_session),
):
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    all_forecasts = prediction.forecasts
    if not all_forecasts:
        raise HTTPException(status_code=400, detail="No forecasts found for prediction")

    # Resolve canonical stored period: forecasts may store calendar periods like "202406"
    # while the request arrives with a horizon-step ID like "202405_1".
    instance_idx = find_forecast_row_index(all_forecasts, request.org_unit, request.period)
    canonical_period = all_forecasts[instance_idx].period if instance_idx is not None else request.period

    existing = session.exec(
        select(PredictionExplanation).where(
            PredictionExplanation.prediction_id == prediction_id,
            PredictionExplanation.org_unit == request.org_unit,
            PredictionExplanation.period == canonical_period,
            PredictionExplanation.method == request.xai_method,
        )
    ).first()

    if existing and request.force:
        session.delete(existing)
        session.commit()
        existing = None

    if existing:
        return _explanation_to_response(existing)

    if instance_idx is None:
        available = list({f.org_unit for f in all_forecasts})
        raise HTTPException(
            status_code=404,
            detail=f"No forecast found for org_unit={request.org_unit}. Available: {available[:10]}",
        )

    try:
        session_wrapper = SessionWrapper(session=session)
        dataset = session_wrapper.get_dataset(prediction.dataset_id)

        try:
            feature_names = [f for f in dataset.field_names() if f != "time_period"]
        except Exception:
            feature_names = []
        if not feature_names:
            feature_names = ["disease_cases", "rainfall", "mean_temperature", "population"]

        if request.xai_method == "native_shap":
            resp = _native_shap_local_response(
                prediction_id, request.org_unit, canonical_period, request.output_statistic, prediction, session
            )
            if resp is None:
                raise HTTPException(status_code=404, detail="No native SHAP data for this prediction/period")
            return resp

        if request.xai_method == "occlusion":
            return _compute_local_occlusion(prediction_id, request, all_forecasts, dataset, feature_names, session)

        # --- Surrogate path (shap_* and lime) ---
        model_type = _METHOD_TO_MODEL_TYPE.get(request.xai_method, "auto")
        X, y, groups, imputation_rates, covariate_provenance_rows = _build_surrogate_data(
            all_forecasts, dataset, feature_names, request.output_statistic
        )
        cache_key = (prediction_id, request.xai_method, request.output_statistic)
        explainer = _fit_surrogate(
            X, y, groups, model_type, feature_names, imputation_rates, request.xai_method, cache_key=cache_key
        )

        target_forecast = all_forecasts[instance_idx]
        samples = np.array(target_forecast.values, dtype=float)
        if request.output_statistic == "mean":
            actual_value = float(np.mean(samples))
        else:
            actual_value = float(np.median(samples))

        feature_actual_values = {name: float(X[instance_idx, i]) for i, name in enumerate(feature_names)}

        local_exp = explainer.explain_local(
            X=X,
            instance_idx=instance_idx,
            prediction_id=prediction_id,
            org_unit=request.org_unit,
            period=request.period,
            feature_actual_values=feature_actual_values,
            top_k=request.top_k,
            output_statistic=request.output_statistic,
            actual_forecast_value=actual_value,
        )

        quality = explainer.quality_dict()
        explanation = PredictionExplanation(
            prediction_id=prediction_id,
            org_unit=request.org_unit,
            period=canonical_period,
            method=request.xai_method,
            output_statistic=request.output_statistic,
            params={"top_k": request.top_k},
            result={
                "feature_attributions": [f.model_dump() for f in local_exp.feature_attributions],
                "baseline_prediction": local_exp.baseline_prediction,
                "actual_prediction": local_exp.actual_prediction,
                "xai_method_name": request.xai_method,
                "surrogate_quality": quality or None,
                "covariate_provenance": covariate_provenance_rows[instance_idx],
            },
            status="completed",
        )
        session.add(explanation)
        session.commit()
        session.refresh(explanation)
        return _explanation_to_response(explanation)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error computing local explanation: %s", e)
        raise HTTPException(status_code=500, detail=f"Error computing explanation: {e!s}") from e


def _compute_local_occlusion(
    prediction_id: int,
    request: LocalExplanationRequest,
    all_forecasts: list,
    dataset: Any,
    feature_names: list[str],
    session: Session,
) -> "LocalExplanationResponse":
    """Legacy PerturbationExplainer path for the occlusion method."""
    from chap_core.xai import PerturbationExplainer

    instance_idx = find_forecast_row_index(all_forecasts, request.org_unit, request.period)
    forecasts_for_unit = [f for f in all_forecasts if f.org_unit == request.org_unit]
    target_forecast = all_forecasts[instance_idx] if instance_idx is not None else None
    if target_forecast is None:
        available = list({f.org_unit for f in all_forecasts})
        raise HTTPException(
            status_code=404,
            detail=f"No forecast for org_unit={request.org_unit}. Available: {available[:10]}",
        )

    df = dataset.to_pandas()
    location_df = df[df["location"] == request.org_unit] if "location" in df.columns else df
    n_samples = max(len(forecasts_for_unit), 1)

    X: dict = {}
    for feature in feature_names:
        if feature in location_df.columns:
            data = location_df[feature].dropna().values
            if len(data) >= n_samples:
                X[feature] = data[-n_samples:].astype(float)
            elif len(data) > 0:
                X[feature] = np.pad(data.astype(float), (0, n_samples - len(data)), mode="edge")
            else:
                X[feature] = np.random.randn(n_samples)
        else:
            X[feature] = np.random.randn(n_samples)

    samples = np.array(target_forecast.values, dtype=float)
    actual_value = float(np.median(samples) if request.output_statistic != "mean" else np.mean(samples))

    feature_means = {f: np.mean(X[f]) for f in feature_names}
    feature_stds = {f: np.std(X[f]) or 1.0 for f in feature_names}
    original_normalized = {f: (X[f] - feature_means[f]) / feature_stds[f] for f in feature_names}
    raw_weights = {f: min(feature_stds[f] / (abs(feature_means[f]) + 1e-10), 1.0) for f in feature_names}
    total = sum(raw_weights.values()) or 1.0
    feature_weights = {f: v / total for f, v in raw_weights.items()}
    base_values = np.full(n_samples, actual_value)

    def predict_fn(features: dict) -> np.ndarray:
        result = base_values.copy()
        for f in feature_names:
            if f not in features or len(features[f]) != n_samples:
                continue
            diff = (features[f] - feature_means[f]) / feature_stds[f] - original_normalized[f]
            result = result + diff * feature_weights[f] * actual_value * 0.2
        return result

    explainer = PerturbationExplainer(predict_fn=predict_fn, feature_names=feature_names)
    local_exp = explainer.explain_local(
        X=X,
        prediction_id=prediction_id,
        org_unit=request.org_unit,
        period=request.period,
        target_idx=0,
        top_k=request.top_k,
    )

    explanation = PredictionExplanation(
        prediction_id=prediction_id,
        org_unit=request.org_unit,
        period=request.period,
        method="occlusion",
        output_statistic=request.output_statistic,
        params={"top_k": request.top_k},
        result={
            "feature_attributions": [f.model_dump() for f in local_exp.feature_attributions],
            "baseline_prediction": local_exp.baseline_prediction,
            "actual_prediction": local_exp.actual_prediction,
            "xai_method_name": "occlusion",
            "surrogate_quality": None,
        },
        status="completed",
    )
    session.add(explanation)
    session.commit()
    session.refresh(explanation)
    return _explanation_to_response(explanation)


# ---------------------------------------------------------------------------
# SHAP beeswarm endpoint
# ---------------------------------------------------------------------------


def _beeswarm_from_stored(
    prediction_id: int,
    output_statistic: str,
    explanations: Sequence[PredictionExplanation],
) -> ShapBeeswarmResponse:
    """Build a beeswarm response from stored PredictionExplanation rows."""
    points: list[ShapBeeswarmPoint] = []
    feature_names_seen: list[str] = []
    quality = None

    for exp in explanations:
        result = exp.result or {}
        if quality is None:
            quality = result.get("surrogate_quality")
        for attr in result.get("feature_attributions", []):
            fname = attr.get("feature_name", "")
            if not fname:
                continue
            points.append(
                ShapBeeswarmPoint(
                    feature_name=fname,
                    shap_value=float(attr.get("importance", 0.0)),
                    feature_value=float(attr.get("actual_value") or 0.0),
                    org_unit=exp.org_unit,
                    period=exp.period,
                )
            )
            if fname not in feature_names_seen:
                feature_names_seen.append(fname)

    return ShapBeeswarmResponse(
        prediction_id=prediction_id,
        output_statistic=output_statistic,
        feature_names=feature_names_seen,
        points=points,
        surrogate_quality=quality,
    )


@router.post(
    "/predictions/{predictionId}/shap-beeswarm",
    response_model=ShapBeeswarmResponse,
    response_model_by_alias=True,
)
async def compute_shap_beeswarm(
    prediction_id: Annotated[int, Path(alias="predictionId")],
    output_statistic: str = Query("median", alias="outputStatistic"),
    xai_method: str = Query("shap_auto", alias="xaiMethod"),
    session: Session = Depends(get_session),
):
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    # Return from stored explanations when available — avoids re-fitting the surrogate
    stored = session.exec(
        select(PredictionExplanation).where(
            PredictionExplanation.prediction_id == prediction_id,
            PredictionExplanation.method == xai_method,
            PredictionExplanation.output_statistic == output_statistic,
        )
    ).all()
    if stored:
        return _beeswarm_from_stored(prediction_id, output_statistic, stored)

    forecasts = prediction.forecasts
    if not forecasts:
        raise HTTPException(status_code=400, detail="No forecasts found for prediction")

    session_wrapper = SessionWrapper(session=session)
    dataset = session_wrapper.get_dataset(prediction.dataset_id)

    try:
        feature_names = [f for f in dataset.field_names() if f != "time_period"]
    except Exception:
        feature_names = []
    if not feature_names:
        feature_names = ["disease_cases", "rainfall", "mean_temperature", "population"]

    try:
        if xai_method == "native_shap":
            resp = _native_shap_beeswarm(prediction_id, output_statistic, prediction, dataset)
            if resp is None:
                raise HTTPException(status_code=404, detail="No native SHAP data for this prediction")
            return resp

        model_type = _METHOD_TO_MODEL_TYPE.get(xai_method, "auto")
        X, y, groups, imputation_rates, _ = _build_surrogate_data(forecasts, dataset, feature_names, output_statistic)
        explainer = _fit_surrogate(X, y, groups, model_type, feature_names, imputation_rates, xai_method)

        points: list[ShapBeeswarmPoint] = []
        for i, fc in enumerate(forecasts):
            samples = np.array(fc.values, dtype=float)
            actual = float(np.mean(samples) if output_statistic == "mean" else np.median(samples))
            feature_actual_values = {name: float(X[i, j]) for j, name in enumerate(feature_names)}
            local_exp = explainer.explain_local(
                X=X,
                instance_idx=i,
                prediction_id=prediction_id,
                org_unit=fc.org_unit,
                period=fc.period,
                feature_actual_values=feature_actual_values,
                top_k=len(feature_names),
                output_statistic=output_statistic,
                actual_forecast_value=actual,
            )
            attr_by_name = {a.feature_name: a.importance for a in local_exp.feature_attributions}
            for j, fname in enumerate(feature_names):
                points.append(
                    ShapBeeswarmPoint(
                        feature_name=fname,
                        shap_value=float(attr_by_name.get(fname, 0.0)),
                        feature_value=float(X[i, j]),
                        org_unit=fc.org_unit,
                        period=fc.period,
                    )
                )

        return ShapBeeswarmResponse(
            prediction_id=prediction_id,
            output_statistic=output_statistic,
            feature_names=feature_names,
            points=points,
            surrogate_quality=_quality_response_dict(explainer.quality_dict()),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error computing SHAP beeswarm: %s", e)
        raise HTTPException(status_code=500, detail=f"Error computing beeswarm: {e!s}") from e


# ---------------------------------------------------------------------------
# Horizon summary endpoint
# ---------------------------------------------------------------------------


def _horizon_summary_from_stored(
    prediction_id: int,
    org_unit: str,
    method: str,
    output_statistic: str,
    stored: Sequence[PredictionExplanation],
) -> HorizonSummaryResponse:
    """Build a HorizonSummaryResponse from stored PredictionExplanation rows."""
    stored_sorted = sorted(stored, key=lambda e: e.period)
    steps: list[HorizonStepSummary] = []
    all_importances: dict[str, list[float]] = {}
    quality = None

    for step_num, exp in enumerate(stored_sorted, start=1):
        result = exp.result or {}
        if quality is None:
            quality = result.get("surrogate_quality")
        feat_imps: list[HorizonFeatureImportance] = []
        for attr in result.get("feature_attributions", []):
            fname = attr.get("feature_name", "")
            if not fname:
                continue
            val = float(attr.get("importance", 0.0))
            all_importances.setdefault(fname, []).append(val)
            feat_imps.append(
                HorizonFeatureImportance(
                    feature_name=fname,
                    importance=abs(val),
                    direction="positive" if val >= 0 else "negative",
                )
            )
        feat_imps.sort(key=lambda x: x.importance, reverse=True)
        steps.append(
            HorizonStepSummary(
                period=exp.period,
                target_period=exp.period,
                forecast_step=step_num,
                feature_importances=feat_imps,
                actual_prediction=result.get("actual_prediction"),
            )
        )

    avg_importance: list[AverageImportance] = []
    for fname, vals in all_importances.items():
        mean_signed = float(np.mean(vals)) if vals else 0.0
        mean_abs = float(np.mean(np.abs(vals))) if vals else 0.0
        avg_importance.append(
            AverageImportance(
                feature_name=fname,
                mean_abs_importance=mean_abs,
                mean_signed_importance=mean_signed,
                direction="positive" if mean_signed >= 0 else "negative",
            )
        )
    avg_importance.sort(key=lambda x: x.mean_abs_importance, reverse=True)

    return HorizonSummaryResponse(
        prediction_id=prediction_id,
        org_unit=org_unit,
        method=method,
        output_statistic=output_statistic,
        steps=steps,
        average_importance=avg_importance,
        surrogate_quality=quality,
    )


@router.post(
    "/predictions/{predictionId}/local/horizon-summary",
    response_model=HorizonSummaryResponse,
    response_model_by_alias=True,
)
async def compute_horizon_summary(
    prediction_id: Annotated[int, Path(alias="predictionId")],
    org_unit: str = Query(..., alias="orgUnit"),
    output_statistic: str = Query("median", alias="outputStatistic"),
    method: str = Query("shap", alias="method"),
    xai_method: str = Query("shap_auto", alias="xaiMethod"),
    session: Session = Depends(get_session),
):
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    forecasts = prediction.forecasts
    if not forecasts:
        raise HTTPException(status_code=400, detail="No forecasts found for prediction")

    session_wrapper = SessionWrapper(session=session)
    dataset = session_wrapper.get_dataset(prediction.dataset_id)

    try:
        feature_names = [f for f in dataset.field_names() if f != "time_period"]
    except Exception:
        feature_names = []
    if not feature_names:
        feature_names = ["disease_cases", "rainfall", "mean_temperature", "population"]

    # Return from stored explanations when available — avoids re-fitting the surrogate
    stored_unit = session.exec(
        select(PredictionExplanation).where(
            PredictionExplanation.prediction_id == prediction_id,
            PredictionExplanation.org_unit == org_unit,
            PredictionExplanation.method == xai_method,
            PredictionExplanation.output_statistic == output_statistic,
        )
    ).all()
    if stored_unit:
        return _horizon_summary_from_stored(prediction_id, org_unit, method, output_statistic, stored_unit)

    try:
        if xai_method == "native_shap":
            raise HTTPException(
                status_code=404,
                detail="No native SHAP explanations stored for this org_unit/output_statistic combination",
            )

        model_type = _METHOD_TO_MODEL_TYPE.get(xai_method, "auto")
        X, y, groups, imputation_rates, _ = _build_surrogate_data(forecasts, dataset, feature_names, output_statistic)
        cache_key = (prediction_id, xai_method, output_statistic)
        explainer = _fit_surrogate(X, y, groups, model_type, feature_names, imputation_rates, cache_key=cache_key)

        # Filter forecasts for the requested org_unit
        unit_entries = [(i, fc) for i, fc in enumerate(forecasts) if fc.org_unit == org_unit]
        if not unit_entries:
            available = list({fc.org_unit for fc in forecasts})
            raise HTTPException(
                status_code=404,
                detail=f"No forecasts for org_unit={org_unit}. Available: {available[:10]}",
            )

        unit_entries.sort(key=lambda x: x[1].period)

        steps: list[HorizonStepSummary] = []
        all_importances: dict[str, list[float]] = {f: [] for f in feature_names}

        for step_num, (idx, fc) in enumerate(unit_entries, start=1):
            samples = np.array(fc.values, dtype=float)
            actual = float(np.mean(samples) if output_statistic == "mean" else np.median(samples))
            feature_actual_values = {name: float(X[idx, i]) for i, name in enumerate(feature_names)}
            local_exp = explainer.explain_local(
                X=X,
                instance_idx=idx,
                prediction_id=prediction_id,
                org_unit=fc.org_unit,
                period=fc.period,
                feature_actual_values=feature_actual_values,
                top_k=len(feature_names),
                output_statistic=output_statistic,
                actual_forecast_value=actual,
            )
            attr_by_name = {a.feature_name: a.importance for a in local_exp.feature_attributions}

            feat_imps: list[HorizonFeatureImportance] = []
            for fname in feature_names:
                val = float(attr_by_name.get(fname, 0.0))
                all_importances[fname].append(val)
                feat_imps.append(
                    HorizonFeatureImportance(
                        feature_name=fname,
                        importance=abs(val),
                        direction="positive" if val >= 0 else "negative",
                    )
                )
            feat_imps.sort(key=lambda x: x.importance, reverse=True)

            steps.append(
                HorizonStepSummary(
                    period=fc.period,
                    target_period=fc.period,
                    forecast_step=step_num,
                    feature_importances=feat_imps,
                    actual_prediction=actual,
                )
            )

        avg_importance: list[AverageImportance] = []
        for fname in feature_names:
            vals = all_importances[fname]
            mean_signed = float(np.mean(vals)) if vals else 0.0
            mean_abs = float(np.mean(np.abs(vals))) if vals else 0.0
            avg_importance.append(
                AverageImportance(
                    feature_name=fname,
                    mean_abs_importance=mean_abs,
                    mean_signed_importance=mean_signed,
                    direction="positive" if mean_signed >= 0 else "negative",
                )
            )
        avg_importance.sort(key=lambda x: x.mean_abs_importance, reverse=True)

        return HorizonSummaryResponse(
            prediction_id=prediction_id,
            org_unit=org_unit,
            method=method,
            output_statistic=output_statistic,
            steps=steps,
            average_importance=avg_importance,
            surrogate_quality=_quality_response_dict(explainer.quality_dict()),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error computing horizon summary: %s", e)
        raise HTTPException(status_code=500, detail=f"Error computing horizon summary: {e!s}") from e


# ---------------------------------------------------------------------------
# Single / delete local explanation endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/predictions/{predictionId}/local/{explanationId}",
    response_model=LocalExplanationResponse,
    response_model_by_alias=True,
)
async def get_local_explanation(
    prediction_id: Annotated[int, Path(alias="predictionId")],
    explanation_id: Annotated[int, Path(alias="explanationId")],
    session: Session = Depends(get_session),
):
    explanation = session.get(PredictionExplanation, explanation_id)
    if explanation is None or explanation.prediction_id != prediction_id:
        raise HTTPException(status_code=404, detail="Explanation not found")
    return _explanation_to_response(explanation)


@router.delete("/predictions/{predictionId}/local/{explanationId}")
async def delete_local_explanation(
    prediction_id: Annotated[int, Path(alias="predictionId")],
    explanation_id: Annotated[int, Path(alias="explanationId")],
    session: Session = Depends(get_session),
):
    explanation = session.get(PredictionExplanation, explanation_id)
    if explanation is None or explanation.prediction_id != prediction_id:
        raise HTTPException(status_code=404, detail="Explanation not found")
    session.delete(explanation)
    session.commit()
    return {"message": "deleted"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _explanation_to_response(exp: PredictionExplanation) -> LocalExplanationResponse:
    result = exp.result or {}
    return LocalExplanationResponse(
        id=exp.id,
        prediction_id=exp.prediction_id,
        org_unit=exp.org_unit,
        period=exp.period,
        method=exp.method,
        xai_method_name=result.get("xai_method_name", exp.method),
        output_statistic=exp.output_statistic,
        feature_attributions=result.get("feature_attributions", []),
        baseline_prediction=result.get("baseline_prediction", 0),
        actual_prediction=result.get("actual_prediction", 0),
        computed_at=exp.created.isoformat() if exp.created else None,
        status=exp.status,
        surrogate_quality=result.get("surrogate_quality"),
        covariate_provenance=result.get("covariate_provenance"),
    )
