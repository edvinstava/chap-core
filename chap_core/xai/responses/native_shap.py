from typing import Any

import numpy as np
import pandas as pd

from chap_core.rest_api.v1.xai_schemas import (
    GlobalExplanationResponse,
    LocalExplanationResponse,
    ShapBeeswarmPoint,
    ShapBeeswarmResponse,
)
from chap_core.xai.types import ForecastLookupRow

from ..covariate_fallback import resolve_covariate_row
from ..forecast_matching import find_forecast_row_index
from ..method_registry import NATIVE_SHAP


def has_native_shap(prediction: Any) -> bool:
    return bool((prediction.meta_data or {}).get(NATIVE_SHAP))


def native_shap_global_response(
    prediction_id: int, prediction: Any, xai_method: str
) -> GlobalExplanationResponse | None:
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


def native_shap_local_response(
    prediction_id: int,
    org_unit: str,
    period: str,
    output_statistic: str,
    prediction: Any,
) -> LocalExplanationResponse | None:
    native_shap = (prediction.meta_data or {}).get(NATIVE_SHAP)
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
        method=NATIVE_SHAP,
        output_statistic=output_statistic,
        feature_attributions=feature_attributions,
        baseline_prediction=expected_value,
        actual_prediction=actual_prediction,
        surrogate_quality=None,
        covariate_provenance=None,
    )


def native_shap_beeswarm(
    prediction_id: int,
    output_statistic: str,
    prediction: Any,
    dataset: Any,
) -> ShapBeeswarmResponse | None:
    native_shap = (prediction.meta_data or {}).get(NATIVE_SHAP)
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


def list_filtered_native_shap_locals(
    prediction_id: int,
    prediction: Any,
    org_unit: str | None,
    period: str | None,
    output_statistic: str = "median",
) -> list[LocalExplanationResponse]:
    native_shap = (prediction.meta_data or {}).get(NATIVE_SHAP)
    if not native_shap:
        return []

    if org_unit and period:
        one = native_shap_local_response(prediction_id, org_unit, period, output_statistic, prediction)
        return [one] if one is not None else []

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
                method=NATIVE_SHAP,
                output_statistic=output_statistic,
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
    return items
