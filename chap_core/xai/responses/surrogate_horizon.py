from typing import Any

import numpy as np

from chap_core.rest_api.v1.xai_schemas import (
    AverageImportance,
    HorizonFeatureImportance,
    HorizonStepSummary,
    HorizonSummaryResponse,
)
from chap_core.xai.forecast_utils import forecast_actual_value
from chap_core.xai.responses.quality import quality_response_dict


def horizon_summary_from_surrogate(
    prediction_id: int,
    org_unit: str,
    xai_method: str,
    output_statistic: str,
    forecasts: list[Any],
    X: np.ndarray,
    feature_names: list[str],
    explainer: Any,
) -> HorizonSummaryResponse:
    unit_entries = [(i, fc) for i, fc in enumerate(forecasts) if fc.org_unit == org_unit]
    unit_entries.sort(key=lambda x: x[1].period)

    steps: list[HorizonStepSummary] = []
    all_importances: dict[str, list[float]] = {f: [] for f in feature_names}

    for step_num, (idx, fc) in enumerate(unit_entries, start=1):
        actual = forecast_actual_value(fc.values, output_statistic)
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
        method=xai_method,
        output_statistic=output_statistic,
        steps=steps,
        average_importance=avg_importance,
        surrogate_quality=quality_response_dict(explainer.quality_dict()),
    )
