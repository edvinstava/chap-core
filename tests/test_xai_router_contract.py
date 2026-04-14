import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from chap_core.rest_api.v1.routers.xai import (
    LocalExplanationRequest,
    compute_global_explanation,
    compute_horizon_summary,
    compute_local_explanation,
    compute_shap_beeswarm,
    get_global_explanation,
    run_explanations,
)
from chap_core.rest_api.v1.routers.xai import RunExplanationsRequest


class _SessionStub:
    def __init__(self, prediction):
        self._prediction = prediction

    def get(self, *_args, **_kwargs):
        return self._prediction


def test_compute_endpoints_reject_archived_method():
    with pytest.raises(HTTPException, match="archived"):
        asyncio.run(
            compute_global_explanation(
                prediction_id=1,
                xai_method="occlusion",
                output_statistic="median",
                top_k=10,
                force=False,
                session=_SessionStub(prediction=None),
            )
        )

    with pytest.raises(HTTPException, match="archived"):
        asyncio.run(
            compute_local_explanation(
                prediction_id=1,
                request=LocalExplanationRequest(orgUnit="A", period="202401", xaiMethod="occlusion"),
                session=_SessionStub(prediction=None),
            )
        )

    with pytest.raises(HTTPException, match="archived"):
        asyncio.run(
            compute_shap_beeswarm(
                prediction_id=1,
                output_statistic="median",
                xai_method="occlusion",
                session=_SessionStub(prediction=None),
            )
        )


def test_compute_endpoints_reject_unknown_method():
    with pytest.raises(HTTPException, match="not found"):
        asyncio.run(
            compute_global_explanation(
                prediction_id=1,
                xai_method="nope",
                output_statistic="median",
                top_k=10,
                force=False,
                session=_SessionStub(prediction=None),
            )
        )

    with pytest.raises(HTTPException, match="not found"):
        asyncio.run(
            run_explanations(
                prediction_id=1,
                request=RunExplanationsRequest(xaiMethodName="nope"),
                database_url="sqlite:///tmp.db",
            )
        )


class _EmptyQueryResult:
    def all(self):
        return []


class _HorizonSessionStub:
    def __init__(self, prediction):
        self._prediction = prediction

    def get(self, *_args, **_kwargs):
        return self._prediction

    def exec(self, _query):
        return _EmptyQueryResult()


def test_horizon_summary_native_shap_uses_meta_not_dataset():
    forecasts = [
        SimpleNamespace(org_unit="W6", period="202401", values=[1.0, 2.0]),
        SimpleNamespace(org_unit="W6", period="202402", values=[3.0]),
    ]
    prediction = SimpleNamespace(
        forecasts=forecasts,
        meta_data={
            "native_shap": {
                "feature_names": ["f1", "f2"],
                "expected_value": 0.5,
                "values": [
                    {"location": "W6", "time_period": "202401", "shap_values": [0.1, -0.1]},
                    {"location": "W6", "time_period": "202402", "shap_values": [0.2, -0.15]},
                ],
            }
        },
    )
    response = asyncio.run(
        compute_horizon_summary(
            prediction_id=7,
            org_unit="W6",
            output_statistic="median",
            xai_method="native_shap",
            session=_HorizonSessionStub(prediction=prediction),
        )
    )
    assert response.method == "native_shap"
    assert len(response.steps) == 2
    assert response.steps[0].forecast_step == 1
    assert response.steps[0].actual_prediction == pytest.approx(0.5 + 0.1 - 0.1)
    assert response.surrogate_quality is None


def test_global_endpoint_does_not_fallback_to_legacy_meta():
    prediction = SimpleNamespace(
        meta_data={
            "xai": {
                "global": {
                    "method": "shap",
                    "topFeatures": [{"feature_name": "rainfall", "importance": 0.9}],
                    "computedAt": "2024-01-01T00:00:00+00:00",
                    "nSamples": 10,
                    "stabilityScore": 0.5,
                }
            }
        }
    )
    response = asyncio.run(
        get_global_explanation(
            prediction_id=1,
            xai_method="shap_auto",
            session=_SessionStub(prediction=prediction),
        )
    )
    assert response.available is False
    assert response.top_features == []
