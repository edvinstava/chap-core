from types import SimpleNamespace

from chap_core.xai.native_shap_horizon import build_native_shap_horizon_summary


def test_build_native_shap_horizon_summary_none_without_native_meta():
    assert (
        build_native_shap_horizon_summary(
            1,
            "A",
            "median",
            meta_data={},
            forecasts=[SimpleNamespace(org_unit="A", period="202401", values=[1.0])],
        )
        is None
    )


def test_build_native_shap_horizon_summary_none_when_period_missing_in_native():
    forecasts = [SimpleNamespace(org_unit="A", period="202401", values=[1.0])]
    meta = {
        "native_shap": {
            "feature_names": ["f1"],
            "values": [{"location": "A", "time_period": "209999", "shap_values": [0.1]}],
        }
    }
    assert build_native_shap_horizon_summary(1, "A", "median", meta_data=meta, forecasts=forecasts) is None
