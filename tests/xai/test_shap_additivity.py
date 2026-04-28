"""Tests for SHAP additivity guarantees under target transformations."""

from __future__ import annotations

import numpy as np
import pytest

from chap_core.xai.surrogate.shap_explainer import SurrogateSHAPExplainer

shap = pytest.importorskip("shap")


def _make_fitted_explainer(transform: str | None) -> tuple[SurrogateSHAPExplainer, np.ndarray]:
    rng = np.random.default_rng(0)
    n, d = 60, 4
    X = rng.normal(size=(n, d)).astype(float)
    # Strictly positive target so log1p is valid.
    y = np.abs(X[:, 0] * 2.0 + X[:, 1] * 0.5 + 1.0) + rng.normal(scale=0.1, size=n)
    feature_names = [f"f{i}" for i in range(d)]
    explainer = SurrogateSHAPExplainer(
        feature_names=feature_names,
        model_config={"model_type": "hist_gradient_boosting"},
    )
    explainer.fit(X, y)
    if transform is not None:
        explainer._target_transform_method = transform
    return explainer, X


def test_additivity_preserved_for_log1p_when_shap_sum_is_zero():
    """When the surrogate happens to produce shap_sum == 0 for a row, additivity must still hold."""
    explainer, X = _make_fitted_explainer("log1p")

    # Force shap_sum_t to zero on one synthetic row by patching the inner explainer.
    real_shap_values = explainer._shap_explainer.shap_values

    def fake_shap_values(arr):
        sv = real_shap_values(arr)
        sv[0] = 0.0  # force exact zero contributions for first row
        return sv

    explainer._shap_explainer.shap_values = fake_shap_values
    sv_full = explainer.shap_values_matrix(X[:1])

    baseline = explainer.expected_value
    actual = float(explainer.predict(X[:1])[0])
    reconstructed = baseline + float(np.sum(sv_full[0]))

    assert reconstructed == pytest.approx(actual, rel=1e-6, abs=1e-6)


def test_yeo_johnson_without_transformer_raises_rather_than_returns_mixed_units():
    explainer, X = _make_fitted_explainer("yeo_johnson")
    # Strip the transformer to simulate a stale / unwrapped model.
    if hasattr(explainer._model, "transformer_"):
        delattr(explainer._model, "transformer_")
    if hasattr(explainer._model, "transformer"):
        delattr(explainer._model, "transformer")

    with pytest.raises(RuntimeError, match="transformer"):
        explainer.shap_values_matrix(X[:5])
