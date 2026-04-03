"""
Perturbation-based explainer for CHAP models.

This module provides model-agnostic feature attribution by perturbing input features
and measuring the impact on forecast outputs.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .types import (
    ExplanationMethod,
    FeatureAttribution,
    GlobalExplanation,
    LocalExplanation,
)

logger = logging.getLogger(__name__)


class PerturbationExplainer:
    def __init__(
        self,
        predict_fn: Callable[[Dict[str, np.ndarray]], np.ndarray],
        feature_names: List[str],
        n_repeats: int = 5,
        random_state: Optional[int] = 42,
    ):
        self.predict_fn = predict_fn
        self.feature_names = feature_names
        self.n_repeats = n_repeats
        self.rng = np.random.RandomState(random_state)

    def explain_global(
        self,
        X: Dict[str, np.ndarray],
        top_k: int = 10,
    ) -> GlobalExplanation:
        baseline_pred = self.predict_fn(X)
        baseline_mean = np.mean(baseline_pred)

        importances = []
        for feature in self.feature_names:
            if feature not in X:
                continue

            feature_importance = self._permutation_importance(X, feature, baseline_mean)
            importances.append(
                FeatureAttribution(
                    feature_name=feature,
                    importance=float(feature_importance),
                    direction="positive" if feature_importance > 0 else "negative",
                )
            )

        importances.sort(key=lambda x: abs(x.importance), reverse=True)
        top_features = importances[:top_k]

        if len(importances) > 1:
            imp_values = [abs(f.importance) for f in importances]
            stability = 1.0 - (np.std(imp_values) / (np.mean(imp_values) + 1e-10))
            stability = max(0.0, min(1.0, stability))
        else:
            stability = 1.0

        return GlobalExplanation(
            method=ExplanationMethod.PERMUTATION_IMPORTANCE,
            top_features=top_features,
            n_samples=len(next(iter(X.values()))) if X else 0,
            stability_score=stability,
        )

    def explain_local(
        self,
        X: Dict[str, np.ndarray],
        prediction_id: int,
        org_unit: str,
        period: str,
        target_idx: int = 0,
        top_k: int = 10,
    ) -> LocalExplanation:
        baseline_pred = self.predict_fn(X)
        actual_value = float(baseline_pred[target_idx])

        attributions = []
        for feature in self.feature_names:
            if feature not in X:
                continue

            X_occluded = self._occlude_feature(X, feature, target_idx)
            occluded_pred = self.predict_fn(X_occluded)
            delta = actual_value - float(occluded_pred[target_idx])

            attributions.append(
                FeatureAttribution(
                    feature_name=feature,
                    importance=float(delta),
                    direction="increases" if delta > 0 else "decreases",
                    actual_value=float(X[feature][target_idx]) if target_idx < len(X[feature]) else None,
                )
            )

        attributions.sort(key=lambda x: abs(x.importance), reverse=True)

        X_baseline = self._create_baseline(X)
        baseline_value = float(self.predict_fn(X_baseline)[target_idx])

        return LocalExplanation(
            prediction_id=prediction_id,
            org_unit=org_unit,
            period=period,
            method=ExplanationMethod.OCCLUSION,
            feature_attributions=attributions[:top_k],
            baseline_prediction=baseline_value,
            actual_prediction=actual_value,
        )

    def _permutation_importance(
        self,
        X: Dict[str, np.ndarray],
        feature: str,
        baseline_mean: float,
    ) -> float:
        deltas = []
        original = X[feature].copy()

        for _ in range(self.n_repeats):
            X_permuted = X.copy()
            X_permuted[feature] = self.rng.permutation(original)
            permuted_pred = self.predict_fn(X_permuted)
            delta = baseline_mean - np.mean(permuted_pred)
            deltas.append(delta)

        return float(np.mean(deltas))

    def _occlude_feature(
        self,
        X: Dict[str, np.ndarray],
        feature: str,
        target_idx: int,
    ) -> Dict[str, np.ndarray]:
        X_occluded = {k: v.copy() for k, v in X.items()}
        baseline_value = np.mean(X[feature])
        X_occluded[feature][target_idx] = baseline_value
        return X_occluded

    def _create_baseline(
        self,
        X: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        return {k: np.full_like(v, np.mean(v)) for k, v in X.items()}


def create_predict_fn_from_samples(
    samples: np.ndarray,
    output_statistic: str = "median",
) -> Callable[[Dict[str, np.ndarray]], np.ndarray]:
    def predict_fn(X: Dict[str, np.ndarray]) -> np.ndarray:
        if output_statistic == "median":
            return np.median(samples, axis=-1)
        elif output_statistic == "mean":
            return np.mean(samples, axis=-1)
        else:
            return np.median(samples, axis=-1)

    return predict_fn
