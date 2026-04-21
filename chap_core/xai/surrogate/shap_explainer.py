"""Surrogate-based SHAP explainer for CHAP external models.

Trains an sklearn surrogate on stored (feature, forecast_outcome) pairs and applies SHAP
for feature attributions. A surrogate is needed because CHAP models run in Docker
containers and cannot be called at explanation time.
"""

import io
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from ..types import (
    ExplanationMethod,
    FeatureAttribution,
    GlobalExplanation,
    LocalExplanation,
)
from .model import (
    DEFAULT_MODEL_TYPE,
    build_shap_explainer,
    build_surrogate_model,
    resolve_model_params,
)
from .preprocessing import FilterResult, filter_features
from .training import train_surrogate

if TYPE_CHECKING:
    from .quality import SurrogateQuality

logger = logging.getLogger(__name__)

MIN_SAMPLES_GLOBAL = 10
MIN_SAMPLES_LOCAL = 3


def _importance_to_normalized_ranks(mean_abs: np.ndarray) -> np.ndarray:
    order = np.argsort(-mean_abs)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(mean_abs))
    return ranks / max(len(mean_abs) - 1, 1)


class SurrogateSHAPExplainer:
    """
    Fit a configurable sklearn surrogate on stored (features, target) pairs,
    then use a SHAP explainer to produce feature attributions.

    The surrogate model type is controlled by model_config:
      {"model_type": "random_forest", "n_estimators": 200}

    Supported model types are defined in surrogate.registry.SUPPORTED_MODELS.

    SHAP guarantee (local, tree models): predicted_value = baseline + sum(shap_values)
    """

    def __init__(
        self,
        feature_names: list[str],
        model_config: dict | None = None,
        random_state: int = 42,
        hyperparams: dict | None = None,
        imputation_rates: dict[str, float] | None = None,
    ):
        self.feature_names = list(feature_names)
        self.model_config = model_config or {}
        self.hyperparams = hyperparams or {}
        self.imputation_rates = imputation_rates or {}
        self._random_state = random_state

        # Resolve "auto" to the default for initial construction; fit() rebuilds with the actual type.
        model_type = self.model_config.get("model_type", DEFAULT_MODEL_TYPE)
        if model_type == "auto":
            model_type = DEFAULT_MODEL_TYPE
        params = resolve_model_params(model_type, self.model_config, hyperparams)
        self._model = build_surrogate_model(model_type, params, random_state=random_state)
        self._shap_explainer = None
        self._is_fitted = False
        self.quality: SurrogateQuality | None = None
        self._shap_batch_cache: np.ndarray | None = None
        self._shap_batch_cached_X: np.ndarray | None = None
        self._keep_indices: list[int] | None = None
        self._kept_feature_names: list[str] | None = None
        self._target_transform_method: str | None = None
        self._X_train: np.ndarray | None = None
        self._baseline_prediction: float = 0.0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        min_samples: int = MIN_SAMPLES_GLOBAL,
        groups: np.ndarray | None = None,
        filter_result: FilterResult | None = None,
    ) -> None:
        """Fit the surrogate model on (X, y).

        When *filter_result* is provided (from a prior ``filter_features()``
        call), feature filtering is skipped — model selection and tuning
        already operated on the same filtered matrix.
        """
        if len(X) < min_samples:
            raise ValueError(f"Not enough data to train surrogate: got {len(X)} rows, need at least {min_samples}.")

        if filter_result is not None:
            X_filtered = filter_result.X_filtered
            self._kept_feature_names = list(filter_result.kept_feature_names)
            self._keep_indices = list(filter_result.kept_indices)
            constant_feats = filter_result.constant_features
            removed_feats = filter_result.removed_features
            perm_removed_feats = filter_result.perm_removed_features
        else:
            fr = filter_features(
                X,
                y,
                self.feature_names,
                self.imputation_rates,
                model_type=self.model_config.get("model_type", DEFAULT_MODEL_TYPE),
                random_state=self._random_state,
            )
            X_filtered = fr.X_filtered
            self._kept_feature_names = list(fr.kept_feature_names)
            self._keep_indices = list(fr.kept_indices)
            constant_feats = fr.constant_features
            removed_feats = fr.removed_features
            perm_removed_feats = fr.perm_removed_features

        model_type = self.model_config.get("model_type", DEFAULT_MODEL_TYPE)
        params = resolve_model_params(model_type, self.model_config, self.hyperparams)

        result = train_surrogate(
            X_filtered=X_filtered,
            y=y,
            model_type=model_type,
            params=params,
            groups=groups,
            random_state=self._random_state,
            imputation_rates=self.imputation_rates,
            constant_features=constant_feats,
            removed_features=removed_feats,
            perm_removed_features=perm_removed_feats,
            n_samples=len(X),
        )

        self._model = result.model
        self.quality = result.quality
        self._target_transform_method = result.target_transform_method
        self._baseline_prediction = result.baseline_prediction
        self._X_train = result.X_background

        shap_base = getattr(result.model, "regressor_", result.model)
        self._shap_explainer = self._try_build_shap_explainer(shap_base, model_type, result.X_background)

        self._is_fitted = True
        self._shap_batch_cache = None
        self._shap_batch_cached_X = None

    @staticmethod
    def _try_build_shap_explainer(model: Any, model_type: str, X_train: np.ndarray | None = None) -> Any:
        """Build a SHAP explainer, returning None if the shap package is not installed."""
        try:
            return build_shap_explainer(model, model_type, X_train=X_train)
        except ModuleNotFoundError:
            logger.debug("shap package not installed; SHAP explanations will be unavailable.")
            return None

    def _filter_X(self, X: np.ndarray) -> np.ndarray:
        return X[:, self._keep_indices] if self._keep_indices is not None else X

    def _expand_shap_values(self, sv_filtered: np.ndarray) -> np.ndarray:
        if self._keep_indices is None:
            return sv_filtered
        full = np.zeros((sv_filtered.shape[0], len(self.feature_names)))
        full[:, self._keep_indices] = sv_filtered
        return full

    def shap_values_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute SHAP values for the given feature matrix. Shape: (n_rows, n_features)."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before shap_values_matrix().")
        if self._shap_explainer is None:
            raise RuntimeError(
                "SHAP explainer is not available. Install the 'shap' package to use SHAP-based explanations."
            )
        sv = self._shap_explainer.shap_values(self._filter_X(X))
        sv_full = self._expand_shap_values(sv)

        if self._target_transform_method in ("log1p", "yeo_johnson"):
            ev = self._shap_explainer.expected_value
            baseline_t = float(ev[0]) if hasattr(ev, "__len__") else float(ev)

            shap_sum_t = np.sum(sv_full, axis=1)
            t_pred = baseline_t + shap_sum_t

            if self._target_transform_method == "log1p":
                inv = np.expm1
                y_base = float(inv(baseline_t))
                y_pred = inv(t_pred)
            else:
                transformer = getattr(self._model, "transformer_", None)
                if transformer is None:
                    transformer = getattr(self._model, "transformer", None)
                if transformer is None:
                    return sv_full
                y_base = float(transformer.inverse_transform(np.array([[baseline_t]], dtype=float)).reshape(-1)[0])
                y_pred = transformer.inverse_transform(t_pred.reshape(-1, 1)).reshape(-1)

            scale = np.zeros_like(shap_sum_t, dtype=float)
            nonzero = shap_sum_t != 0
            scale[nonzero] = (y_pred[nonzero] - y_base) / shap_sum_t[nonzero]
            sv_full = sv_full * scale[:, None]

        return sv_full

    @property
    def expected_value(self) -> float:
        """Baseline prediction (E[f(X)]). Falls back to the training mean when SHAP is unavailable."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before accessing expected_value.")
        if self._shap_explainer is None:
            return self._baseline_prediction
        ev = self._shap_explainer.expected_value
        baseline_t = float(ev[0]) if hasattr(ev, "__len__") else float(ev)

        if self._target_transform_method == "log1p":
            return float(np.expm1(baseline_t))

        if self._target_transform_method == "yeo_johnson":
            transformer = getattr(self._model, "transformer_", None)
            if transformer is None:
                transformer = getattr(self._model, "transformer", None)
            if transformer is None:
                return baseline_t
            inv = transformer.inverse_transform(np.array([[baseline_t]], dtype=float))
            return float(inv.reshape(-1)[0])

        return baseline_t

    def to_bytes(self) -> bytes:
        """Serialize the fitted surrogate model to bytes."""
        import joblib  # type: ignore[import-untyped]

        if not self._is_fitted:
            raise RuntimeError("Call fit() before to_bytes().")
        buf = io.BytesIO()
        joblib.dump(
            {
                "model": self._model,
                "feature_names": self.feature_names,
                "model_config": self.model_config,
                "hyperparams": self.hyperparams,
                "quality": self.quality,
                "imputation_rates": self.imputation_rates,
                "keep_indices": self._keep_indices,
                "kept_feature_names": self._kept_feature_names,
                "target_transformed": self._target_transform_method is not None,
                "target_transform_method": self._target_transform_method,
                "X_train": self._X_train,
            },
            buf,
        )
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> "SurrogateSHAPExplainer":
        """Deserialize a stored surrogate model."""
        import joblib

        buf = io.BytesIO(data)
        state = joblib.load(buf)

        # "gbm" key is the old serialization format — kept for backward compat
        model = state.get("model") or state.get("gbm")
        model_config = state.get("model_config", {})

        instance = cls(
            feature_names=state["feature_names"],
            model_config=model_config,
            hyperparams=state.get("hyperparams"),
            imputation_rates=state.get("imputation_rates", {}),
        )
        instance._model = model
        instance._keep_indices = state.get("keep_indices")
        instance._kept_feature_names = state.get("kept_feature_names")
        instance._X_train = state.get("X_train")
        # Backward compat: old models have target_transformed but no target_transform_method
        target_transform_method = state.get("target_transform_method")
        if target_transform_method is None and state.get("target_transformed", False):
            target_transform_method = "log1p"
        instance._target_transform_method = target_transform_method

        quality = state.get("quality")
        model_type = model_config.get("model_type", DEFAULT_MODEL_TYPE)
        if model_type == "auto":
            model_type = quality.selected_model_type if quality and quality.selected_model_type else DEFAULT_MODEL_TYPE
        is_transformed = target_transform_method is not None
        shap_model = model.regressor_ if is_transformed and hasattr(model, "regressor_") else model
        instance._shap_explainer = cls._try_build_shap_explainer(shap_model, model_type, instance._X_train)
        instance._is_fitted = True
        instance.quality = quality
        return instance

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")
        return np.asarray(self._model.predict(self._filter_X(X)), dtype=float)

    def quality_dict(self) -> dict:
        return self.quality.to_dict() if self.quality is not None else {}

    def explain_global(
        self,
        X: np.ndarray,
        top_k: int = 10,
        n_bootstrap: int = 20,
        random_state: int = 42,
    ) -> GlobalExplanation:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before explain_global().")

        if self._shap_explainer is not None:
            shap_values = self.shap_values_matrix(X)
            mean_abs = np.mean(np.abs(shap_values), axis=0)
            mean_signed = np.mean(shap_values, axis=0)
        else:
            logger.debug("shap unavailable; using permutation importance for global explanation.")
            mean_abs, mean_signed = self._global_importances_fallback(X)

        mean_feature_values = np.nanmean(X, axis=0)

        ranking_std = self._bootstrap_ranking_std(
            X,
            n_bootstrap,
            np.random.RandomState(random_state),
            shap_values if self._shap_explainer is not None else None,
        )
        stability = float(max(0.0, min(1.0, 1.0 - np.mean(ranking_std))))

        attributions = []
        for i, name in enumerate(self.feature_names):
            attributions.append(
                FeatureAttribution(
                    feature_name=name,
                    importance=float(mean_abs[i]),
                    direction="positive" if mean_signed[i] >= 0 else "negative",
                    baseline_value=None,
                    actual_value=float(mean_feature_values[i]),
                )
            )
        attributions.sort(key=lambda a: abs(a.importance), reverse=True)

        return GlobalExplanation(
            method=ExplanationMethod.SHAP,
            top_features=attributions[:top_k],
            n_samples=len(X),
            stability_score=stability,
        )

    def explain_local(
        self,
        X: np.ndarray,
        instance_idx: int,
        prediction_id: int,
        org_unit: str,
        period: str,
        feature_actual_values: dict[str, float],
        top_k: int = 10,
        output_statistic: str = "median",
        actual_forecast_value: float | None = None,
    ) -> LocalExplanation:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before explain_local().")

        instance = X[instance_idx : instance_idx + 1]
        if self._shap_explainer is not None:
            need_shap_recompute = (
                self._shap_batch_cache is None
                or self._shap_batch_cached_X is None
                or self._shap_batch_cached_X.shape != X.shape
                or not np.array_equal(self._shap_batch_cached_X, X)
            )
            if need_shap_recompute:
                self._shap_batch_cached_X = np.array(X, copy=True)
                self._shap_batch_cache = self.shap_values_matrix(X)
            shap_values = self._shap_batch_cache[instance_idx]
        else:
            logger.debug("shap unavailable; using occlusion for local explanation.")
            shap_values = self._local_importances_fallback(X, instance_idx)
        baseline = self.expected_value

        actual = actual_forecast_value if actual_forecast_value is not None else float(self.predict(instance)[0])

        attributions = []
        for i, name in enumerate(self.feature_names):
            sv = float(shap_values[i])
            attributions.append(
                FeatureAttribution(
                    feature_name=name,
                    importance=sv,
                    direction="positive" if sv > 0 else "negative",
                    actual_value=feature_actual_values.get(name),
                    baseline_value=baseline,
                )
            )
        attributions.sort(key=lambda a: abs(a.importance), reverse=True)

        return LocalExplanation(
            prediction_id=prediction_id,
            org_unit=org_unit,
            period=period,
            method=ExplanationMethod.SHAP,
            output_statistic=output_statistic,
            feature_attributions=attributions[:top_k],
            baseline_prediction=baseline,
            actual_prediction=actual,
        )

    def explain_local_with_interactions(
        self,
        X: np.ndarray,
        instance_idx: int,
        prediction_id: int,
        org_unit: str,
        period: str,
        feature_actual_values: dict[str, float],
        top_k: int = 10,
        output_statistic: str = "median",
        actual_forecast_value: float | None = None,
        interaction_top_k: int = 3,
    ) -> tuple[LocalExplanation, list[dict]]:
        """Like explain_local but also returns SHAP interaction values for top features."""
        local_exp = self.explain_local(
            X=X,
            instance_idx=instance_idx,
            prediction_id=prediction_id,
            org_unit=org_unit,
            period=period,
            feature_actual_values=feature_actual_values,
            top_k=top_k,
            output_statistic=output_statistic,
            actual_forecast_value=actual_forecast_value,
        )

        interactions = self._compute_interactions(X, instance_idx, interaction_top_k)
        return local_exp, interactions

    def _compute_interactions(
        self,
        X: np.ndarray,
        instance_idx: int,
        top_k: int = 3,
    ) -> list[dict]:
        """Compute SHAP interaction values for a single instance."""
        if self._shap_explainer is None:
            return []
        try:
            X_f = self._filter_X(X)
            iv_filtered = self._shap_explainer.shap_interaction_values(X_f[instance_idx : instance_idx + 1])[0]
        except Exception as e:
            logger.warning("SHAP interaction values not available: %s", e)
            return []

        n_feats = len(self.feature_names)
        interaction_values = np.zeros((n_feats, n_feats))
        keep = self._keep_indices if self._keep_indices is not None else list(range(n_feats))
        interaction_values[np.ix_(keep, keep)] = iv_filtered

        pairs: list[tuple[float, int, int]] = []
        for i in range(n_feats):
            pairs.extend((abs(float(interaction_values[i, j])), i, j) for j in range(i + 1, n_feats))
        pairs.sort(reverse=True)

        result = []
        for _abs_val, i, j in pairs[:top_k]:
            val = float(interaction_values[i, j])
            result.append(
                {
                    "feature_1": self.feature_names[i],
                    "feature_2": self.feature_names[j],
                    "interaction_value": val,
                    "direction": "positive" if val > 0 else "negative",
                }
            )
        return result

    def _global_importances_fallback(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Permutation-based global feature importances, used when shap is unavailable.

        Returns (mean_abs, mean_signed) arrays aligned to self.feature_names.
        Direction is derived by checking whether above/below-mean feature values
        correlate with above-mean predictions.
        """
        from sklearn.inspection import permutation_importance

        X_f = self._filter_X(X)
        y_surrogate = self._model.predict(X_f)
        result = permutation_importance(
            self._model, X_f, y_surrogate, n_repeats=10, random_state=self._random_state
        )
        mean_abs_filtered = np.maximum(result.importances_mean, 0.0)
        mean_abs_full = np.zeros(len(self.feature_names))
        if self._keep_indices is not None:
            mean_abs_full[self._keep_indices] = mean_abs_filtered

        # Estimate direction: positive if high feature value correlates with high prediction
        mean_pred = float(np.mean(y_surrogate))
        mean_feat = np.mean(X, axis=0)
        mean_signed = np.where(
            np.mean((X - mean_feat) * (y_surrogate[:, None] - mean_pred), axis=0) >= 0,
            mean_abs_full,
            -mean_abs_full,
        )
        return mean_abs_full, mean_signed

    def _local_importances_fallback(self, X: np.ndarray, instance_idx: int) -> np.ndarray:
        """Occlusion-based local attributions, used when shap is unavailable.

        Each feature is replaced by its column mean; the attribution is the
        resulting change in prediction (pred_original - pred_occluded).
        """
        X_f = self._filter_X(X)
        instance = X_f[instance_idx : instance_idx + 1]
        baseline_pred = float(self._model.predict(instance)[0])
        col_means = np.mean(X_f, axis=0)

        n_feats = X_f.shape[1]
        occluded_batch = np.repeat(instance, n_feats, axis=0)
        for i in range(n_feats):
            occluded_batch[i, i] = col_means[i]
        attributions_filtered = baseline_pred - self._model.predict(occluded_batch)

        full = np.zeros(len(self.feature_names))
        if self._keep_indices is not None:
            full[self._keep_indices] = attributions_filtered
        else:
            full = attributions_filtered
        return full

    def _bootstrap_ranking_std(
        self,
        X: np.ndarray,
        n_bootstrap: int,
        rng: np.random.RandomState,
        sv_full: np.ndarray | None = None,
    ) -> np.ndarray:
        n_rows, n_feats = X.shape
        rank_matrix = np.zeros((n_bootstrap, n_feats))
        if self._shap_explainer is not None:
            if sv_full is None:
                sv_full = self.shap_values_matrix(X)
            for b in range(n_bootstrap):
                idx = rng.choice(n_rows, size=n_rows, replace=True)
                rank_matrix[b] = _importance_to_normalized_ranks(np.mean(np.abs(sv_full[idx]), axis=0))
        else:
            for b in range(n_bootstrap):
                idx = rng.choice(n_rows, size=n_rows, replace=True)
                mean_abs_boot, _ = self._global_importances_fallback(X[idx])
                rank_matrix[b] = _importance_to_normalized_ranks(mean_abs_boot)
        return np.asarray(np.std(rank_matrix, axis=0), dtype=float)
