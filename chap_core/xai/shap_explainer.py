"""
Surrogate-based SHAP explainer for CHAP external models.

Trains a configurable sklearn surrogate from stored (feature, forecast_outcome)
pairs, then applies a SHAP explainer (TreeExplainer for tree models) for
exact feature attributions.

Why a surrogate? CHAP models run in Docker containers and cannot be called at
explanation time. The surrogate learns the same input-output mapping from stored
predictions and enables proper SHAP decomposition.

The underlying model type is controlled by model_config, e.g.:
  {"model_type": "random_forest", "n_estimators": 200}

Supported model types and their tunable parameters are defined in surrogate_model.py.
"""

import io
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .surrogate_model import (
    DEFAULT_MODEL_TYPE,
    _loo_r2,
    build_shap_explainer,
    build_surrogate_model,
    get_display_name,
    get_model_info,
    make_loo_model_factory,
    resolve_model_params,
    tune_surrogate_hyperparameters,
)
from .types import (
    ExplanationMethod,
    FeatureAttribution,
    GlobalExplanation,
    LocalExplanation,
)

logger = logging.getLogger(__name__)

MIN_SAMPLES_GLOBAL = 10
MIN_SAMPLES_LOCAL = 3
DEFAULT_TUNING_TRIALS = 30
MIN_SAMPLES_FOR_TUNING = 15
MIN_SAMPLES_FOR_PERMUTATION = 100
MIN_FEATURES_FOR_PERMUTATION = 6
MIN_SAMPLES_FOR_TARGET_TRANSFORM = 30


@dataclass
class SurrogateQuality:
    r_squared: Optional[float] = None
    mae: float = 0.0
    mape: Optional[float] = None
    n_samples: int = 0
    n_unique_rows: int = 0
    constant_features: list[str] = field(default_factory=list)
    imputation_rates: dict[str, float] = field(default_factory=dict)
    removed_features: list[str] = field(default_factory=list)
    selected_model_type: Optional[str] = None
    selected_model_display_name: Optional[str] = None
    n_groups: Optional[int] = None
    fidelity_tier: str = "good"
    fidelity_warning: Optional[str] = None
    residual_mean: Optional[float] = None
    residual_std: Optional[float] = None
    target_transformed: bool = False
    target_transform_method: Optional[str] = None
    permutation_removed_features: list[str] = field(default_factory=list)
    r_squared_train: Optional[float] = None


def _compute_fidelity_tier(r2: Optional[float]) -> tuple[str, Optional[str]]:
    """Return (fidelity_tier, fidelity_warning) based on LOO-R².

    Thresholds follow XAI best practice for surrogate fidelity:
      R² < 0.5  → poor   (surrogate explains less than half the variance)
      R² < 0.8  → moderate (reasonable but notable unexplained variance remains)
      R² >= 0.8 → good
    """
    if r2 is None or r2 < 0.5:
        return (
            "poor",
            "Surrogate R\u00b2 is low (< 0.5); the surrogate does not closely mimic the original model "
            "and SHAP attributions may not reflect true model behaviour.",
        )
    if r2 < 0.8:
        return (
            "moderate",
            "Surrogate R\u00b2 is moderate (0.5\u20130.8); attributions are indicative but should be "
            "interpreted with caution.",
        )
    return "good", None


def tune_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = DEFAULT_MODEL_TYPE,
    n_trials: int = DEFAULT_TUNING_TRIALS,
    groups: Optional[np.ndarray] = None,
    random_state: int = 42,
) -> dict:
    """Use Optuna to find optimal hyperparameters for the given surrogate model type."""
    return tune_surrogate_hyperparameters(
        X, y, model_type=model_type, n_trials=n_trials, groups=groups, random_state=random_state
    )



def _compute_target(
    forecast_values_list: list[list[float]],
    output_statistic: str,
) -> np.ndarray:
    if output_statistic == "mean":
        return np.array([np.mean(v) for v in forecast_values_list], dtype=float)
    elif output_statistic.startswith("q"):
        try:
            q = float(output_statistic[1:]) / 100.0
        except ValueError:
            q = 0.5
        return np.array([np.quantile(v, q) for v in forecast_values_list], dtype=float)
    else:
        return np.array([np.median(v) for v in forecast_values_list], dtype=float)


@dataclass
class FilterResult:
    """Result of feature filtering: constant removal + optional permutation selection."""

    X_filtered: np.ndarray
    kept_feature_names: list[str]
    kept_indices: list[int]
    removed_features: list[str]
    constant_features: list[str]
    perm_removed_features: list[str]


def filter_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    imputation_rates: dict[str, float],
    model_type: str = DEFAULT_MODEL_TYPE,
    random_state: int = 42,
    imputation_threshold: float = 0.9,
) -> FilterResult:
    """Filter constant, heavily-imputed, and (optionally) noise features.

    Permutation-based selection only runs when the dataset is large enough
    (>= MIN_SAMPLES_FOR_PERMUTATION rows and > MIN_FEATURES_FOR_PERMUTATION features)
    to produce reliable importance estimates.
    """
    constant_feats: list[str] = []
    removed_feats: list[str] = []
    keep_mask: list[bool] = []

    for i, name in enumerate(feature_names):
        is_constant = np.ptp(X[:, i]) == 0
        imp_rate = imputation_rates.get(name, 0.0)
        if is_constant:
            constant_feats.append(name)
        if is_constant or imp_rate >= imputation_threshold:
            removed_feats.append(name)
            keep_mask.append(False)
            if imp_rate >= imputation_threshold:
                logger.warning(
                    "Removing feature '%s' before fitting: %.0f%% imputed.",
                    name,
                    imp_rate * 100,
                )
        else:
            keep_mask.append(True)

    if constant_feats:
        logger.warning(
            "Features with zero variance (constant): %s -- removed before fitting.",
            constant_feats,
        )

    if not any(keep_mask):
        logger.warning("All features would be removed -- keeping originals as fallback.")
        keep_mask = [True] * len(feature_names)
        removed_feats = []

    keep_indices = [i for i, keep in enumerate(keep_mask) if keep]
    X_filtered = X[:, keep_indices]
    kept_names = [feature_names[i] for i in keep_indices]

    # Permutation-based feature selection: only for large datasets with many features
    perm_removed_feats: list[str] = []
    n_features_filtered = X_filtered.shape[1]
    if (
        n_features_filtered > MIN_FEATURES_FOR_PERMUTATION
        and len(X_filtered) >= MIN_SAMPLES_FOR_PERMUTATION
    ):
        try:
            from sklearn.inspection import permutation_importance
            from sklearn.model_selection import train_test_split

            X_tr, X_val, y_tr, y_val = train_test_split(
                X_filtered, y, test_size=0.25, random_state=random_state
            )
            quick_model = make_loo_model_factory(model_type, random_state=random_state, n_samples=len(X_tr))()
            quick_model.fit(X_tr, y_tr)
            perm_result = permutation_importance(
                quick_model,
                X_val,
                y_val,
                n_repeats=10,
                random_state=random_state,
                scoring="neg_mean_squared_error",
            )
            importances = perm_result.importances_mean
            min_keep = min(3, n_features_filtered)
            threshold = importances.mean() - importances.std()
            perm_keep_mask = importances >= threshold
            if perm_keep_mask.sum() < min_keep:
                top_indices = np.argsort(importances)[-min_keep:]
                perm_keep_mask = np.zeros(n_features_filtered, dtype=bool)
                perm_keep_mask[top_indices] = True

            if perm_keep_mask.sum() < n_features_filtered:
                perm_removed_indices = np.where(~perm_keep_mask)[0]
                perm_removed_feats = [kept_names[i] for i in perm_removed_indices]
                perm_kept_indices = np.where(perm_keep_mask)[0]
                X_filtered = X_filtered[:, perm_kept_indices]
                kept_names = [kept_names[i] for i in perm_kept_indices]
                keep_indices = [keep_indices[i] for i in perm_kept_indices]
                logger.info(
                    "Permutation feature selection removed %d noise feature(s): %s",
                    len(perm_removed_feats),
                    perm_removed_feats,
                )
        except Exception as e:
            logger.debug("Permutation feature selection failed: %s", e)

    return FilterResult(
        X_filtered=X_filtered,
        kept_feature_names=kept_names,
        kept_indices=keep_indices,
        removed_features=removed_feats,
        constant_features=constant_feats,
        perm_removed_features=perm_removed_feats,
    )


class SurrogateSHAPExplainer:
    """
    Fit a configurable sklearn surrogate on stored (features, target) pairs,
    then use a SHAP explainer to produce feature attributions.

    The surrogate model type is controlled by model_config:
      {"model_type": "random_forest", "n_estimators": 200}

    Supported model types are defined in surrogate_model.SUPPORTED_MODELS.

    SHAP guarantee (local, tree models): predicted_value = baseline + sum(shap_values)
    """

    # Features with an imputation rate above this threshold are removed before fitting.
    IMPUTATION_REMOVAL_THRESHOLD = 0.9

    def __init__(
        self,
        feature_names: list[str],
        model_config: Optional[dict] = None,
        random_state: int = 42,
        hyperparams: Optional[dict] = None,
        imputation_rates: Optional[dict[str, float]] = None,
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
        self.quality: Optional[SurrogateQuality] = None
        self._shap_batch_cache: Optional[np.ndarray] = None
        self._shap_batch_cached_X: Optional[np.ndarray] = None
        self._lime_explainer_cache: Optional[Any] = None
        self._lime_cached_X: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        min_samples: int = MIN_SAMPLES_GLOBAL,
        groups: Optional[np.ndarray] = None,
        filter_result: Optional[FilterResult] = None,
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

        unique_rows = len(np.unique(X_filtered, axis=0))

        model_type = self.model_config.get("model_type", DEFAULT_MODEL_TYPE)
        params = resolve_model_params(model_type, self.model_config, self.hyperparams)
        n_fit = len(X_filtered)

        # Target transform: only for datasets large enough to estimate reliably
        target_transform_method: Optional[str] = None
        if n_fit >= MIN_SAMPLES_FOR_TARGET_TRANSFORM:
            try:
                from sklearn.compose import TransformedTargetRegressor
                from sklearn.model_selection import cross_val_score as _cvs
                from sklearn.preprocessing import PowerTransformer

                cv_folds = min(n_fit, 5) if n_fit >= 10 else max(2, n_fit)

                def _make_model():
                    return build_surrogate_model(
                        model_type, params, random_state=self._random_state, n_samples=n_fit
                    )

                r2_raw = float(np.mean(_cvs(_make_model(), X_filtered, y, cv=cv_folds, scoring="r2")))
                best_r2, best_method = r2_raw, None
                r2_improvement_threshold = 0.05

                if float(np.min(y)) >= 0.0:
                    r2_log = float(
                        np.mean(
                            _cvs(
                                TransformedTargetRegressor(
                                    regressor=_make_model(), func=np.log1p, inverse_func=np.expm1
                                ),
                                X_filtered,
                                y,
                                cv=cv_folds,
                                scoring="r2",
                            )
                        )
                    )
                    if r2_log > best_r2 + r2_improvement_threshold:
                        best_r2, best_method = r2_log, "log1p"

                if n_fit >= 10:
                    r2_yj = float(
                        np.mean(
                            _cvs(
                                TransformedTargetRegressor(
                                    regressor=_make_model(),
                                    transformer=PowerTransformer(method="yeo-johnson", standardize=False),
                                ),
                                X_filtered,
                                y,
                                cv=cv_folds,
                                scoring="r2",
                            )
                        )
                    )
                    if r2_yj > best_r2 + r2_improvement_threshold:
                        best_r2, best_method = r2_yj, "yeo_johnson"

                target_transform_method = best_method
            except Exception as e:
                logger.debug("Target-transform check failed, using raw y: %s", e)

        if target_transform_method == "log1p":
            from sklearn.compose import TransformedTargetRegressor

            final_model: Any = TransformedTargetRegressor(
                regressor=build_surrogate_model(model_type, params, random_state=self._random_state, n_samples=n_fit),
                func=np.log1p,
                inverse_func=np.expm1,
            )
            final_model.fit(X_filtered, y)
            self._model = final_model
            self._shap_explainer = self._try_build_shap_explainer(
                final_model.regressor_, model_type, X_filtered
            )
        elif target_transform_method == "yeo_johnson":
            from sklearn.compose import TransformedTargetRegressor
            from sklearn.preprocessing import PowerTransformer

            final_model = TransformedTargetRegressor(
                regressor=build_surrogate_model(model_type, params, random_state=self._random_state, n_samples=n_fit),
                transformer=PowerTransformer(method="yeo-johnson", standardize=False),
            )
            final_model.fit(X_filtered, y)
            self._model = final_model
            self._shap_explainer = self._try_build_shap_explainer(
                final_model.regressor_, model_type, X_filtered
            )
        else:
            final_model = build_surrogate_model(model_type, params, random_state=self._random_state, n_samples=n_fit)
            final_model.fit(X_filtered, y)
            self._model = final_model
            self._shap_explainer = self._try_build_shap_explainer(final_model, model_type, X_filtered)

        self._is_fitted = True
        self._shap_batch_cache = None
        self._shap_batch_cached_X = None
        self._lime_explainer_cache = None
        self._lime_cached_X = None
        self._target_transform_method = target_transform_method
        self._baseline_prediction = float(np.mean(self._model.predict(X_filtered)))
        # Store training data for linear SHAP (needed for serialization roundtrip)
        shap_type = get_model_info(model_type).get("shap_type", "tree")
        self._X_train = X_filtered if shap_type == "linear" else None

        def loo_factory():
            base_model = build_surrogate_model(model_type, params, random_state=self._random_state, n_samples=n_fit)
            if target_transform_method == "log1p":
                from sklearn.compose import TransformedTargetRegressor

                return TransformedTargetRegressor(
                    regressor=base_model,
                    func=np.log1p,
                    inverse_func=np.expm1,
                )
            if target_transform_method == "yeo_johnson":
                from sklearn.compose import TransformedTargetRegressor
                from sklearn.preprocessing import PowerTransformer

                return TransformedTargetRegressor(
                    regressor=base_model,
                    transformer=PowerTransformer(method="yeo-johnson", standardize=False),
                )
            return base_model

        r2_cv, loo_preds = _loo_r2(X_filtered, y, loo_factory, groups=groups)
        if r2_cv is not None:
            errors = np.abs(y - loo_preds)
            mae = float(np.mean(errors))
            nonzero = np.abs(y) > 1e-8
            mape: Optional[float] = float(np.mean(errors[nonzero] / np.abs(y[nonzero]))) if nonzero.any() else None
        else:
            preds = self._model.predict(X_filtered)
            errors = np.abs(y - preds)
            mae = float(np.mean(errors))
            nonzero = np.abs(y) > 1e-8
            mape = float(np.mean(errors[nonzero] / np.abs(y[nonzero]))) if nonzero.any() else None

        residual_mean: Optional[float] = None
        residual_std: Optional[float] = None
        if r2_cv is not None:
            residuals = y - loo_preds
            residual_mean = float(np.mean(residuals))
            residual_std = float(np.std(residuals))

        n_groups: Optional[int] = None
        if groups is not None:
            n_groups = int(len(np.unique(groups)))

        from sklearn.metrics import r2_score as _r2_score

        train_preds = self._model.predict(X_filtered)
        ss_tot_train = float(np.sum((y - np.mean(y)) ** 2))
        r2_train: Optional[float] = float(_r2_score(y, train_preds)) if ss_tot_train > 0 else None

        fidelity_tier, fidelity_warning = _compute_fidelity_tier(r2_cv)

        self.quality = SurrogateQuality(
            r_squared=r2_cv,
            mae=mae,
            mape=mape,
            n_samples=len(X),
            n_unique_rows=unique_rows,
            constant_features=constant_feats,
            imputation_rates=self.imputation_rates,
            removed_features=removed_feats,
            selected_model_type=model_type,
            selected_model_display_name=get_display_name(model_type),
            n_groups=n_groups,
            fidelity_tier=fidelity_tier,
            fidelity_warning=fidelity_warning,
            residual_mean=residual_mean,
            residual_std=residual_std,
            target_transformed=target_transform_method is not None,
            target_transform_method=target_transform_method,
            permutation_removed_features=perm_removed_feats,
            r_squared_train=r2_train,
        )

    @staticmethod
    def _try_build_shap_explainer(
        model: Any, model_type: str, X_train: Optional[np.ndarray] = None
    ) -> Any:
        """Build a SHAP explainer, returning None if the shap package is not installed."""
        try:
            return build_shap_explainer(model, model_type, X_train=X_train)
        except ModuleNotFoundError:
            logger.debug("shap package not installed; SHAP explanations will be unavailable.")
            return None

    def _filter_X(self, X: np.ndarray) -> np.ndarray:
        """Select only the kept feature columns from X."""
        if hasattr(self, "_keep_indices") and self._keep_indices is not None:
            return X[:, self._keep_indices]
        return X

    def _expand_shap_values(self, sv_filtered: np.ndarray) -> np.ndarray:
        """Map filtered SHAP values back to the full feature list (zeros for removed features)."""
        if not hasattr(self, "_keep_indices") or self._keep_indices is None:
            return sv_filtered
        n_rows = sv_filtered.shape[0]
        full = np.zeros((n_rows, len(self.feature_names)))
        for out_idx, orig_idx in enumerate(self._keep_indices):
            full[:, orig_idx] = sv_filtered[:, out_idx]
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
        import joblib

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
                "keep_indices": self._keep_indices if hasattr(self, "_keep_indices") else None,
                "kept_feature_names": self._kept_feature_names if hasattr(self, "_kept_feature_names") else None,
                "target_transformed": getattr(self, "_target_transform_method", None) is not None,
                "target_transform_method": getattr(self, "_target_transform_method", None),
                "X_train": getattr(self, "_X_train", None),
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
        instance._shap_explainer = cls._try_build_shap_explainer(
            shap_model, model_type, instance._X_train
        )
        instance._is_fitted = True
        instance.quality = quality
        return instance

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")
        return self._model.predict(self._filter_X(X))

    def quality_dict(self) -> dict:
        if self.quality is None:
            return {}
        r2 = round(self.quality.r_squared, 6) if self.quality.r_squared is not None else None
        return {
            "r_squared": r2,
            "mae": round(self.quality.mae, 4),
            "n_samples": self.quality.n_samples,
            "n_unique_rows": self.quality.n_unique_rows,
            "constant_features": self.quality.constant_features,
            "imputation_rates": {k: round(v, 4) for k, v in self.quality.imputation_rates.items()},
            "removed_features": self.quality.removed_features,
            "selected_model_type": self.quality.selected_model_type,
            "n_groups": self.quality.n_groups,
            "fidelity_tier": self.quality.fidelity_tier,
            "fidelity_warning": self.quality.fidelity_warning,
            "residual_mean": round(self.quality.residual_mean, 6) if self.quality.residual_mean is not None else None,
            "residual_std": round(self.quality.residual_std, 6) if self.quality.residual_std is not None else None,
            "target_transformed": self.quality.target_transformed,
            "target_transform_method": self.quality.target_transform_method,
            "permutation_removed_features": self.quality.permutation_removed_features,
            "r_squared_train": round(self.quality.r_squared_train, 6) if self.quality.r_squared_train is not None else None,
        }

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

        ranking_std = self._bootstrap_ranking_std(X, n_bootstrap, np.random.RandomState(random_state))
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
        actual_forecast_value: Optional[float] = None,
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
        actual_forecast_value: Optional[float] = None,
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
        try:
            X_f = self._filter_X(X)
            iv_filtered = self._shap_explainer.shap_interaction_values(X_f[instance_idx : instance_idx + 1])[0]
        except Exception as e:
            logger.warning("SHAP interaction values not available: %s", e)
            return []

        # Expand filtered (n_kept, n_kept) interaction matrix back to full feature space.
        n_feats = len(self.feature_names)
        interaction_values = np.zeros((n_feats, n_feats))
        keep = self._keep_indices if hasattr(self, "_keep_indices") and self._keep_indices is not None else list(range(n_feats))
        for out_i, orig_i in enumerate(keep):
            for out_j, orig_j in enumerate(keep):
                interaction_values[orig_i, orig_j] = iv_filtered[out_i, out_j]

        pairs: list[tuple[float, int, int]] = []
        for i in range(n_feats):
            for j in range(i + 1, n_feats):
                pairs.append((abs(float(interaction_values[i, j])), i, j))
        pairs.sort(reverse=True)

        result = []
        for abs_val, i, j in pairs[:top_k]:
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

    def explain_local_lime(
        self,
        X: np.ndarray,
        instance_idx: int,
        prediction_id: int,
        org_unit: str,
        period: str,
        feature_actual_values: dict[str, float],
        top_k: int = 10,
        output_statistic: str = "median",
        n_samples: Optional[int] = None,
        actual_forecast_value: Optional[float] = None,
    ) -> LocalExplanation:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before explain_local_lime().")

        baseline = float(np.mean(self.predict(X)))
        actual = (
            actual_forecast_value
            if actual_forecast_value is not None
            else float(self.predict(X[instance_idx : instance_idx + 1])[0])
        )

        try:
            from lime.lime_tabular import LimeTabularExplainer

            X_f = self._filter_X(X)
            kept_names = (
                self._kept_feature_names
                if hasattr(self, "_kept_feature_names") and self._kept_feature_names is not None
                else self.feature_names
            )
            effective_n_samples = n_samples if n_samples is not None else min(2000, max(500, 50 * X_f.shape[1]))
            need_lime_rebuild = (
                self._lime_explainer_cache is None
                or self._lime_cached_X is None
                or self._lime_cached_X.shape != X.shape
                or not np.array_equal(self._lime_cached_X, X)
            )
            if need_lime_rebuild:
                self._lime_cached_X = np.array(X, copy=True)
                self._lime_explainer_cache = LimeTabularExplainer(
                    training_data=X_f,
                    feature_names=kept_names,
                    mode="regression",
                    discretize_continuous=False,
                )
            exp = self._lime_explainer_cache.explain_instance(
                X_f[instance_idx],
                self._model.predict,
                num_features=len(kept_names),
                num_samples=effective_n_samples,
            )
            # Use index-based matching to avoid ambiguous substring resolution
            lime_by_feature: dict[str, float] = {}
            for feat_idx, weight in next(iter(exp.local_exp.values()), []):
                if 0 <= feat_idx < len(kept_names):
                    name = kept_names[feat_idx]
                    lime_by_feature[name] = lime_by_feature.get(name, 0.0) + float(weight)

            attributions = [
                FeatureAttribution(
                    feature_name=name,
                    importance=weight,
                    direction="positive" if weight > 0 else "negative",
                    actual_value=feature_actual_values.get(name),
                    baseline_value=baseline,
                )
                for name, weight in lime_by_feature.items()
            ]
        except ModuleNotFoundError:
            logger.debug("lime package not installed; falling back to occlusion for local explanation.")
            occlusion_values = self._local_importances_fallback(X, instance_idx)
            attributions = [
                FeatureAttribution(
                    feature_name=name,
                    importance=float(occlusion_values[i]),
                    direction="positive" if occlusion_values[i] > 0 else "negative",
                    actual_value=feature_actual_values.get(name),
                    baseline_value=baseline,
                )
                for i, name in enumerate(self.feature_names)
            ]

        attributions.sort(key=lambda a: abs(a.importance), reverse=True)

        return LocalExplanation(
            prediction_id=prediction_id,
            org_unit=org_unit,
            period=period,
            method=ExplanationMethod.LIME,
            output_statistic=output_statistic,
            feature_attributions=attributions[:top_k],
            baseline_prediction=baseline,
            actual_prediction=actual,
        )

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
        # Expand filtered importances back to full feature space
        mean_abs_filtered = np.maximum(result.importances_mean, 0.0)
        mean_abs_full = np.zeros(len(self.feature_names))
        for out_idx, orig_idx in enumerate(self._keep_indices):
            mean_abs_full[orig_idx] = mean_abs_filtered[out_idx]

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
        instance = X_f[instance_idx : instance_idx + 1].copy()
        baseline_pred = float(self._model.predict(instance)[0])
        col_means = np.mean(X_f, axis=0)

        attributions_filtered = np.zeros(X_f.shape[1])
        for i in range(X_f.shape[1]):
            occluded = instance.copy()
            occluded[0, i] = col_means[i]
            attributions_filtered[i] = baseline_pred - float(self._model.predict(occluded)[0])

        # Expand back to full feature space
        full = np.zeros(len(self.feature_names))
        for out_idx, orig_idx in enumerate(self._keep_indices):
            full[orig_idx] = attributions_filtered[out_idx]
        return full

    def _bootstrap_ranking_std(
        self,
        X: np.ndarray,
        n_bootstrap: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        n_rows, n_feats = X.shape
        rank_matrix = np.zeros((n_bootstrap, n_feats))
        if self._shap_explainer is not None:
            sv_full = self.shap_values_matrix(X)
            for b in range(n_bootstrap):
                idx = rng.choice(n_rows, size=n_rows, replace=True)
                mean_abs_boot = np.mean(np.abs(sv_full[idx]), axis=0)
                order = np.argsort(-mean_abs_boot)
                ranks = np.empty_like(order)
                ranks[order] = np.arange(n_feats)
                rank_matrix[b] = ranks / max(n_feats - 1, 1)
        else:
            for b in range(n_bootstrap):
                idx = rng.choice(n_rows, size=n_rows, replace=True)
                mean_abs_boot, _ = self._global_importances_fallback(X[idx])
                order = np.argsort(-mean_abs_boot)
                ranks = np.empty_like(order)
                ranks[order] = np.arange(n_feats)
                rank_matrix[b] = ranks / max(n_feats - 1, 1)
        return np.std(rank_matrix, axis=0)


class SurrogateLIMEExplainer(SurrogateSHAPExplainer):
    """
    Same configurable surrogate as SurrogateSHAPExplainer, but uses LIME for attributions.

    This allows the XAI registry to dispatch to the correct explanation method
    without hardcoded checks in the router. The surrogate model (and its type) is
    identical; only the attribution extraction differs.
    """

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
        actual_forecast_value: Optional[float] = None,
    ) -> LocalExplanation:
        return self.explain_local_lime(
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

    def explain_global(
        self,
        X: np.ndarray,
        top_k: int = 10,
        n_bootstrap: int = 20,
        random_state: int = 42,
    ) -> GlobalExplanation:
        """Aggregate LIME importances across a sample of instances for global view."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before explain_global().")

        n_rows = len(X)
        max_instances = min(n_rows, 50)
        rng = np.random.RandomState(random_state)
        indices = rng.choice(n_rows, size=max_instances, replace=False) if n_rows > max_instances else np.arange(n_rows)

        try:
            from lime.lime_tabular import LimeTabularExplainer

            importance_matrix = np.zeros((len(indices), len(self.feature_names)))
            lime_explainer = LimeTabularExplainer(
                training_data=X,
                feature_names=self.feature_names,
                mode="regression",
                discretize_continuous=False,
            )
            global_n_samples = min(1000, max(300, 30 * X.shape[1]))
            for row_idx, data_idx in enumerate(indices):
                exp = lime_explainer.explain_instance(
                    X[data_idx],
                    self.predict,
                    num_features=len(self.feature_names),
                    num_samples=global_n_samples,
                )
                for feat_idx, weight in next(iter(exp.local_exp.values()), []):
                    if 0 <= feat_idx < len(self.feature_names):
                        importance_matrix[row_idx, feat_idx] += float(weight)

            mean_abs = np.mean(np.abs(importance_matrix), axis=0)
            mean_signed = np.mean(importance_matrix, axis=0)
        except ModuleNotFoundError:
            logger.debug("lime package not installed; falling back to permutation importance for global explanation.")
            mean_abs, mean_signed = self._global_importances_fallback(X)

        attributions = []
        for i, name in enumerate(self.feature_names):
            attributions.append(
                FeatureAttribution(
                    feature_name=name,
                    importance=float(mean_abs[i]),
                    direction="positive" if mean_signed[i] >= 0 else "negative",
                )
            )
        attributions.sort(key=lambda a: abs(a.importance), reverse=True)

        return GlobalExplanation(
            method=ExplanationMethod.LIME,
            top_features=attributions[:top_k],
            n_samples=len(indices),
        )
