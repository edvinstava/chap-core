"""
Registry of surrogate model types for XAI explainers.

SUPPORTED_MODELS is the single source of truth. Each entry defines:
  - display_name: human-readable name shown in the UI
  - class_dotted: dotted import path to the sklearn-compatible regressor class
  - default_params: parameters used when no tuning or overrides are provided
  - loo_params: simplified parameters for fast leave-one-out cross-validation
  - tunable_params: Optuna search-space spec (see _suggest_param for format)
  - shap_type: "tree" or "linear" — determines which SHAP algorithm is used
  - optional: if True, the model is skipped gracefully when its package is missing

To add a new model type:
  1. Add an entry to SUPPORTED_MODELS.
  2. Optionally register a matching XaiMethod in xai_method_seed.py.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

SUPPORTED_MODELS: dict[str, dict[str, Any]] = {
    "hist_gradient_boosting": {
        "display_name": "Histogram Gradient Boosting",
        "class_dotted": "sklearn.ensemble.HistGradientBoostingRegressor",
        "default_params": {
            "max_iter": 600,
            "max_depth": 6,
            "learning_rate": 0.05,
            "early_stopping": True,
            "n_iter_no_change": 20,
            "validation_fraction": 0.15,
            "max_leaf_nodes": 31,
            "l2_regularization": 0.1,
            "min_samples_leaf": 10,
        },
        "loo_params": {"max_iter": 250, "max_depth": 6, "early_stopping": False},
        "tunable_params": {
            "max_iter": {"type": "int", "low": 300, "high": 2000},
            "max_depth": {"type": "int", "low": 2, "high": 12},
            "learning_rate": {"type": "float", "low": 0.005, "high": 0.2, "log": True},
            "min_samples_leaf": {"type": "int", "low": 1, "high": None, "high_n_fraction": 4},
            "max_leaf_nodes": {"type": "int", "low": 15, "high": 255},
            "l2_regularization": {"type": "float", "low": 0.0, "high": 10.0},
            "max_features": {"type": "float", "low": 0.3, "high": 1.0},
        },
        "shap_type": "tree",
        "optional": False,
    },
    "xgboost": {
        "display_name": "XGBoost",
        "class_dotted": "xgboost.XGBRegressor",
        "default_params": {
            "n_estimators": 600,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_weight": 3,
            "verbosity": 0,
            "eval_metric": "rmse",
            "early_stopping_rounds": 20,
        },
        "loo_params": {
            "n_estimators": 250,
            "max_depth": 5,
            "verbosity": 0,
        },
        "tunable_params": {
            "n_estimators": {"type": "int", "low": 200, "high": 2000},
            "max_depth": {"type": "int", "low": 2, "high": 12},
            "learning_rate": {"type": "float", "low": 0.005, "high": 0.3, "log": True},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.4, "high": 1.0},
            "reg_alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True},
            "reg_lambda": {"type": "float", "low": 1e-4, "high": 10.0, "log": True},
            "min_child_weight": {"type": "int", "low": 1, "high": None, "high_n_fraction": 5},
            "gamma": {"type": "float", "low": 0.0, "high": 5.0},
        },
        "shap_type": "tree",
        "optional": True,
    },
    "lightgbm": {
        "display_name": "LightGBM",
        "class_dotted": "lightgbm.LGBMRegressor",
        "default_params": {
            "n_estimators": 600,
            "max_depth": -1,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_samples": 10,
            "verbose": -1,
        },
        "loo_params": {
            "n_estimators": 250,
            "num_leaves": 20,
            "verbose": -1,
        },
        "tunable_params": {
            "n_estimators": {"type": "int", "low": 200, "high": 2000},
            "num_leaves": {"type": "int", "low": 15, "high": 255},
            "learning_rate": {"type": "float", "low": 0.005, "high": 0.3, "log": True},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.4, "high": 1.0},
            "reg_alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True},
            "reg_lambda": {"type": "float", "low": 1e-4, "high": 10.0, "log": True},
            "min_child_samples": {"type": "int", "low": 1, "high": None, "high_n_fraction": 4},
            "min_split_gain": {"type": "float", "low": 0.0, "high": 1.0},
        },
        "shap_type": "tree",
        "optional": True,
    },
    "gradient_boosting": {
        "display_name": "Gradient Boosted Trees (sklearn)",
        "class_dotted": "sklearn.ensemble.GradientBoostingRegressor",
        "default_params": {
            "n_estimators": 600,
            "max_depth": 5,
            "learning_rate": 0.05,
            "n_iter_no_change": 15,
            "validation_fraction": 0.15,
            "subsample": 0.8,
            "max_features": 0.8,
            "min_samples_leaf": 5,
        },
        "loo_params": {"n_estimators": 150, "max_depth": 4},
        "tunable_params": {
            "n_estimators": {"type": "int", "low": 200, "high": 1500},
            "max_depth": {"type": "int", "low": 2, "high": 8},
            "learning_rate": {"type": "float", "low": 0.005, "high": 0.2, "log": True},
            "min_samples_leaf": {"type": "int", "low": 1, "high": None, "high_n_fraction": 5},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
            "min_samples_split": {"type": "int", "low": 2, "high": None, "high_n_fraction": 10},
            "max_features": {"type": "float", "low": 0.3, "high": 1.0},
        },
        "shap_type": "tree",
        "optional": False,
    },
    "random_forest": {
        "display_name": "Random Forest",
        "class_dotted": "sklearn.ensemble.RandomForestRegressor",
        "default_params": {"n_estimators": 800, "max_depth": None, "min_samples_leaf": 2, "max_features": 0.5},
        "loo_params": {"n_estimators": 80, "max_depth": 8},
        "tunable_params": {
            "n_estimators": {"type": "int", "low": 200, "high": 1200},
            "max_depth": {"type": "int", "low": 3, "high": 30},
            "min_samples_leaf": {"type": "int", "low": 1, "high": None, "high_n_fraction": 5},
            "min_samples_split": {"type": "int", "low": 2, "high": None, "high_n_fraction": 10},
            "max_features": {"type": "float", "low": 0.2, "high": 1.0},
        },
        "shap_type": "tree",
        "optional": False,
    },
    "extra_trees": {
        "display_name": "Extra Trees",
        "class_dotted": "sklearn.ensemble.ExtraTreesRegressor",
        "default_params": {"n_estimators": 800, "max_depth": None, "min_samples_leaf": 2, "max_features": 0.5},
        "loo_params": {"n_estimators": 80, "max_depth": 8},
        "tunable_params": {
            "n_estimators": {"type": "int", "low": 200, "high": 1200},
            "max_depth": {"type": "int", "low": 3, "high": 30},
            "min_samples_leaf": {"type": "int", "low": 1, "high": None, "high_n_fraction": 5},
            "min_samples_split": {"type": "int", "low": 2, "high": None, "high_n_fraction": 10},
            "max_features": {"type": "float", "low": 0.2, "high": 1.0},
        },
        "shap_type": "tree",
        "optional": False,
    },
    "decision_tree": {
        "display_name": "Decision Tree",
        "class_dotted": "sklearn.tree.DecisionTreeRegressor",
        "default_params": {"max_depth": 6, "min_samples_leaf": 2},
        "loo_params": {"max_depth": 4, "min_samples_leaf": 2},
        "tunable_params": {
            "max_depth": {"type": "int", "low": 2, "high": 15},
            "min_samples_leaf": {"type": "int", "low": 1, "high": None, "high_n_fraction": 5},
            "min_samples_split": {"type": "int", "low": 2, "high": None, "high_n_fraction": 10},
            "max_features": {"type": "float", "low": 0.3, "high": 1.0},
        },
        "shap_type": "tree",
        "optional": False,
    },
    "ridge": {
        "display_name": "Ridge Regression",
        "class_dotted": "sklearn.linear_model.Ridge",
        "default_params": {"alpha": 1.0},
        "loo_params": {"alpha": 1.0},
        "tunable_params": {
            "alpha": {"type": "float", "low": 0.001, "high": 1000.0, "log": True},
        },
        "shap_type": "linear",
        "optional": False,
    },
    "lasso": {
        "display_name": "Lasso Regression",
        "class_dotted": "sklearn.linear_model.Lasso",
        "default_params": {"alpha": 0.1, "max_iter": 10000},
        "loo_params": {"alpha": 0.1, "max_iter": 5000},
        "tunable_params": {
            "alpha": {"type": "float", "low": 0.0001, "high": 100.0, "log": True},
        },
        "shap_type": "linear",
        "optional": False,
    },
    "catboost": {
        "display_name": "CatBoost",
        "class_dotted": "catboost.CatBoostRegressor",
        "default_params": {
            "iterations": 600,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
            "min_data_in_leaf": 5,
            "verbose": 0,
        },
        "loo_params": {
            "iterations": 250,
            "depth": 5,
            "verbose": 0,
        },
        "tunable_params": {
            "iterations": {"type": "int", "low": 200, "high": 2000},
            "depth": {"type": "int", "low": 2, "high": 10},
            "learning_rate": {"type": "float", "low": 0.005, "high": 0.3, "log": True},
            "l2_leaf_reg": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
            "min_data_in_leaf": {"type": "int", "low": 1, "high": None, "high_n_fraction": 4},
            "colsample_bylevel": {"type": "float", "low": 0.4, "high": 1.0},
        },
        "shap_type": "tree",
        "optional": True,
    },
}

DEFAULT_MODEL_TYPE = "hist_gradient_boosting"

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def get_model_info(model_type: str) -> dict[str, Any]:
    """Return registry entry for model_type, raising ValueError if unknown."""
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown surrogate model type '{model_type}'. Supported: {sorted(SUPPORTED_MODELS)}")
    return SUPPORTED_MODELS[model_type]


def get_display_name(model_type: str) -> str:
    """Return the human-readable display name for a model type."""
    info = SUPPORTED_MODELS.get(model_type)
    if info is None:
        return model_type
    value = info.get("display_name", model_type)
    return str(value)


def _is_model_available(model_type: str) -> bool:
    """Return True if the model's package can be imported."""
    info = SUPPORTED_MODELS.get(model_type, {})
    if not info.get("optional", False):
        return True
    module_path = info["class_dotted"].rsplit(".", 1)[0]
    try:
        import importlib

        importlib.import_module(module_path)
        return True
    except (ImportError, OSError):
        return False


def build_surrogate_model(
    model_type: str, params: dict[str, Any], random_state: int = 42, n_samples: int | None = None
):
    """Instantiate an unfitted sklearn-compatible surrogate model.

    When *n_samples* is provided and < 20, early-stopping parameters are
    removed because the validation split would be too small.
    When *n_samples* < 30, tree complexity is clamped to prevent overfitting.
    HistGradientBoostingRegressor defaults to min_samples_leaf=20; with small n
    that forbids any split (constant predictions, zero TreeSHAP).
    """
    info = get_model_info(model_type)
    effective_params = dict(params)

    if n_samples is not None and model_type == "hist_gradient_boosting":
        msl = effective_params.get("min_samples_leaf", 20)
        cap = max(1, n_samples // 3)
        effective_params["min_samples_leaf"] = min(int(msl), cap)

    if n_samples is not None and n_samples < 32:
        for key in ("n_iter_no_change", "validation_fraction", "early_stopping", "early_stopping_rounds"):
            effective_params.pop(key, None)

    if n_samples is not None and n_samples < 30:
        for key, cap in (("max_depth", 4), ("max_leaf_nodes", 15), ("num_leaves", 15)):
            if key in effective_params and effective_params[key] is not None and effective_params[key] != -1:
                effective_params[key] = min(effective_params[key], cap)
        for key in ("n_estimators", "max_iter"):
            if key in effective_params:
                effective_params[key] = min(effective_params[key], 200)

    module_path, class_name = info["class_dotted"].rsplit(".", 1)
    import importlib

    cls = getattr(importlib.import_module(module_path), class_name)
    # Ridge, Lasso, and some linear models don't accept random_state.
    # CatBoost uses random_seed instead of random_state.
    import inspect

    sig_params = inspect.signature(cls).parameters
    if "random_state" in sig_params:
        return cls(random_state=random_state, **effective_params)
    if "random_seed" in sig_params:
        return cls(random_seed=random_state, **effective_params)
    return cls(**effective_params)


def resolve_model_params(
    model_type: str,
    model_config: dict[str, Any],
    hyperparams: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Merge default, config-level, and tuned params into a final params dict.

    Priority (lowest → highest): defaults < model_config overrides < hyperparams.
    The "model_type" key in model_config is ignored here.
    """
    info = get_model_info(model_type)
    params: dict[str, Any] = {**info["default_params"]}
    params.update({k: v for k, v in model_config.items() if k != "model_type"})
    if hyperparams:
        params.update(hyperparams)
    return params


def make_loo_model_factory(model_type: str, random_state: int = 42, n_samples: int | None = None) -> Callable:
    """Return a zero-argument callable that produces a cheap LOO model instance."""
    info = get_model_info(model_type)
    loo_params = info["loo_params"]
    return lambda: build_surrogate_model(model_type, loo_params, random_state=random_state, n_samples=n_samples)


def make_model_factory(
    model_type: str,
    params: dict[str, Any],
    random_state: int = 42,
    max_estimators: int = 400,
    n_samples: int | None = None,
) -> Callable:
    """Return a zero-argument callable that produces a model with *params*.

    To keep LOO runtime bounded, ``n_estimators`` / ``max_iter`` are capped at
    *max_estimators*.  When *n_samples* is provided it is forwarded to
    ``build_surrogate_model`` for small-dataset guards.
    """
    capped = dict(params)
    for key in ("n_estimators", "max_iter"):
        if key in capped:
            capped[key] = min(capped[key], max_estimators)
    return lambda: build_surrogate_model(model_type, capped, random_state=random_state, n_samples=n_samples)


def build_shap_explainer(model, model_type: str, X_train: np.ndarray | None = None):
    """Create a SHAP explainer appropriate for the given surrogate model type.

    For linear models, *X_train* is required as background data.
    For XGBoost/LightGBM, TreeExplainer is used with model_output='raw'.
    """
    import shap  # type: ignore[import-untyped]

    info = get_model_info(model_type)
    shap_type = info["shap_type"]
    if shap_type == "tree":
        return shap.TreeExplainer(model)
    if shap_type == "linear":
        if X_train is None:
            raise ValueError("X_train is required for linear SHAP explainer")
        return shap.LinearExplainer(model, X_train)
    raise ValueError(f"Unsupported shap_type '{shap_type}' for model_type '{model_type}'")


# ---------------------------------------------------------------------------
# Auto-select best surrogate
# ---------------------------------------------------------------------------


def _loo_r2(
    X: np.ndarray,
    y: np.ndarray,
    model_factory: Callable,
    groups: np.ndarray | None = None,
) -> tuple[float | None, np.ndarray]:
    """
    Leave-one-out (or leave-one-group-out) R² for a model factory.

    When ``groups`` is provided and has at least 2 unique values, uses
    leave-one-group-out CV (hold out all rows for one org_unit at a time).
    Falls back to plain LOO otherwise.

    Returns ``(r2, loo_preds)`` where ``loo_preds`` is the array of
    held-out predictions (all zeros / NaN when n < 4).
    """
    n = len(X)
    loo_preds = np.zeros(n)
    if n < 4:
        return None, loo_preds

    use_logo = False
    if groups is not None:
        unique_groups = np.unique(groups)
        if len(unique_groups) >= 2:
            use_logo = True

    import warnings

    if use_logo:
        for g in unique_groups:
            test_mask = groups == g
            train_mask = ~test_mask
            if train_mask.sum() == 0:
                continue
            m = model_factory()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names")
                m.fit(X[train_mask], y[train_mask])
                loo_preds[test_mask] = m.predict(X[test_mask])
    else:
        for i in range(n):
            X_train = np.delete(X, i, axis=0)
            y_train = np.delete(y, i)
            m = model_factory()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names")
                m.fit(X_train, y_train)
                loo_preds[i] = m.predict(X[i : i + 1])[0]

    ss_res = np.sum((y - loo_preds) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return r2, loo_preds


def auto_select_best_model_type(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    random_state: int = 42,
) -> list[str]:
    """
    Try all supported surrogate model types with their cheap LOO params and
    return them ranked by LOO-R², best first.  Returns the top-3 types so
    callers can tune multiple candidates and pick the winner.

    Optional models (XGBoost, LightGBM) are included when their packages are
    available, and silently skipped otherwise.

    Falls back to ``[DEFAULT_MODEL_TYPE]`` if no model can be evaluated
    (e.g. too few samples).
    """
    import logging

    logger = logging.getLogger(__name__)

    ranked: list[tuple[float, str]] = []

    for model_type in SUPPORTED_MODELS:
        if not _is_model_available(model_type):
            logger.debug("Auto-select: skipping %s (package not available)", model_type)
            continue
        try:
            factory = make_loo_model_factory(model_type, random_state=random_state, n_samples=len(X))
            r2, _ = _loo_r2(X, y, factory, groups=groups)
        except Exception as exc:
            logger.debug("Auto-select: %s failed with %s", model_type, exc)
            continue
        logger.info("Auto-select: %s LOO-R²=%.4f", model_type, r2 if r2 is not None else float("nan"))
        if r2 is not None:
            ranked.append((r2, model_type))

    if not ranked:
        logger.info("Auto-select: no model evaluated, falling back to %s", DEFAULT_MODEL_TYPE)
        return [DEFAULT_MODEL_TYPE]

    ranked.sort(key=lambda t: t[0], reverse=True)
    top = [mt for _, mt in ranked[:3]]
    logger.info("Auto-select: top models %s", top)
    return top


# ---------------------------------------------------------------------------
# Optuna tuning
# ---------------------------------------------------------------------------


def _suggest_param(trial, name: str, spec: dict[str, Any], n_samples: int) -> Any:
    """Map a tunable_param spec entry to an Optuna trial suggestion."""
    kind = spec["type"]
    low = spec["low"]
    high = spec.get("high")
    if high is None:
        fraction = spec.get("high_n_fraction", 5)
        high = max(low, n_samples // fraction)
    if kind == "int":
        return trial.suggest_int(name, low, high)
    if kind == "float":
        return trial.suggest_float(name, low, high, log=spec.get("log", False))
    raise ValueError(f"Unknown tunable_param type '{kind}' for param '{name}'")


def _run_tuning_study(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    n_trials: int,
    groups: np.ndarray | None,
    random_state: int,
) -> tuple[dict[str, Any], float]:
    """Run an Optuna study for *model_type* and return ``(best_params, best_cv_score)``.

    The CV score is ``-MSE / Var(y)`` (scale-invariant, ≈ ``-(1 - R²)``), averaged
    across folds.  ``GroupKFold`` is used when *groups* has ≥ 2 unique values.
    """
    import optuna
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import GroupKFold, KFold

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    info = get_model_info(model_type)
    tunable = info["tunable_params"]
    n_samples = len(X)

    # Normalisation constant so the objective is scale-invariant (approximately -(1-R^2)).
    y_var = float(np.var(y)) + 1e-8

    effective_groups: np.ndarray | None = None
    if groups is not None and len(np.unique(groups)) >= 2:
        effective_groups = groups
        n_unique = len(np.unique(groups))
        cv_splitter = GroupKFold(n_splits=min(n_unique, 5))
    else:
        n_splits = min(5, max(2, n_samples // 3)) if n_samples >= 6 else max(2, min(n_samples, 5))
        n_splits = min(n_splits, n_samples)
        if n_splits < 2:
            n_splits = 2
        cv_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        params = {name: _suggest_param(trial, name, spec, n_samples) for name, spec in tunable.items()}
        # XGBoost early stopping requires eval_set in fit(); omit it during tuning
        if model_type == "xgboost":
            params.pop("early_stopping_rounds", None)
        model_template = lambda: build_surrogate_model(
            model_type, params, random_state=random_state, n_samples=n_samples
        )
        split_kwargs = {"groups": effective_groups} if effective_groups is not None else {}
        splits = list(cv_splitter.split(X, y, **split_kwargs))
        import warnings

        scores: list[float] = []
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            m = model_template()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names")
                m.fit(X[train_idx], y[train_idx])
                fold_score = -mean_squared_error(y[test_idx], m.predict(X[test_idx])) / y_var
            scores.append(fold_score)
            trial.report(float(np.mean(scores)), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned
        return float(np.mean(scores))

    # Use enough random startup trials so TPE has a good initial landscape estimate.
    n_startup = max(10, min(30, n_trials // 4))
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state, n_startup_trials=n_startup),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=n_startup, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value


def tune_surrogate_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = DEFAULT_MODEL_TYPE,
    n_trials: int = 30,
    groups: np.ndarray | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Use Optuna to find optimal hyperparameters for the given surrogate model type.

    Uses ``neg_mean_squared_error`` as the Optuna objective (more stable than R²
    for optimisation on small datasets).  When ``groups`` is provided and has at
    least 2 unique values, uses ``GroupKFold`` so that all rows from the same
    org_unit stay in the same fold.

    Returns the best params dict found.
    """
    params, _ = _run_tuning_study(X, y, model_type, n_trials, groups, random_state)
    return params


def select_and_tune_best_model_type(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    n_trials: int = 80,
    random_state: int = 42,
) -> tuple[str, dict[str, Any]]:
    """Select the best surrogate model type and its tuned hyperparameters.

    Combines LOO-based candidate ranking with Optuna tuning so that the final
    winner is chosen on the same CV metric used for fitting, not on cheap LOO
    proxies alone.

    Steps:
      1. Run ``auto_select_best_model_type`` to get up to 3 LOO-ranked candidates.
      2. Tune each candidate with ``n_trials // n_candidates`` trials (floor: 20).
      3. Return ``(model_type, best_params)`` for the candidate with the highest
         tuned CV score.

    Falls back to the LOO winner with empty params when tuning fails for all
    candidates.
    """
    import logging

    logger = logging.getLogger(__name__)

    candidates = auto_select_best_model_type(X, y, groups=groups, random_state=random_state)
    n_candidates = min(3, len(candidates))
    candidates = candidates[:n_candidates]
    trials_per = max(20, n_trials // n_candidates)

    best_model_type = candidates[0]
    best_params: dict[str, Any] = {}
    best_score = float("-inf")

    for model_type in candidates:
        try:
            params, score = _run_tuning_study(X, y, model_type, trials_per, groups, random_state)
            logger.info("Candidate %s tuned CV score=%.4f", model_type, score)
            if score > best_score:
                best_score = score
                best_model_type = model_type
                best_params = params
        except Exception as exc:
            logger.warning("Candidate %s tuning failed: %s", model_type, exc)

    logger.info("Selected surrogate: %s (tuned CV score=%.4f)", best_model_type, best_score)
    return best_model_type, best_params
