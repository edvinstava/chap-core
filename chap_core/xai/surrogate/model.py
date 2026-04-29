"""Surrogate model construction, selection, and tuning utilities for XAI."""

import importlib
import inspect
import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from .registry import (
    DEFAULT_MODEL_TYPE,
    SUPPORTED_MODELS,
    get_display_name,
    get_model_info,
    is_model_available,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MODEL_TYPE",
    "SUPPORTED_MODELS",
    "auto_select_best_model_type",
    "build_shap_explainer",
    "build_surrogate_model",
    "get_display_name",
    "get_model_info",
    "loo_r2",
    "make_loo_model_factory",
    "make_model_factory",
    "resolve_model_params",
    "select_and_tune_best_model_type",
    "select_target_transform",
    "tune_surrogate_hyperparameters",
    "wrap_with_transform",
]


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
    cls = getattr(importlib.import_module(module_path), class_name)

    # Ridge, Lasso, and some linear models don't accept random_state.
    # CatBoost uses random_seed instead of random_state.
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
    import shap

    info = get_model_info(model_type)
    shap_type = info["shap_type"]
    if shap_type == "tree":
        return shap.TreeExplainer(model)
    if shap_type == "linear":
        if X_train is None:
            raise ValueError("X_train is required for linear SHAP explainer")
        return shap.LinearExplainer(model, X_train)
    raise ValueError(f"Unsupported shap_type '{shap_type}' for model_type '{model_type}'")


def select_target_transform(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    params: dict,
    random_state: int,
    n_fit: int,
) -> str | None:
    """Choose the best target transform (None, 'log1p', or 'yeo_johnson') via CV."""
    try:
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.model_selection import cross_val_score as _cvs
        from sklearn.preprocessing import PowerTransformer

        X, y = np.asarray(X), np.asarray(y)
        cv_folds = min(n_fit, 5) if n_fit >= 10 else max(2, n_fit)

        def _make_model():
            return build_surrogate_model(model_type, params, random_state=random_state, n_samples=n_fit)

        r2_improvement_threshold = 0.05

        r2_raw = float(np.mean(_cvs(_make_model(), X, y, cv=cv_folds, scoring="r2")))
        best_r2, best_method = r2_raw, None

        if float(np.min(y)) >= 0.0:
            r2_log = float(
                np.mean(
                    _cvs(
                        TransformedTargetRegressor(regressor=_make_model(), func=np.log1p, inverse_func=np.expm1),
                        X,
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
                        X,
                        y,
                        cv=cv_folds,
                        scoring="r2",
                    )
                )
            )
            if r2_yj > best_r2 + r2_improvement_threshold:
                best_r2, best_method = r2_yj, "yeo_johnson"

        return best_method
    except Exception as e:
        logger.debug("Target-transform check failed, using raw y: %s", e)
        return None


def wrap_with_transform(base_model: Any, transform_method: str | None) -> Any:
    """Wrap *base_model* with a TransformedTargetRegressor if *transform_method* is set."""
    if transform_method == "log1p":
        from sklearn.compose import TransformedTargetRegressor

        return TransformedTargetRegressor(regressor=base_model, func=np.log1p, inverse_func=np.expm1)
    if transform_method == "yeo_johnson":
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.preprocessing import PowerTransformer

        return TransformedTargetRegressor(
            regressor=base_model,
            transformer=PowerTransformer(method="yeo-johnson", standardize=False),
        )
    return base_model


# ---------------------------------------------------------------------------
# Auto-select best surrogate
# ---------------------------------------------------------------------------


def loo_r2(
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
    X, y = np.asarray(X), np.asarray(y)
    n = len(X)
    loo_preds = np.zeros(n)
    if n < 4:
        return None, loo_preds

    use_logo = False
    if groups is not None:
        unique_groups = np.unique(groups)
        if len(unique_groups) >= 2:
            use_logo = True

    if use_logo:
        for g in unique_groups:
            test_mask = groups == g
            train_mask = ~test_mask
            if train_mask.sum() == 0:
                continue
            m = model_factory()
            m.fit(X[train_mask], y[train_mask])
            loo_preds[test_mask] = m.predict(X[test_mask])
    else:
        for i in range(n):
            X_train = np.delete(X, i, axis=0)
            y_train = np.delete(y, i)
            m = model_factory()
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
    ranked: list[tuple[float, str]] = []

    for model_type, model_info in SUPPORTED_MODELS.items():
        if not model_info.get("auto_eligible", True):
            continue
        if not is_model_available(model_type):
            logger.debug("Auto-select: skipping %s (package not available)", model_type)
            continue
        try:
            factory = make_loo_model_factory(model_type, random_state=random_state, n_samples=len(X))
            r2, _ = loo_r2(X, y, factory, groups=groups)
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


class _PatienceCallback:
    """Stop an Optuna study after *patience* consecutive completed trials with no improvement.

    Create a new instance per study; do not reuse across multiple calls to ``study.optimize``.
    """

    def __init__(self, patience: int = 20) -> None:
        self._patience = patience
        self._best: float = float("-inf")
        self._no_improve: int = 0

    def __call__(self, study: Any, trial: Any) -> None:
        import optuna

        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        if study.best_value > self._best:
            self._best = study.best_value
            self._no_improve = 0
        else:
            self._no_improve += 1
        if self._no_improve >= self._patience:
            study.stop()


def _suggest_param(trial, name: str, spec: dict[str, Any], n_samples: int) -> Any:
    """Map a tunable_param spec entry to an Optuna trial suggestion."""
    kind = spec["type"]
    low = spec["low"]
    high = spec.get("high")
    if high is None:
        fraction = spec.get("high_n_fraction", 5)
        high = max(low, n_samples // fraction)
    if n_samples <= 80:
        if name == "min_samples_leaf":
            low = max(int(low), 2)
        elif name == "min_samples_split":
            low = max(int(low), 4)
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
    across folds. The optimisation objective also penalizes fold-score variance and
    train-vs-validation overfitting gap to improve generalization stability on
    small datasets. ``GroupKFold`` is used when *groups* has ≥ 2 unique values.
    """
    import optuna
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import GroupKFold, KFold

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X, y = np.asarray(X), np.asarray(y)
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

    n_features = int(X.shape[1]) if X.ndim == 2 else 1
    low_sample_regime = n_samples <= 80
    stability_weight = 0.18 if low_sample_regime else 0.08
    overfit_weight = 0.25 if low_sample_regime else 0.12
    if model_type == "gradient_boosting" and low_sample_regime:
        stability_weight = 0.22
        overfit_weight = 0.35
    if n_features <= 6:
        overfit_weight += 0.05

    def objective(trial: optuna.Trial) -> float:
        trial_params = {name: _suggest_param(trial, name, spec, n_samples) for name, spec in tunable.items()}
        # XGBoost early stopping requires eval_set in fit(); omit it during tuning
        if model_type == "xgboost":
            trial_params.pop("early_stopping_rounds", None)
        params = resolve_model_params(model_type, {}, trial_params)

        def model_template():
            return build_surrogate_model(model_type, params, random_state=random_state, n_samples=n_samples)

        split_kwargs = {"groups": effective_groups} if effective_groups is not None else {}
        splits = list(cv_splitter.split(X, y, **split_kwargs))
        val_scores: list[float] = []
        train_scores: list[float] = []
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            m = model_template()
            m.fit(X[train_idx], y[train_idx])
            val_score = -mean_squared_error(y[test_idx], m.predict(X[test_idx])) / y_var
            train_score = -mean_squared_error(y[train_idx], m.predict(X[train_idx])) / y_var
            val_scores.append(val_score)
            train_scores.append(train_score)
            current_mean = float(np.mean(val_scores))
            current_var_penalty = stability_weight * float(np.std(val_scores)) if len(val_scores) > 1 else 0.0
            current_overfit_penalty = overfit_weight * float(
                np.mean(np.maximum(0.0, np.asarray(train_scores) - np.asarray(val_scores)))
            )
            trial.report(current_mean - current_var_penalty - current_overfit_penalty, fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned
        mean_val = float(np.mean(val_scores))
        var_penalty = stability_weight * float(np.std(val_scores)) if len(val_scores) > 1 else 0.0
        mean_gap = float(np.mean(np.maximum(0.0, np.asarray(train_scores) - np.asarray(val_scores))))
        if mean_gap > 0.5:
            return float("-inf")
        overfit_penalty = overfit_weight * mean_gap
        return mean_val - var_penalty - overfit_penalty

    # Use enough random startup trials so TPE has a good initial landscape estimate.
    n_startup = max(10, min(30, n_trials // 4))
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state, n_startup_trials=n_startup),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=n_startup, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=n_trials, callbacks=[_PatienceCallback(patience=20)])
    return study.best_params, study.best_value


def _score_fixed_params(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    params: dict[str, Any],
    groups: np.ndarray | None,
    random_state: int,
) -> float:
    """Score a fixed param set via CV with a given random_state. Returns mean -MSE/Var(y).

    Uses plain (unpenalized) score intentionally so the 3x3 cross-evaluation in
    ``_run_multi_seed_tuning`` ranks candidates on neutral predictive performance,
    not on a seed-dependent variance penalty.
    """
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import GroupKFold, KFold

    X, y = np.asarray(X), np.asarray(y)
    n_samples = len(X)
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

    split_kwargs = {"groups": effective_groups} if effective_groups is not None else {}
    splits = list(cv_splitter.split(X, y, **split_kwargs))

    resolved = resolve_model_params(model_type, {}, params)
    val_scores: list[float] = []
    for train_idx, test_idx in splits:
        m = build_surrogate_model(model_type, resolved, random_state=random_state, n_samples=n_samples)
        m.fit(X[train_idx], y[train_idx])
        val_scores.append(-mean_squared_error(y[test_idx], m.predict(X[test_idx])) / y_var)

    return float(np.mean(val_scores))


_MULTI_SEED_SEEDS = (42, 123)


def _run_multi_seed_tuning(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    n_trials: int,
    groups: np.ndarray | None,
    seeds: tuple[int, ...] = _MULTI_SEED_SEEDS,
) -> tuple[dict[str, Any], float]:
    """Run N seeded Optuna studies, cross-evaluate all best-param candidates, return the one
    with the best median score across seeds.

    For small datasets (n <= 80) where a single random seed may yield unstable results,
    this selects params that generalise consistently rather than just performing well on
    one lucky CV split.
    """
    candidates: list[dict[str, Any]] = []
    for seed in seeds:
        params, _ = _run_tuning_study(X, y, model_type, n_trials, groups, seed)
        candidates.append(params)

    scores_matrix = [[_score_fixed_params(X, y, model_type, p, groups, seed) for seed in seeds] for p in candidates]
    medians = [float(np.median(row)) for row in scores_matrix]
    best_idx = int(np.argmax(medians))
    logger.info(
        "Multi-seed tuning (%s): candidate medians=%s, selected candidate %d",
        model_type,
        [round(m, 4) for m in medians],
        best_idx,
    )
    return candidates[best_idx], medians[best_idx]


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

    For ``n_samples <= 80``, runs multi-seed tuning and returns the most stable params.

    Returns the best params dict found.
    """
    if len(X) <= 80:
        params, _ = _run_multi_seed_tuning(X, y, model_type, n_trials, groups)
    else:
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

    For ``n_samples <= 80``, each candidate is tuned with multi-seed tuning for
    more stable hyperparameter selection.

    Falls back to the LOO winner with empty params when tuning fails for all
    candidates.
    """
    candidates = auto_select_best_model_type(X, y, groups=groups, random_state=random_state)
    n_candidates = min(3, len(candidates))
    candidates = candidates[:n_candidates]
    trials_per = max(20, n_trials // n_candidates)
    use_multi_seed = len(X) <= 80

    best_model_type = candidates[0]
    best_params: dict[str, Any] = {}
    best_score = float("-inf")

    for model_type in candidates:
        try:
            if use_multi_seed:
                params, score = _run_multi_seed_tuning(X, y, model_type, trials_per, groups)
            else:
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
