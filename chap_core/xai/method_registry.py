NATIVE_SHAP = "native_shap"
SHAP_AUTO = "shap_auto"
LIME_AUTO = "lime_auto"

XAI_METHODS = [
    {
        "id": 1,
        "name": "shap_auto",
        "display_name": "SHAP \u2014 Auto (best surrogate)",
        "description": (
            "Automatically benchmarks all available surrogate models using leave-one-out R\u00b2 "
            "(XGBoost, LightGBM, Histogram Gradient Boosting, Random Forest, and others), "
            "tunes the top candidates with Optuna, and applies TreeSHAP for exact, "
            "additive feature attributions. Recommended for most use cases."
        ),
        "method_type": "surrogate_shap_auto",
        "author": "CHAP",
        "archived": False,
        "supported_visualizations": ["importance", "waterfall", "beeswarm"],
    },
    {
        "id": 2,
        "name": "shap_xgboost",
        "display_name": "SHAP \u2014 XGBoost",
        "description": (
            "Fits an XGBoost surrogate on stored predictions, then applies TreeSHAP "
            "for exact, additive feature attributions. Often the most accurate surrogate "
            "for structured tabular data."
        ),
        "method_type": "surrogate_shap",
        "author": "CHAP",
        "archived": False,
        "supported_visualizations": ["importance", "waterfall", "beeswarm"],
    },
    {
        "id": 3,
        "name": "shap_lightgbm",
        "display_name": "SHAP \u2014 LightGBM",
        "description": (
            "Fits a LightGBM surrogate on stored predictions, then applies TreeSHAP "
            "for exact, additive feature attributions. Fast training with strong accuracy."
        ),
        "method_type": "surrogate_shap",
        "author": "CHAP",
        "archived": False,
        "supported_visualizations": ["importance", "waterfall", "beeswarm"],
    },
    {
        "id": 4,
        "name": "shap_hist_gradient_boosting",
        "display_name": "SHAP \u2014 Histogram Gradient Boosting",
        "description": (
            "Fits a scikit-learn HistGradientBoostingRegressor surrogate on stored predictions, "
            "then applies TreeSHAP for exact feature attributions. Native missing-value support."
        ),
        "method_type": "surrogate_shap",
        "author": "CHAP",
        "archived": False,
        "supported_visualizations": ["importance", "waterfall", "beeswarm"],
    },
    {
        "id": 5,
        "name": "shap_random_forest",
        "display_name": "SHAP \u2014 Random Forest",
        "description": (
            "Fits a Random Forest surrogate on stored predictions, "
            "then applies TreeSHAP for exact, additive feature attributions."
        ),
        "method_type": "surrogate_shap",
        "author": "CHAP",
        "archived": False,
        "supported_visualizations": ["importance", "waterfall", "beeswarm"],
    },
    {
        "id": 6,
        "name": "shap_gradient_boosting",
        "display_name": "SHAP \u2014 Gradient Boosted Trees (sklearn)",
        "description": (
            "Fits a scikit-learn GradientBoostingRegressor surrogate on stored predictions, "
            "then applies TreeSHAP for exact feature attributions."
        ),
        "method_type": "surrogate_shap",
        "author": "CHAP",
        "archived": False,
        "supported_visualizations": ["importance", "waterfall", "beeswarm"],
    },
    {
        "id": 7,
        "name": "shap_extra_trees",
        "display_name": "SHAP \u2014 Extra Trees",
        "description": (
            "Fits an Extra Trees surrogate on stored predictions, "
            "then applies TreeSHAP for exact, additive feature attributions. "
            "Faster training than Random Forest with comparable accuracy."
        ),
        "method_type": "surrogate_shap",
        "author": "CHAP",
        "archived": False,
        "supported_visualizations": ["importance", "waterfall", "beeswarm"],
    },
    {
        "id": 8,
        "name": "lime_auto",
        "display_name": "LIME \u2014 Auto (best surrogate)",
        "description": (
            "Automatically selects the surrogate model with the best leave-one-out R\u00b2, "
            "then applies LIME for local, per-instance feature attribution."
        ),
        "method_type": "surrogate_lime_auto",
        "author": "CHAP",
        "archived": False,
        "supported_visualizations": ["importance"],
    },
    {
        "id": 9,
        "name": "native_shap",
        "display_name": "SHAP \u2014 Native (from model)",
        "description": (
            "Uses SHAP values computed directly by the prediction model. "
            "No surrogate approximation is needed \u2014 these are exact attributions "
            "from the model itself. Only available when the model provides native SHAP output."
        ),
        "method_type": "native_shap",
        "author": "Model",
        "archived": False,
        "supported_visualizations": ["importance", "waterfall", "beeswarm"],
    },
]
