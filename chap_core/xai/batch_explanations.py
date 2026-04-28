from chap_core.database.database import SessionWrapper
from chap_core.database.tables import Prediction
from chap_core.database.xai_tables import PredictionExplanation
from chap_core.log_config import get_status_logger
from chap_core.xai.method_registry import NATIVE_SHAP
from chap_core.xai.responses.native_shap import has_native_shap
from chap_core.xai.responses.quality import quality_response_dict
from chap_core.xai.router_services import forecast_actual_value, persist_global_entry, resolve_feature_names
from chap_core.xai.surrogate.methods import METHOD_TO_MODEL_TYPE
from chap_core.xai.surrogate.pipeline import build_surrogate_data, fit_surrogate_explainer


def run_explanations_task(
    prediction_id: int,
    xai_method_name: str,
    output_statistic: str,
    top_k: int,
    session: SessionWrapper,
) -> None:
    status_logger = get_status_logger()
    status_logger.info(
        "Starting XAI explanations (prediction=%d, method=%s, statistic=%s)",
        prediction_id,
        xai_method_name,
        output_statistic,
    )
    prediction = session.session.get(Prediction, prediction_id)
    if prediction is None:
        raise ValueError(f"Prediction {prediction_id} not found")

    forecasts = prediction.forecasts
    if not forecasts:
        raise ValueError(f"No forecasts found for prediction {prediction_id}")

    if xai_method_name == NATIVE_SHAP:
        if not has_native_shap(prediction):
            raise ValueError(f"Prediction {prediction_id} has no native SHAP data")
        status_logger.info("Native SHAP: explanations already stored, skipping surrogate fitting")
        return

    dataset = session.get_dataset(prediction.dataset_id)
    feature_names = resolve_feature_names(dataset)

    model_type = METHOD_TO_MODEL_TYPE.get(xai_method_name, "auto")
    X, y, groups, imputation_rates, covariate_provenance_rows = build_surrogate_data(
        forecasts, dataset, feature_names, output_statistic
    )
    explainer = fit_surrogate_explainer(X, y, groups, model_type, feature_names, imputation_rates, xai_method_name)
    quality = explainer.quality_dict()

    if quality:
        status_logger.info(
            "Surrogate ready: type=%s, LOO-R²=%s, train-R²=%s, fidelity=%s, n_samples=%d",
            quality.get("selected_model_type"),
            quality.get("r_squared"),
            quality.get("r_squared_train"),
            quality.get("fidelity_tier"),
            quality.get("n_samples", 0),
        )

    status_logger.info("Computing global explanation...")
    global_exp = explainer.explain_global(X, top_k=top_k)
    persist_global_entry(session.session, prediction, xai_method_name, global_exp, quality, commit=False)

    status_logger.info("Computing %d local explanations (method=%s)...", len(forecasts), xai_method_name)
    for idx, fc in enumerate(forecasts):
        actual_value = forecast_actual_value(fc.values, output_statistic)
        feature_actual_values = {name: float(X[idx, i]) for i, name in enumerate(feature_names)}

        local_exp = explainer.explain_local(
            X=X,
            instance_idx=idx,
            prediction_id=prediction_id,
            org_unit=fc.org_unit,
            period=fc.period,
            feature_actual_values=feature_actual_values,
            top_k=top_k,
            output_statistic=output_statistic,
            actual_forecast_value=actual_value,
        )

        explanation = PredictionExplanation(
            prediction_id=prediction_id,
            org_unit=fc.org_unit,
            period=fc.period,
            method=xai_method_name,
            output_statistic=output_statistic,
            params={"top_k": top_k},
            result={
                "feature_attributions": [f.model_dump() for f in local_exp.feature_attributions],
                "baseline_prediction": local_exp.baseline_prediction,
                "actual_prediction": local_exp.actual_prediction,
                "xai_method_name": xai_method_name,
                "surrogate_quality": quality_response_dict(quality),
                "covariate_provenance": covariate_provenance_rows[idx],
            },
            status="completed",
        )
        session.session.add(explanation)

    session.session.commit()
    status_logger.info(
        "XAI explanations complete: %d explanations saved (prediction=%d)",
        len(forecasts),
        prediction_id,
    )
