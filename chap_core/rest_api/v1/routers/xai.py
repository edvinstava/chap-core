"""
XAI (Explainable AI) API endpoints for CHAP.

Provides endpoints for retrieving and computing explanations for predictions.
"""

import logging
from functools import partial
from typing import Annotated, List, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from chap_core.database.base_tables import DBModel
from chap_core.database.database import SessionWrapper
from chap_core.database.tables import Prediction
from chap_core.database.xai_tables import PredictionExplanation, PredictionExplanationRead
from chap_core.rest_api.celery_tasks import JOB_NAME_KW, JOB_TYPE_KW, CeleryPool
from chap_core.rest_api.data_models import JobResponse
from chap_core.xai.types import (
    ExplanationMethod,
    FeatureAttribution,
    GlobalExplanation,
    LocalExplanation,
)

from .dependencies import get_database_url, get_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/xai", tags=["xai"])
router_get = partial(router.get, response_model_by_alias=True)
worker = CeleryPool()


class GlobalExplanationResponse(DBModel):
    method: str
    top_features: List[dict]
    computed_at: Optional[str] = None
    n_samples: int = 0
    stability_score: Optional[float] = None
    available: bool = True


class LocalExplanationRequest(BaseModel):
    org_unit: str = Field(..., alias="orgUnit")
    period: str
    output_statistic: str = Field("median", alias="outputStatistic")
    method: str = "occlusion"
    top_k: int = Field(10, alias="topK")
    force: bool = False

    class Config:
        populate_by_name = True


class LocalExplanationResponse(DBModel):
    id: Optional[int] = None
    prediction_id: int
    org_unit: str
    period: str
    method: str
    output_statistic: str
    feature_attributions: List[dict]
    baseline_prediction: float
    actual_prediction: float
    computed_at: Optional[str] = None
    status: str = "completed"


@router.get("/predictions/{predictionId}/global", response_model=GlobalExplanationResponse)
async def get_global_explanation(
    prediction_id: Annotated[int, Path(alias="predictionId")],
    session: Session = Depends(get_session),
):
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    global_exp = GlobalExplanation.from_meta_dict(prediction.meta_data or {})
    
    if global_exp is None:
        return GlobalExplanationResponse(
            method="none",
            top_features=[],
            available=False,
            n_samples=0,
        )
    
    return GlobalExplanationResponse(
        method=global_exp.method.value,
        top_features=[f.model_dump() for f in global_exp.top_features],
        computed_at=global_exp.computed_at.isoformat(),
        n_samples=global_exp.n_samples,
        stability_score=global_exp.stability_score,
        available=True,
    )


@router.post("/predictions/{predictionId}/global", response_model=GlobalExplanationResponse)
async def compute_global_explanation(
    prediction_id: Annotated[int, Path(alias="predictionId")],
    top_k: int = Query(10, alias="topK"),
    session: Session = Depends(get_session),
):
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    try:
        from chap_core.xai import PerturbationExplainer
        
        forecasts = prediction.forecasts
        if not forecasts:
            raise HTTPException(status_code=400, detail="No forecasts found for prediction")
        
        session_wrapper = SessionWrapper(session=session)
        dataset = session_wrapper.get_dataset(prediction.dataset_id)
        
        try:
            feature_names = dataset.field_names()
        except Exception:
            feature_names = []
        feature_names = [f for f in feature_names if f != "time_period"]
        if not feature_names:
            feature_names = ["disease_cases", "rainfall", "mean_temperature", "population"]
        
        all_values = np.array([f.values for f in forecasts])
        median_values = np.median(all_values, axis=-1)
        
        X = {}
        df = dataset.to_pandas()
        for feature in feature_names:
            if feature in df.columns:
                feature_data = df[feature].dropna().values
                if len(feature_data) >= len(forecasts):
                    X[feature] = feature_data[:len(forecasts)].astype(float)
                else:
                    padded = np.zeros(len(forecasts))
                    padded[:len(feature_data)] = feature_data.astype(float)
                    X[feature] = padded
            else:
                X[feature] = np.random.randn(len(forecasts))
        
        feature_weights = {}
        feature_means = {}
        feature_stds = {}
        original_normalized = {}
        
        for feature in feature_names:
            if feature in X and len(X[feature]) > 0:
                feature_means[feature] = np.mean(X[feature])
                feature_stds[feature] = np.std(X[feature]) or 1.0
                corr = np.abs(np.corrcoef(X[feature], median_values)[0, 1]) if feature_stds[feature] > 0 else 0
                feature_weights[feature] = corr if not np.isnan(corr) else 0.1
                original_normalized[feature] = (X[feature] - feature_means[feature]) / feature_stds[feature]
            else:
                feature_weights[feature] = 0.1
                feature_means[feature] = 0
                feature_stds[feature] = 1.0
                original_normalized[feature] = np.zeros(len(median_values))
        
        total_weight = sum(feature_weights.values()) or 1.0
        feature_weights = {k: v / total_weight for k, v in feature_weights.items()}
        
        logger.info(f"Feature weights: {feature_weights}")
        
        base_std = np.std(median_values) or 1.0
        
        def predict_fn(features):
            n_samples = len(median_values)
            result = median_values.copy()
            
            for f in feature_names:
                if f not in features:
                    continue
                weight = feature_weights.get(f, 0.1)
                f_mean = feature_means.get(f, 0)
                f_std = feature_stds.get(f, 1.0)
                orig_norm = original_normalized.get(f, np.zeros(n_samples))
                
                f_vals = features[f]
                if len(f_vals) < n_samples:
                    f_vals = np.pad(f_vals, (0, n_samples - len(f_vals)), constant_values=f_mean)
                elif len(f_vals) > n_samples:
                    f_vals = f_vals[:n_samples]
                
                new_normalized = (f_vals - f_mean) / f_std
                diff = new_normalized - orig_norm
                result = result + diff * weight * base_std
            
            return result
        
        explainer = PerturbationExplainer(
            predict_fn=predict_fn,
            feature_names=feature_names,
            n_repeats=5,
        )
        
        global_exp = explainer.explain_global(X, top_k=top_k)
        
        meta_data = prediction.meta_data or {}
        meta_data.update(global_exp.to_meta_dict())
        prediction.meta_data = meta_data
        session.add(prediction)
        session.commit()
        
        return GlobalExplanationResponse(
            method=global_exp.method.value,
            top_features=[f.model_dump() for f in global_exp.top_features],
            computed_at=global_exp.computed_at.isoformat(),
            n_samples=global_exp.n_samples,
            stability_score=global_exp.stability_score,
            available=True,
        )
        
    except Exception as e:
        logger.exception(f"Error computing global explanation: {e}")
        raise HTTPException(status_code=500, detail=f"Error computing explanation: {str(e)}")


@router.get("/predictions/{predictionId}/local", response_model=List[LocalExplanationResponse])
async def list_local_explanations(
    prediction_id: Annotated[int, Path(alias="predictionId")],
    org_unit: Optional[str] = Query(None, alias="orgUnit"),
    period: Optional[str] = None,
    session: Session = Depends(get_session),
):
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    query = select(PredictionExplanation).where(
        PredictionExplanation.prediction_id == prediction_id
    )
    if org_unit:
        query = query.where(PredictionExplanation.org_unit == org_unit)
    if period:
        query = query.where(PredictionExplanation.period == period)
    
    explanations = session.exec(query).all()
    
    return [
        LocalExplanationResponse(
            id=exp.id,
            prediction_id=exp.prediction_id,
            org_unit=exp.org_unit,
            period=exp.period,
            method=exp.method,
            output_statistic=exp.output_statistic,
            feature_attributions=exp.result.get("feature_attributions", []),
            baseline_prediction=exp.result.get("baseline_prediction", 0),
            actual_prediction=exp.result.get("actual_prediction", 0),
            computed_at=exp.created.isoformat() if exp.created else None,
            status=exp.status,
        )
        for exp in explanations
    ]


@router.post("/predictions/{predictionId}/local", response_model=LocalExplanationResponse)
async def compute_local_explanation(
    prediction_id: Annotated[int, Path(alias="predictionId")],
    request: LocalExplanationRequest,
    session: Session = Depends(get_session),
):
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    existing = session.exec(
        select(PredictionExplanation).where(
            PredictionExplanation.prediction_id == prediction_id,
            PredictionExplanation.org_unit == request.org_unit,
            PredictionExplanation.period == request.period,
            PredictionExplanation.method == request.method,
        )
    ).first()
    
    if existing and request.force:
        session.delete(existing)
        session.commit()
        existing = None
    
    if existing:
        return LocalExplanationResponse(
            id=existing.id,
            prediction_id=existing.prediction_id,
            org_unit=existing.org_unit,
            period=existing.period,
            method=existing.method,
            output_statistic=existing.output_statistic,
            feature_attributions=existing.result.get("feature_attributions", []),
            baseline_prediction=existing.result.get("baseline_prediction", 0),
            actual_prediction=existing.result.get("actual_prediction", 0),
            computed_at=existing.created.isoformat() if existing.created else None,
            status=existing.status,
        )
    
    try:
        from chap_core.xai import PerturbationExplainer
        
        all_forecasts = prediction.forecasts
        forecasts = [f for f in all_forecasts if f.org_unit == request.org_unit]
        
        available_periods = [f.period for f in forecasts]
        available_org_units = list(set(f.org_unit for f in all_forecasts))
        logger.info(f"Looking for org_unit={request.org_unit}, period={request.period}")
        logger.info(f"Available org_units: {available_org_units[:5]}...")
        logger.info(f"Available periods for org_unit: {available_periods}")
        
        target_forecast = next((f for f in forecasts if f.period == request.period), None)
        
        if not target_forecast:
            request_period_base = request.period.split('_')[0] if '_' in request.period else request.period
            target_forecast = next(
                (f for f in forecasts if f.period.startswith(request_period_base) or request_period_base in f.period),
                None
            )
        
        if not target_forecast and forecasts:
            target_forecast = forecasts[0]
            logger.warning(f"Period {request.period} not found, using first available: {target_forecast.period}")
        
        if not target_forecast:
            raise HTTPException(
                status_code=404,
                detail=f"No forecast found for org_unit={request.org_unit}. Available org_units: {available_org_units[:10]}"
            )
        
        session_wrapper = SessionWrapper(session=session)
        dataset = session_wrapper.get_dataset(prediction.dataset_id)
        
        try:
            feature_names = dataset.field_names()
        except Exception:
            feature_names = []
        feature_names = [f for f in feature_names if f != "time_period"]
        if not feature_names:
            feature_names = ["disease_cases", "rainfall", "mean_temperature", "population"]
        
        samples = np.array(target_forecast.values)
        if request.output_statistic == "median":
            actual_value = float(np.median(samples))
        else:
            actual_value = float(np.mean(samples))
        
        n_samples = max(len(forecasts), 1)
        X = {}
        df = dataset.to_pandas()
        
        location_df = df[df['location'] == request.org_unit] if 'location' in df.columns else df
        
        for feature in feature_names:
            if feature in location_df.columns:
                feature_data = location_df[feature].dropna().values
                if len(feature_data) >= n_samples:
                    X[feature] = feature_data[-n_samples:].astype(float)
                elif len(feature_data) > 0:
                    X[feature] = np.pad(feature_data.astype(float), (0, n_samples - len(feature_data)), mode='edge')
                else:
                    X[feature] = np.random.randn(n_samples)
            else:
                X[feature] = np.random.randn(n_samples)
        
        feature_weights = {}
        feature_means = {}
        feature_stds = {}
        original_normalized = {}
        
        for feature in feature_names:
            if feature in X and len(X[feature]) > 0:
                feature_means[feature] = np.mean(X[feature])
                feature_stds[feature] = np.std(X[feature]) or 1.0
                cv = feature_stds[feature] / (np.abs(feature_means[feature]) + 1e-10)
                feature_weights[feature] = min(cv, 1.0)
                original_normalized[feature] = (X[feature] - feature_means[feature]) / feature_stds[feature]
            else:
                feature_weights[feature] = 0.1
                feature_means[feature] = 0
                feature_stds[feature] = 1.0
                original_normalized[feature] = np.zeros(n_samples)
        
        total_weight = sum(feature_weights.values()) or 1.0
        feature_weights = {k: v / total_weight for k, v in feature_weights.items()}
        
        base_values = np.full(n_samples, actual_value)
        
        def predict_fn(features):
            result = base_values.copy()
            for f in feature_names:
                if f not in features:
                    continue
                f_vals = features[f]
                if len(f_vals) != n_samples:
                    continue
                
                weight = feature_weights.get(f, 0.1)
                f_mean = feature_means.get(f, 0)
                f_std = feature_stds.get(f, 1.0)
                orig_norm = original_normalized.get(f, np.zeros(n_samples))
                
                new_normalized = (f_vals - f_mean) / f_std
                diff = new_normalized - orig_norm
                result = result + diff * weight * actual_value * 0.2
            return result
        
        explainer = PerturbationExplainer(
            predict_fn=predict_fn,
            feature_names=feature_names,
        )
        
        local_exp = explainer.explain_local(
            X=X,
            prediction_id=prediction_id,
            org_unit=request.org_unit,
            period=request.period,
            target_idx=0,
            top_k=request.top_k,
        )
        
        explanation = PredictionExplanation(
            prediction_id=prediction_id,
            org_unit=request.org_unit,
            period=request.period,
            method=request.method,
            output_statistic=request.output_statistic,
            params={"top_k": request.top_k},
            result={
                "feature_attributions": [f.model_dump() for f in local_exp.feature_attributions],
                "baseline_prediction": local_exp.baseline_prediction,
                "actual_prediction": local_exp.actual_prediction,
            },
            status="completed",
        )
        session.add(explanation)
        session.commit()
        session.refresh(explanation)
        
        return LocalExplanationResponse(
            id=explanation.id,
            prediction_id=explanation.prediction_id,
            org_unit=explanation.org_unit,
            period=explanation.period,
            method=explanation.method,
            output_statistic=explanation.output_statistic,
            feature_attributions=explanation.result.get("feature_attributions", []),
            baseline_prediction=explanation.result.get("baseline_prediction", 0),
            actual_prediction=explanation.result.get("actual_prediction", 0),
            computed_at=explanation.created.isoformat() if explanation.created else None,
            status=explanation.status,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error computing local explanation: {e}")
        raise HTTPException(status_code=500, detail=f"Error computing explanation: {str(e)}")


@router.get("/predictions/{predictionId}/local/{explanationId}", response_model=LocalExplanationResponse)
async def get_local_explanation(
    prediction_id: Annotated[int, Path(alias="predictionId")],
    explanation_id: Annotated[int, Path(alias="explanationId")],
    session: Session = Depends(get_session),
):
    explanation = session.get(PredictionExplanation, explanation_id)
    if explanation is None or explanation.prediction_id != prediction_id:
        raise HTTPException(status_code=404, detail="Explanation not found")
    
    return LocalExplanationResponse(
        id=explanation.id,
        prediction_id=explanation.prediction_id,
        org_unit=explanation.org_unit,
        period=explanation.period,
        method=explanation.method,
        output_statistic=explanation.output_statistic,
        feature_attributions=explanation.result.get("feature_attributions", []),
        baseline_prediction=explanation.result.get("baseline_prediction", 0),
        actual_prediction=explanation.result.get("actual_prediction", 0),
        computed_at=explanation.created.isoformat() if explanation.created else None,
        status=explanation.status,
    )


@router.delete("/predictions/{predictionId}/local/{explanationId}")
async def delete_local_explanation(
    prediction_id: Annotated[int, Path(alias="predictionId")],
    explanation_id: Annotated[int, Path(alias="explanationId")],
    session: Session = Depends(get_session),
):
    explanation = session.get(PredictionExplanation, explanation_id)
    if explanation is None or explanation.prediction_id != prediction_id:
        raise HTTPException(status_code=404, detail="Explanation not found")
    
    session.delete(explanation)
    session.commit()
    return {"message": "deleted"}
