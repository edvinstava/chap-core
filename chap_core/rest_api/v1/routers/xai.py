"""
XAI (Explainable AI) API endpoints for CHAP.

Provides endpoints for retrieving and computing explanations for predictions.
"""

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from sqlmodel import Session, select

from chap_core.database.tables import Prediction
from chap_core.database.xai_tables import PredictionExplanation
from chap_core.rest_api.celery_tasks import JOB_NAME_KW, JOB_TYPE_KW, CeleryPool
from chap_core.rest_api.data_models import JobResponse
from chap_core.rest_api.v1.xai_schemas import (
    GlobalExplanationResponse,
    HorizonSummaryResponse,
    LocalExplanationRequest,
    LocalExplanationResponse,
    RunExplanationsRequest,
    ShapBeeswarmResponse,
    XaiMethodRead,
)
from chap_core.xai.batch_explanations import run_explanations_task
from chap_core.xai.forecast_matching import find_forecast_row_index
from chap_core.xai.method_registry import NATIVE_SHAP, SHAP_AUTO
from chap_core.xai.method_registry import XAI_METHODS as XAI_METHOD_DEFINITIONS
from chap_core.xai.responses.native_shap import has_native_shap, list_filtered_native_shap_locals
from chap_core.xai.responses.stored_views import (
    beeswarm_from_stored,
    explanation_to_response,
    horizon_summary_from_stored,
)
from chap_core.xai.router_services import (
    compute_beeswarm_service,
    compute_global_explanation_service,
    compute_horizon_summary_service,
    compute_local_explanation_service,
    global_response_from_entry,
    load_global_entry,
    resolve_canonical_period,
    validate_xai_method_name,
)

from .dependencies import get_database_url, get_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/xai")
worker: CeleryPool[Any] = CeleryPool()

_XAI_METHODS = [XaiMethodRead(**definition) for definition in XAI_METHOD_DEFINITIONS]


@router.get("/methods", response_model=list[XaiMethodRead], response_model_by_alias=True, tags=["XAI"])
async def list_xai_methods(
    include_archived: bool = Query(False, alias="includeArchived"),
    prediction_id: int | None = Query(None, alias="predictionId"),
    session: Session = Depends(get_session),
):
    methods = _XAI_METHODS if include_archived else [m for m in _XAI_METHODS if not m.archived]

    if prediction_id is not None:
        prediction = session.get(Prediction, prediction_id)
        has_native = prediction is not None and has_native_shap(prediction)
        if not has_native:
            methods = [m for m in methods if m.name != NATIVE_SHAP]

    return methods


@router.get("/methods/{name}", response_model=XaiMethodRead, response_model_by_alias=True, tags=["XAI"])
async def get_xai_method(name: str):
    method = validate_xai_method_name(name, allow_archived=True)
    return XaiMethodRead(**method)


@router.post(
    "/predictions/{predictionId}/explanations/run",
    response_model=JobResponse,
    response_model_by_alias=True,
    tags=["XAI"],
)
async def run_explanations(
    prediction_id: Annotated[int, Path(alias="predictionId")],
    request: RunExplanationsRequest,
    database_url: str = Depends(get_database_url),
):
    validate_xai_method_name(request.xai_method_name)
    job = worker.queue_db(
        run_explanations_task,
        prediction_id,
        request.xai_method_name,
        request.output_statistic,
        request.top_k,
        database_url=database_url,
        **{JOB_TYPE_KW: "xai_explanations", JOB_NAME_KW: f"xai_{prediction_id}"},
    )
    return JobResponse(id=job.id)


@router.get(
    "/predictions/{predictionId}/global",
    response_model=GlobalExplanationResponse,
    response_model_by_alias=True,
    tags=["XAI"],
)
async def get_global_explanation(
    prediction_id: Annotated[int, Path(alias="predictionId")],
    xai_method: str | None = Query(None, alias="xaiMethod"),
    session: Session = Depends(get_session),
):
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    meta = prediction.meta_data or {}

    if xai_method:
        entry = load_global_entry(meta, xai_method)
        if entry:
            return global_response_from_entry(xai_method, entry)

    if xai_method is None:
        return GlobalExplanationResponse(method="none", top_features=[], available=False, n_samples=0)
    validate_xai_method_name(xai_method)
    return GlobalExplanationResponse(method=xai_method, top_features=[], available=False, n_samples=0)


@router.post(
    "/predictions/{predictionId}/global",
    response_model=GlobalExplanationResponse,
    response_model_by_alias=True,
    tags=["XAI"],
)
def compute_global_explanation(
    prediction_id: Annotated[int, Path(alias="predictionId")],
    top_k: int = Query(10, alias="topK"),
    xai_method: str = Query(SHAP_AUTO, alias="xaiMethod"),
    output_statistic: str = Query("median", alias="outputStatistic"),
    force: bool = Query(False),
    session: Session = Depends(get_session),
):
    validate_xai_method_name(xai_method)
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    if not force:
        entry = load_global_entry(prediction.meta_data, xai_method)
        if entry:
            return global_response_from_entry(xai_method, entry)

    try:
        return compute_global_explanation_service(
            session, prediction, prediction_id, xai_method, top_k, output_statistic
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error computing global explanation: %s", e)
        raise HTTPException(status_code=500, detail="Error computing explanation") from e


@router.get(
    "/predictions/{predictionId}/local",
    response_model=list[LocalExplanationResponse],
    response_model_by_alias=True,
    tags=["XAI"],
)
async def list_local_explanations(
    prediction_id: Annotated[int, Path(alias="predictionId")],
    org_unit: str | None = Query(None, alias="orgUnit"),
    period: str | None = None,
    xai_method: str | None = Query(None, alias="xaiMethod"),
    session: Session = Depends(get_session),
):
    if xai_method is not None:
        validate_xai_method_name(xai_method)
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    query = select(PredictionExplanation).where(PredictionExplanation.prediction_id == prediction_id)
    if org_unit:
        query = query.where(PredictionExplanation.org_unit == org_unit)
    if period:
        canonical_period = period
        if xai_method != NATIVE_SHAP and "_" in period and org_unit and prediction.forecasts:
            idx = find_forecast_row_index(prediction.forecasts, org_unit, period)
            if idx is not None:
                canonical_period = prediction.forecasts[idx].period
        query = query.where(PredictionExplanation.period == canonical_period)
    if xai_method:
        query = query.where(PredictionExplanation.method == xai_method)

    explanations = session.exec(query).all()

    if not explanations and xai_method == NATIVE_SHAP:
        native_items = list_filtered_native_shap_locals(
            prediction_id, prediction, org_unit, period, output_statistic="median"
        )
        if native_items:
            return native_items

    return [explanation_to_response(exp) for exp in explanations]


@router.post(
    "/predictions/{predictionId}/local",
    response_model=LocalExplanationResponse,
    response_model_by_alias=True,
    tags=["XAI"],
)
def compute_local_explanation(
    prediction_id: Annotated[int, Path(alias="predictionId")],
    request: LocalExplanationRequest,
    session: Session = Depends(get_session),
):
    validate_xai_method_name(request.xai_method)
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    all_forecasts = prediction.forecasts
    if not all_forecasts:
        raise HTTPException(status_code=400, detail="No forecasts found for prediction")

    instance_idx, canonical_period = resolve_canonical_period(all_forecasts, request.org_unit, request.period)

    existing = session.exec(
        select(PredictionExplanation).where(
            PredictionExplanation.prediction_id == prediction_id,
            PredictionExplanation.org_unit == request.org_unit,
            PredictionExplanation.period == canonical_period,
            PredictionExplanation.method == request.xai_method,
        )
    ).first()

    if existing and request.force:
        session.delete(existing)
        session.commit()
        existing = None

    if existing:
        return explanation_to_response(existing)

    if instance_idx is None:
        available = list({f.org_unit for f in all_forecasts})
        raise HTTPException(
            status_code=404,
            detail=f"No forecast found for org_unit={request.org_unit}. Available: {available[:10]}",
        )

    try:
        return compute_local_explanation_service(
            session, prediction, prediction_id, instance_idx, canonical_period, request
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error computing local explanation: %s", e)
        raise HTTPException(status_code=500, detail="Error computing local explanation") from e


@router.post(
    "/predictions/{predictionId}/shap-beeswarm",
    response_model=ShapBeeswarmResponse,
    response_model_by_alias=True,
    tags=["XAI"],
)
def compute_shap_beeswarm(
    prediction_id: Annotated[int, Path(alias="predictionId")],
    output_statistic: str = Query("median", alias="outputStatistic"),
    xai_method: str = Query(SHAP_AUTO, alias="xaiMethod"),
    session: Session = Depends(get_session),
):
    validate_xai_method_name(xai_method)
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    stored_query = select(PredictionExplanation).where(
        PredictionExplanation.prediction_id == prediction_id,
        PredictionExplanation.method == xai_method,
    )
    if xai_method != NATIVE_SHAP:
        stored_query = stored_query.where(PredictionExplanation.output_statistic == output_statistic)
    stored = session.exec(stored_query).all()
    if stored:
        return beeswarm_from_stored(prediction_id, output_statistic, stored)

    try:
        return compute_beeswarm_service(session, prediction, prediction_id, output_statistic, xai_method)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error computing SHAP beeswarm: %s", e)
        raise HTTPException(status_code=500, detail="Error computing beeswarm") from e


@router.post(
    "/predictions/{predictionId}/local/horizon-summary",
    response_model=HorizonSummaryResponse,
    response_model_by_alias=True,
    tags=["XAI"],
)
def compute_horizon_summary(
    prediction_id: Annotated[int, Path(alias="predictionId")],
    org_unit: str = Query(..., alias="orgUnit"),
    output_statistic: str = Query("median", alias="outputStatistic"),
    xai_method: str = Query(SHAP_AUTO, alias="xaiMethod"),
    session: Session = Depends(get_session),
):
    validate_xai_method_name(xai_method)
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    if not prediction.forecasts:
        raise HTTPException(status_code=400, detail="No forecasts found for prediction")

    stored_unit_query = select(PredictionExplanation).where(
        PredictionExplanation.prediction_id == prediction_id,
        PredictionExplanation.org_unit == org_unit,
        PredictionExplanation.method == xai_method,
    )
    if xai_method != NATIVE_SHAP:
        stored_unit_query = stored_unit_query.where(PredictionExplanation.output_statistic == output_statistic)
    stored_unit = session.exec(stored_unit_query).all()
    if stored_unit:
        return horizon_summary_from_stored(prediction_id, org_unit, xai_method, output_statistic, stored_unit)

    try:
        return compute_horizon_summary_service(
            session, prediction, prediction_id, org_unit, output_statistic, xai_method
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error computing horizon summary: %s", e)
        raise HTTPException(status_code=500, detail="Error computing horizon summary") from e


@router.get(
    "/predictions/{predictionId}/local/{explanationId}",
    response_model=LocalExplanationResponse,
    response_model_by_alias=True,
    tags=["XAI"],
)
async def get_local_explanation(
    prediction_id: Annotated[int, Path(alias="predictionId")],
    explanation_id: Annotated[int, Path(alias="explanationId")],
    session: Session = Depends(get_session),
):
    explanation = session.get(PredictionExplanation, explanation_id)
    if explanation is None or explanation.prediction_id != prediction_id:
        raise HTTPException(status_code=404, detail="Explanation not found")
    return explanation_to_response(explanation)


@router.delete("/predictions/{predictionId}/local/{explanationId}", tags=["XAI"])
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
