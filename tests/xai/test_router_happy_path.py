"""Smoke tests for the /v1/xai router."""

from __future__ import annotations

import datetime

import pytest
from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel
from starlette.testclient import TestClient

from chap_core.database.tables import Prediction
from chap_core.rest_api.app import app
from chap_core.rest_api.v1.routers.dependencies import get_session


@pytest.fixture
def engine(tmp_path):
    db_path = tmp_path / "xai_router_test.sqlite"
    eng = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(eng)
    return eng


@pytest.fixture
def client(engine):
    def _override():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = _override
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def prediction_with_cached_global(engine):
    cached_entry = {
        "topFeatures": [
            {"feature_name": "rainfall", "importance": 0.42, "direction": "positive"},
        ],
        "computedAt": datetime.datetime.now(datetime.UTC).isoformat(),
        "nSamples": 12,
        "stabilityScore": 0.9,
        "surrogateQuality": None,
    }
    with Session(engine) as session:
        prediction = Prediction(
            model_id="naive_model",
            model_db_id=1,
            dataset_id=1,
            n_periods=3,
            name="cached prediction",
            created=datetime.datetime.now(),
            meta_data={"xai": {"global_by_method": {"shap_auto": cached_entry}}},
        )
        session.add(prediction)
        session.commit()
        session.refresh(prediction)
        return prediction.id


def test_list_methods_returns_non_empty(client):
    resp = client.get("/v1/xai/methods")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body, list)
    assert len(body) > 0
    names = {m["name"] for m in body}
    assert "shap_auto" in names
    assert "permutation_importance" not in names  # archived by default


def test_global_explanation_returns_cached_entry(client, prediction_with_cached_global):
    prediction_id = prediction_with_cached_global
    resp = client.get(f"/v1/xai/predictions/{prediction_id}/global", params={"xaiMethod": "shap_auto"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["available"] is True
    assert body["method"] == "shap_auto"
    assert body["nSamples"] == 12
    assert body["stabilityScore"] == 0.9
    assert body["topFeatures"][0]["feature_name"] == "rainfall"
