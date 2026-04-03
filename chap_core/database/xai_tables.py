"""
Database tables for XAI explanations.
"""

import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, Column
from sqlmodel import Field, Relationship

from chap_core.database.base_tables import DBModel
from chap_core.database.tables import Prediction


class PredictionExplanationBase(DBModel):
    prediction_id: int = Field(foreign_key="prediction.id")
    org_unit: str
    period: str
    method: str
    output_statistic: str = "median"
    params: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    result: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    status: str = "completed"
    error: Optional[str] = None
    created: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)


class PredictionExplanation(PredictionExplanationBase, table=True):
    id: Optional[int] = Field(primary_key=True, default=None)
    prediction: Prediction = Relationship()


class PredictionExplanationRead(PredictionExplanationBase):
    id: int
