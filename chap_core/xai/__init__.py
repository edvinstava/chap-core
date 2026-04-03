"""
Explainable AI (XAI) module for CHAP.

Provides model-agnostic explainability for probabilistic health forecasts.
"""

from .explainer import PerturbationExplainer
from .types import (
    ExplanationMethod,
    FeatureAttribution,
    GlobalExplanation,
    LocalExplanation,
)

__all__ = [
    "FeatureAttribution",
    "GlobalExplanation",
    "LocalExplanation",
    "ExplanationMethod",
    "PerturbationExplainer",
]
