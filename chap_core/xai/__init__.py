"""
Explainable AI (XAI) module for CHAP.

Provides model-agnostic explainability for probabilistic health forecasts.
"""

from .types import (
    FeatureAttribution,
    GlobalExplanation,
    LocalExplanation,
    ExplanationMethod,
)
from .explainer import PerturbationExplainer

__all__ = [
    "FeatureAttribution",
    "GlobalExplanation",
    "LocalExplanation",
    "ExplanationMethod",
    "PerturbationExplainer",
]
