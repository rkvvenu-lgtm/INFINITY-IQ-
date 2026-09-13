"""
SIH26170 - Generic / Universal Screening Subsystem

Purpose:
    Domain-agnostic statistical screening for ANY uploaded dataset.

    The electronics path (Iddq / Leakage / Delay) remains locked and
    unchanged. This package provides a fully separate, feature-agnostic
    pipeline that auto-discovers numeric parameters from any dataframe
    and produces statistically defensible screening results.

Modules:
    generic_feature_discovery - column classification and readiness
    generic_anomaly          - robust z-score + Isolation Forest
    generic_drift            - time/phase aware drift detection
    generic_rules            - adaptive risk fusion and decision table
    generic_explainability   - per-record WHY explanations
    generic_engine           - end-to-end orchestration
    self_test                - synthetic correctness validation

Design rule:
    This subsystem never modifies the Phase 1 electronics engine.
"""

from modules.generic_screening.generic_engine import (
    GenericScreeningEngine,
)

__all__ = [
    "GenericScreeningEngine",
]

__version__ = "1.0"