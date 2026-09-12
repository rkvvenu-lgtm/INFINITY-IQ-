"""
SIH26170 - Universal Screening Engine
Phase 2, File 4

Purpose:
    Application-level screening layer above the locked Phase 1 AI core.

Responsibilities:
    - Automatically detect the engineering domain.
    - Select the appropriate domain pipeline.
    - Accept already-generated Phase 1 AI results.
    - Apply domain-level screening interpretation.
    - Produce universal screening metadata.
    - Handle unknown domains through manual review.
    - Keep Phase 1 core files unchanged.

Architecture:

    Input Dataset
          |
          v
    Domain Detector
          |
          v
    Domain Framework
          |
          v
    Domain Pipeline
          |
          v
    Phase 1 AI Result
          |
          v
    Universal Screening Result
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from modules.domain_detector import DomainDetector
from modules.domain_framework import DomainFramework
from modules.electronics_pipeline import ElectronicsPipeline


class UniversalScreeningEngine:
    """
    Universal application-level screening engine.

    This layer sits above Phase 1 and does not modify the Phase 1 AI engine.
    """

    VERSION = "3.0"

    def __init__(
        self,
        domain_framework: Optional[DomainFramework] = None,
        domain_detector: Optional[DomainDetector] = None,
        electronics_pipeline: Optional[ElectronicsPipeline] = None,
    ) -> None:

        self.domain_framework = (
            domain_framework or DomainFramework()
        )

        self.domain_detector = (
            domain_detector or DomainDetector(
                framework=self.domain_framework
            )
        )

        self.electronics_pipeline = (
            electronics_pipeline or ElectronicsPipeline()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_engine_info(self) -> Dict[str, Any]:
        """Return engine metadata and supported domains."""

        return {
            "engine_version": self.VERSION,
            "supported_domains": list(
                self.domain_framework.SUPPORTED_DOMAINS
            ),
            "automatic_domain_detection": True,
            "phase": "PHASE_2",
            "phase_1_core_modified": False,
        }

    def detect_domain(
        self,
        data: Optional[pd.DataFrame] = None,
        metadata: Optional[Dict[str, Any]] = None,
        text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Detect the domain from dataset, metadata and optional text.
        """

        return self.domain_detector.detect(
            data=data,
            metadata=metadata,
            text=text,
        )

    def prepare_domain_context(
        self,
        data: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
        text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Detect the domain and prepare the appropriate domain context.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        detection = self.detect_domain(
            data=data,
            metadata=metadata,
            text=text,
        )

        domain = detection["domain"]

        context: Dict[str, Any] = {
            "engine_version": self.VERSION,
            "domain_detection": detection,
            "domain": domain,
            "domain_display_name": detection["display_name"],
            "domain_pipeline_available": False,
            "domain_context": None,
        }

        if domain == "electronics":
            context["domain_pipeline_available"] = True
            context["domain_context"] = (
                self.electronics_pipeline.prepare_screening_context(data)
            )

        return context

    def screen(
        self,
        data: pd.DataFrame,
        phase1_result: Optional[pd.DataFrame] = None,
        mode: str = "LIVE_FUTURE_SCREENING",
        metadata: Optional[Dict[str, Any]] = None,
        text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform universal application-level screening.

        Parameters
        ----------
        data:
            Input dataset used for domain detection.

        phase1_result:
            Optional DataFrame produced by the locked Phase 1 engine.

        mode:
            TRAINING_EVALUATION or LIVE_FUTURE_SCREENING.

        metadata:
            Optional dataset metadata.

        text:
            Optional domain-description text.

        Returns
        -------
        dict
            Universal screening result and application metadata.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        normalized_mode = self._normalize_mode(mode)

        context = self.prepare_domain_context(
            data=data,
            metadata=metadata,
            text=text,
        )

        detection = context["domain_detection"]
        domain = detection["domain"]

        if detection["requires_manual_review"]:
            return self._manual_review_result(
                data=data,
                phase1_result=phase1_result,
                mode=normalized_mode,
                context=context,
                reason=(
                    "Automatic domain detection is not sufficiently "
                    "confident for automatic engineering screening."
                ),
            )

        if domain == "electronics":
            return self._screen_electronics(
                data=data,
                phase1_result=phase1_result,
                mode=normalized_mode,
                context=context,
            )

        return self._manual_review_result(
            data=data,
            phase1_result=phase1_result,
            mode=normalized_mode,
            context=context,
            reason=(
                f"No specialized screening adapter is currently registered "
                f"for domain '{domain}'."
            ),
        )

    # ------------------------------------------------------------------
    # Electronics screening
    # ------------------------------------------------------------------

    def _screen_electronics(
        self,
        data: pd.DataFrame,
        phase1_result: Optional[pd.DataFrame],
        mode: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Interpret electronics-domain Phase 1 results.
        """

        result = (
            phase1_result.copy()
            if isinstance(phase1_result, pd.DataFrame)
            else data.copy()
        )

        parameters = self.electronics_pipeline.get_parameters()

        parameter_summary: Dict[str, Any] = {}

        for parameter in parameters:
            profile = self.electronics_pipeline.get_parameter_profile(
                parameter
            )

            parameter_summary[parameter] = {
                "display_name": profile["display_name"],
                "unit": profile["unit"],
                "direction": profile["direction"],
                "engineering_limit": profile[
                    "engineering_limit"
                ],
                "prediction_column": profile[
                    "prediction_column"
                ],
                "current_columns": profile[
                    "early_features"
                ],
                "future_target_column": profile[
                    "target_column"
                ],
            }

        screening_summary = self._summarize_phase1_result(
            result=result,
            parameters=parameters,
        )

        return {
            "engine_version": self.VERSION,
            "screening_status": "SCREENED",
            "decision": self._derive_decision(
                result
            ),
            "mode": mode,
            "domain": "electronics",
            "domain_display_name": "Electronics",
            "domain_detection": context["domain_detection"],
            "domain_pipeline_version": (
                self.electronics_pipeline.VERSION
            ),
            "parameters": parameter_summary,
            "summary": screening_summary,
            "result": result,
            "manual_review_required": False,
            "phase_1_core_modified": False,
        }

    # ------------------------------------------------------------------
    # Result interpretation
    # ------------------------------------------------------------------

    @staticmethod
    def _summarize_phase1_result(
        result: pd.DataFrame,
        parameters: list[str],
    ) -> Dict[str, Any]:
        """
        Summarize important Phase 1 screening columns.
        """

        summary: Dict[str, Any] = {
            "rows": int(len(result)),
            "anomaly_count": 0,
            "high_risk_count": 0,
            "critical_risk_count": 0,
            "review_count": 0,
            "reject_count": 0,
            "prediction_columns_available": [],
        }

        if "Anomaly_Flag" in result.columns:
            summary["anomaly_count"] = int(
                pd.to_numeric(
                    result["Anomaly_Flag"],
                    errors="coerce",
                )
                .fillna(0)
                .astype(bool)
                .sum()
            )

        if "Risk_Level" in result.columns:
            risk_values = (
                result["Risk_Level"]
                .astype(str)
                .str.upper()
            )

            summary["high_risk_count"] = int(
                (risk_values == "HIGH").sum()
            )

            summary["critical_risk_count"] = int(
                (risk_values == "CRITICAL").sum()
            )

        if "Risk_Decision" in result.columns:
            decisions = (
                result["Risk_Decision"]
                .astype(str)
                .str.upper()
            )

            summary["review_count"] = int(
                (decisions == "REVIEW").sum()
            )

            summary["reject_count"] = int(
                (decisions == "REJECT").sum()
            )

        for parameter in parameters:
            prediction_column = (
                f"Predicted_{parameter}_168h"
            )

            if prediction_column in result.columns:
                summary[
                    "prediction_columns_available"
                ].append(prediction_column)

        return summary

    @staticmethod
    def _derive_decision(result: pd.DataFrame) -> str:
        """
        Derive a dataset-level application decision from Phase 1 results.

        Priority:
            REJECT > REVIEW > ACCEPT
        """

        if "Risk_Decision" not in result.columns:
            return "REVIEW"

        decisions = (
            result["Risk_Decision"]
            .astype(str)
            .str.upper()
        )

        if (decisions == "REJECT").any():
            return "REJECT"

        if (decisions == "REVIEW").any():
            return "REVIEW"

        if (decisions == "ACCEPT").all():
            return "ACCEPT"

        return "REVIEW"

    # ------------------------------------------------------------------
    # Manual-review handling
    # ------------------------------------------------------------------

    def _manual_review_result(
        self,
        data: pd.DataFrame,
        phase1_result: Optional[pd.DataFrame],
        mode: str,
        context: Dict[str, Any],
        reason: str,
    ) -> Dict[str, Any]:
        """
        Return a safe manual-review result when automatic screening
        should not be performed.
        """

        result = (
            phase1_result.copy()
            if isinstance(phase1_result, pd.DataFrame)
            else data.copy()
        )

        return {
            "engine_version": self.VERSION,
            "screening_status": "MANUAL_REVIEW",
            "decision": "REVIEW",
            "mode": mode,
            "domain": context["domain"],
            "domain_display_name": context[
                "domain_display_name"
            ],
            "domain_detection": context[
                "domain_detection"
            ],
            "domain_pipeline_version": None,
            "parameters": {},
            "summary": {
                "rows": int(len(result)),
                "anomaly_count": 0,
                "high_risk_count": 0,
                "critical_risk_count": 0,
                "review_count": int(len(result)),
                "reject_count": 0,
                "prediction_columns_available": [],
            },
            "result": result,
            "manual_review_required": True,
            "review_reason": reason,
            "phase_1_core_modified": False,
        }

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        """
        Validate and normalize operating mode.
        """

        if not isinstance(mode, str):
            raise TypeError("mode must be a string")

        normalized = mode.strip().upper()

        allowed_modes = {
            "TRAINING_EVALUATION",
            "LIVE_FUTURE_SCREENING",
        }

        if normalized not in allowed_modes:
            raise ValueError(
                "Unsupported mode. Use "
                "'TRAINING_EVALUATION' or "
                "'LIVE_FUTURE_SCREENING'."
            )

        return normalized


if __name__ == "__main__":
    engine = UniversalScreeningEngine()

    print("UniversalScreeningEngine : PASS")
    print("Version:", engine.VERSION)
    print(
        "Domains:",
        engine.get_engine_info()["supported_domains"],
    )
    print(
        "Automatic Detection:",
        engine.get_engine_info()[
            "automatic_domain_detection"
        ],
    )