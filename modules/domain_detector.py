"""
SIH26170 - Automatic Domain Detection
Version: 3.0

Purpose:
    Automatically identify the most likely engineering domain
    from dataset column names and optional dataset metadata.

Output:
    - detected domain
    - confidence score
    - evidence
    - candidate scores

This module is part of Phase 2 Application Intelligence.
It does not modify or depend on the Phase-1 core pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from modules.domain_framework import DomainFramework


class DomainDetector:
    """Automatic engineering-domain detector."""

    VERSION = "3.0"

    def __init__(
        self,
        framework: Optional[DomainFramework] = None
    ):

        self.framework = (
            framework
            if framework is not None
            else DomainFramework()
        )

    # =============================================================
    # MAIN DETECTION
    # =============================================================

    def detect(
        self,
        data: Optional[pd.DataFrame] = None,
        metadata: Optional[Dict[str, Any]] = None,
        text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect the most likely domain.

        Evidence can come from:
            1. Dataset column names
            2. Dataset values
            3. Metadata
            4. Optional user-provided text
        """

        scores = {
            domain: 0.0
            for domain in self.framework.supported_domains
            if domain != "general_unknown"
        }

        evidence: Dict[str, List[str]] = {
            domain: []
            for domain in scores
        }

        # ---------------------------------------------------------
        # Column-name evidence
        # ---------------------------------------------------------

        if isinstance(data, pd.DataFrame):

            for column in data.columns:

                column_text = str(
                    column
                ).strip().lower()

                self._score_text(
                    column_text,
                    scores,
                    evidence,
                    source=f"column:{column}"
                )

        # ---------------------------------------------------------
        # Metadata evidence
        # ---------------------------------------------------------

        if isinstance(metadata, dict):

            for key, value in metadata.items():

                text_value = (
                    f"{key} {value}"
                ).strip().lower()

                self._score_text(
                    text_value,
                    scores,
                    evidence,
                    source=f"metadata:{key}"
                )

        # ---------------------------------------------------------
        # Optional free-text evidence
        # ---------------------------------------------------------

        if text:

            self._score_text(
                str(text).lower(),
                scores,
                evidence,
                source="user_text"
            )

        # ---------------------------------------------------------
        # Dataset-value evidence
        # ---------------------------------------------------------

        if isinstance(data, pd.DataFrame):

            value_text = self._extract_value_text(
                data
            )

            if value_text:

                self._score_text(
                    value_text,
                    scores,
                    evidence,
                    source="dataset_values"
                )

        # ---------------------------------------------------------
        # Select domain
        # ---------------------------------------------------------

        if not scores:

            return self._unknown_result()

        best_domain = max(
            scores,
            key=scores.get
        )

        best_score = scores[
            best_domain
        ]

        total_score = sum(
            scores.values()
        )

        if best_score <= 0:

            return self._unknown_result(
                scores=scores,
                evidence=evidence
            )

        # Confidence is based on dominance of the best candidate.
        confidence = (
            best_score / total_score
            if total_score > 0
            else 0.0
        )

        # Evidence count gives additional confidence.
        evidence_count = len(
            evidence.get(
                best_domain,
                []
            )
        )

        if evidence_count >= 5:
            confidence += 0.10

        elif evidence_count >= 3:
            confidence += 0.05

        confidence = min(
            1.0,
            confidence
        )

        # Avoid automatic certainty for weak evidence.
        if best_score < 2:

            detected_domain = (
                self.framework.DEFAULT_DOMAIN
            )

        else:

            detected_domain = (
                self.framework.resolve(
                    best_domain
                )
            )

        return {
            "detector_version":
                self.VERSION,

            "domain":
                detected_domain,

            "display_name":
                self.framework.get_display_name(
                    detected_domain
                ),

            "confidence":
                round(
                    confidence,
                    4
                ),

            "confidence_percentage":
                round(
                    confidence * 100.0,
                    2
                ),

            "evidence":
                evidence.get(
                    best_domain,
                    []
                ),

            "candidate_scores":
                {
                    key: round(
                        value,
                        4
                    )
                    for key, value in scores.items()
                },

            "automatic_detection":
                True,

            "requires_manual_review":
                (
                    detected_domain
                    == self.framework.DEFAULT_DOMAIN
                    or confidence < 0.60
                )
        }

    # =============================================================
    # TEXT SCORING
    # =============================================================

    def _score_text(
        self,
        text: str,
        scores: Dict[str, float],
        evidence: Dict[str, List[str]],
        source: str
    ) -> None:
        """Score domains using framework-supported keywords."""

        if not text:
            return

        normalized = (
            text
            .lower()
            .replace("-", " ")
            .replace("_", " ")
        )

        keyword_groups = {

            "electronics": [
                "electronics",
                "electronic",
                "semiconductor",
                "semiconductors",
                "vlsi",
                "asic",
                "fpga",
                "chip",
                "integrated circuit",
                "iddq",
                "leakage",
                "propagation delay",
                "delay",
                "uA",
                "ns",
            ],

            "mechanical": [
                "mechanical",
                "mechanics",
                "machine",
                "machinery",
                "dimension",
                "bearing",
                "shaft",
                "vibration",
                "torque",
                "pressure",
            ],

            "automotive": [
                "automotive",
                "automobile",
                "vehicle",
                "car",
                "engine",
                "ev",
                "brake",
                "battery management",
            ],

            "manufacturing": [
                "manufacturing",
                "manufacture",
                "production",
                "factory",
                "process",
                "assembly",
                "quality control",
                "yield",
                "defect",
            ],

            "energy": [
                "energy",
                "power",
                "battery",
                "solar",
                "voltage",
                "current",
                "energy storage",
                "inverter",
                "efficiency",
            ],

            "aerospace": [
                "aerospace",
                "aviation",
                "aircraft",
                "spacecraft",
                "satellite",
                "rocket",
                "space",
            ],

            "medical_equipment": [
                "medical",
                "healthcare",
                "hospital",
                "medical equipment",
                "diagnostic equipment",
                "patient monitor",
                "imaging",
            ],
        }

        for domain, keywords in keyword_groups.items():

            if domain not in scores:
                continue

            for keyword in keywords:

                if keyword.lower() in normalized:

                    scores[domain] += 1.0

                    evidence_item = (
                        f"{source} matched '{keyword}'"
                    )

                    if (
                        evidence_item
                        not in evidence[domain]
                    ):

                        evidence[domain].append(
                            evidence_item
                        )

    # =============================================================
    # VALUE EXTRACTION
    # =============================================================

    @staticmethod
    def _extract_value_text(
        data: pd.DataFrame,
        max_columns: int = 20,
        max_values_per_column: int = 5
    ) -> str:
        """
        Extract a small deterministic text sample from
        categorical/object columns.

        Numeric values are intentionally ignored.
        """

        parts: List[str] = []

        object_columns = (
            data.select_dtypes(
                include=[
                    "object",
                    "category"
                ]
            )
            .columns
            .tolist()
        )

        for column in object_columns[
            :max_columns
        ]:

            values = (
                data[column]
                .dropna()
                .astype(str)
                .head(
                    max_values_per_column
                )
                .tolist()
            )

            parts.extend(
                values
            )

        return " ".join(
            parts
        ).lower()

    # =============================================================
    # UNKNOWN RESULT
    # =============================================================

    def _unknown_result(
        self,
        scores: Optional[Dict[str, float]] = None,
        evidence: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """Return safe unknown-domain result."""

        if scores is None:

            scores = {
                domain: 0.0
                for domain in self.framework.supported_domains
                if domain != "general_unknown"
            }

        if evidence is None:

            evidence = {
                domain: []
                for domain in scores
            }

        return {
            "detector_version":
                self.VERSION,

            "domain":
                self.framework.DEFAULT_DOMAIN,

            "display_name":
                self.framework.get_display_name(
                    self.framework.DEFAULT_DOMAIN
                ),

            "confidence":
                0.0,

            "confidence_percentage":
                0.0,

            "evidence":
                [],

            "candidate_scores":
                scores,

            "automatic_detection":
                True,

            "requires_manual_review":
                True
        }

    # =============================================================
    # SUMMARY
    # =============================================================

    def get_detector_info(
        self
    ) -> Dict[str, Any]:
        """Return detector information."""

        return {
            "version":
                self.VERSION,

            "supported_domains":
                list(
                    self.framework.supported_domains
                ),

            "default_domain":
                self.framework.DEFAULT_DOMAIN,

            "automatic_detection":
                True
        }