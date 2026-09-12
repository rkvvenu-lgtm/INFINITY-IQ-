"""
SIH26170 - Data Quality Intelligence
Phase 2, File 6

Purpose:
    Application-level data-quality intelligence layer.

Responsibilities:
    - Inspect dataset quality using the locked Phase 1 validator.
    - Present readiness, warnings and quality statistics.
    - Identify missing and invalid data.
    - Report required-column availability.
    - Provide dashboard/report-friendly output.

Design rule:
    This module does not modify Phase 1 validation or preprocessing files.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from modules.data_validator import DataValidator


class DataQualityIntelligence:
    """
    Application-level wrapper around the Phase 1 data validator.
    """

    VERSION = "3.0"

    def __init__(
        self,
        validator: Optional[DataValidator] = None,
    ) -> None:
        self.validator = validator or DataValidator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_engine_info(self) -> Dict[str, Any]:
        """Return data-quality intelligence metadata."""

        return {
            "version": self.VERSION,
            "capabilities": [
                "quality_status",
                "data_readiness",
                "missing_value_analysis",
                "numeric_validation",
                "required_column_check",
                "warning_summary",
                "quality_report",
            ],
            "phase": "PHASE_2",
            "phase_1_core_modified": False,
        }

    def analyze(
        self,
        data: pd.DataFrame,
        mode: str = "LIVE_FUTURE_SCREENING",
    ) -> Dict[str, Any]:
        """
        Analyze dataset quality.

        Parameters
        ----------
        data:
            Input dataset.

        mode:
            TRAINING_EVALUATION or LIVE_FUTURE_SCREENING.

        Returns
        -------
        dict
            Structured data-quality intelligence.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        normalized_mode = self._normalize_mode(mode)

        validation = self.validator.validate(data)

        readiness = self.validator.readiness(
            data,
            mode=normalized_mode,
        )

        quality_report = self.validator.quality_report(
            data
        )

        missing_analysis = self._missing_value_analysis(
            data
        )

        numeric_analysis = self._numeric_analysis(
            data
        )

        warning_summary = self._warning_summary(
            validation,
            quality_report,
        )

        return {
            "version": self.VERSION,
            "mode": normalized_mode,
            "rows": int(len(data)),
            "columns": int(len(data.columns)),
            "quality_status": self._quality_status(
                validation
            ),
            "data_ready": self._extract_readiness(
                readiness
            ),
            "validation": validation,
            "readiness": readiness,
            "quality_report": quality_report,
            "missing_values": missing_analysis,
            "numeric_validation": numeric_analysis,
            "warnings": warning_summary,
            "phase_1_core_modified": False,
        }

    def get_quality_status(
        self,
        data: pd.DataFrame,
    ) -> str:
        """Return a simple quality status."""

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        validation = self.validator.validate(data)

        return self._quality_status(validation)

    def is_ready(
        self,
        data: pd.DataFrame,
        mode: str = "LIVE_FUTURE_SCREENING",
    ) -> bool:
        """Return whether the dataset is ready for the requested mode."""

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        normalized_mode = self._normalize_mode(mode)

        readiness = self.validator.readiness(
            data,
            mode=normalized_mode,
        )

        return self._extract_readiness(readiness)

    # ------------------------------------------------------------------
    # Missing-value intelligence
    # ------------------------------------------------------------------

    @staticmethod
    def _missing_value_analysis(
        data: pd.DataFrame,
    ) -> Dict[str, Any]:

        missing_counts = data.isna().sum()

        missing_by_column = {
            str(column): int(count)
            for column, count in missing_counts.items()
            if int(count) > 0
        }

        total_missing = int(
            missing_counts.sum()
        )

        total_cells = int(
            data.shape[0] * data.shape[1]
        )

        missing_percentage = (
            float(total_missing / total_cells * 100.0)
            if total_cells > 0
            else 0.0
        )

        return {
            "total_missing_values": total_missing,
            "missing_percentage": missing_percentage,
            "columns_with_missing_values": (
                len(missing_by_column)
            ),
            "by_column": missing_by_column,
        }

    # ------------------------------------------------------------------
    # Numeric intelligence
    # ------------------------------------------------------------------

    @staticmethod
    def _numeric_analysis(
        data: pd.DataFrame,
    ) -> Dict[str, Any]:

        numeric_columns = data.select_dtypes(
            include="number"
        ).columns.tolist()

        non_numeric_columns = [
            column
            for column in data.columns
            if column not in numeric_columns
        ]

        invalid_numeric_columns: Dict[str, int] = {}

        for column in data.columns:

            if column in numeric_columns:
                continue

            converted = pd.to_numeric(
                data[column],
                errors="coerce",
            )

            original_non_null = int(
                data[column].notna().sum()
            )

            converted_non_null = int(
                converted.notna().sum()
            )

            invalid_count = (
                original_non_null
                - converted_non_null
            )

            if invalid_count > 0:
                invalid_numeric_columns[
                    str(column)
                ] = int(invalid_count)

        return {
            "numeric_column_count": len(
                numeric_columns
            ),
            "numeric_columns": [
                str(column)
                for column in numeric_columns
            ],
            "non_numeric_column_count": len(
                non_numeric_columns
            ),
            "non_numeric_columns": [
                str(column)
                for column in non_numeric_columns
            ],
            "invalid_numeric_values": (
                invalid_numeric_columns
            ),
        }

    # ------------------------------------------------------------------
    # Warning intelligence
    # ------------------------------------------------------------------

    @staticmethod
    def _warning_summary(
        validation: Any,
        quality_report: Any,
    ) -> Dict[str, Any]:

        warnings: List[str] = []

        if isinstance(validation, dict):
            raw_warnings = validation.get(
                "warnings",
                [],
            )

            if isinstance(raw_warnings, list):
                warnings.extend(
                    str(item)
                    for item in raw_warnings
                )
            elif raw_warnings:
                warnings.append(
                    str(raw_warnings)
                )

        if isinstance(quality_report, dict):

            report_warnings = quality_report.get(
                "warnings",
                [],
            )

            if isinstance(report_warnings, list):
                warnings.extend(
                    str(item)
                    for item in report_warnings
                )
            elif report_warnings:
                warnings.append(
                    str(report_warnings)
                )

        unique_warnings = list(
            dict.fromkeys(warnings)
        )

        return {
            "count": len(unique_warnings),
            "items": unique_warnings,
            "has_warnings": bool(
                unique_warnings
            ),
        }

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _quality_status(
        validation: Any,
    ) -> str:

        if isinstance(validation, dict):

            status = validation.get(
                "quality_status"
            )

            if status is not None:
                return str(status).upper()

            status = validation.get(
                "status"
            )

            if status is not None:
                return str(status).upper()

            valid = validation.get(
                "valid"
            )

            if valid is True:
                return "PASS"

            if valid is False:
                return "FAIL"

        return "REVIEW"

    @staticmethod
    def _extract_readiness(
        readiness: Any,
    ) -> bool:

        if isinstance(readiness, bool):
            return readiness

        if isinstance(readiness, dict):

            for key in (
                "ready",
                "is_ready",
                "data_ready",
                "ready_for_mode",
            ):
                value = readiness.get(key)

                if isinstance(value, bool):
                    return value

            status = readiness.get(
                "status"
            )

            if isinstance(status, str):
                return status.upper() in {
                    "READY",
                    "PASS",
                    "OK",
                }

        return False

    @staticmethod
    def _normalize_mode(
        mode: str,
    ) -> str:

        if not isinstance(mode, str):
            raise TypeError(
                "mode must be a string"
            )

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
    engine = DataQualityIntelligence()

    print("DataQualityIntelligence : PASS")
    print("Version:", engine.VERSION)
    print(
        "Capabilities:",
        engine.get_engine_info()[
            "capabilities"
        ],
    )