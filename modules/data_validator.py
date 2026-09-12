from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


class DataValidator:
    """
    Central data-quality and structural validation layer.

    Responsibilities:
        - Validate dataset structure
        - Detect missing values
        - Detect invalid numeric values
        - Check duplicate rows
        - Check required columns
        - Check training/live data readiness
        - Produce reusable quality reports

    This class does NOT:
        - perform ML anomaly detection
        - train prediction models
        - make PASS/REJECT engineering decisions
    """

    VALID_MODES = {
        "AUTO",
        "TRAINING_EVALUATION",
        "LIVE_FUTURE_SCREENING",
    }

    # ============================================================
    # MAIN VALIDATION
    # ============================================================

    def validate(
        self,
        data: pd.DataFrame,
        mode: str = "AUTO",
        required_columns: Optional[Iterable[str]] = None,
        minimum_rows: int = 1,
    ) -> pd.DataFrame:
        """
        Validate and return a safe copy of the dataset.

        Raises ValueError when the dataset is structurally unusable.
        Warnings are stored as DataFrame metadata columns.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Input data must be a pandas DataFrame."
            )

        if data.empty:
            raise ValueError(
                "Dataset is empty. At least one row is required."
            )

        if minimum_rows < 1:
            minimum_rows = 1

        if len(data) < minimum_rows:
            raise ValueError(
                f"Dataset contains {len(data)} rows, "
                f"but at least {minimum_rows} rows are required."
            )

        mode = str(mode).upper()

        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unsupported validation mode '{mode}'. "
                f"Allowed modes: {sorted(self.VALID_MODES)}"
            )

        result = data.copy()

        # Normalize column names.
        result.columns = [
            str(column).strip()
            for column in result.columns
        ]

        warnings = []

        # --------------------------------------------------------
        # Duplicate rows
        # --------------------------------------------------------

        duplicate_count = int(
            result.duplicated().sum()
        )

        if duplicate_count > 0:
            warnings.append(
                f"{duplicate_count} duplicate row(s) detected."
            )

        # --------------------------------------------------------
        # Missing values
        # --------------------------------------------------------

        missing_counts = result.isna().sum()
        missing_columns = [
            column
            for column, count in missing_counts.items()
            if count > 0
        ]

        if missing_columns:
            warnings.append(
                "Missing values detected in: "
                + ", ".join(missing_columns)
                + "."
            )

        # --------------------------------------------------------
        # Infinite numeric values
        # --------------------------------------------------------

        numeric_columns = list(
            result.select_dtypes(
                include=np.number
            ).columns
        )

        infinite_columns = []

        for column in numeric_columns:

            values = result[column].to_numpy(
                dtype=float,
                copy=False
            )

            if np.isinf(values).any():
                infinite_columns.append(column)

        if infinite_columns:
            warnings.append(
                "Infinite numeric values detected in: "
                + ", ".join(infinite_columns)
                + "."
            )

        # --------------------------------------------------------
        # Required columns
        # --------------------------------------------------------

        missing_required = []

        if required_columns is not None:

            required = [
                str(column).strip()
                for column in required_columns
            ]

            missing_required = [
                column
                for column in required
                if column not in result.columns
            ]

            if missing_required:
                raise ValueError(
                    "Required column(s) missing: "
                    + ", ".join(missing_required)
                )

        # --------------------------------------------------------
        # Mode-specific readiness
        # --------------------------------------------------------

        mode_warnings = self._check_mode_readiness(
            result,
            mode
        )

        warnings.extend(mode_warnings)

        # --------------------------------------------------------
        # Quality status
        # --------------------------------------------------------

        if missing_required:
            quality_status = "INVALID"

        elif infinite_columns:
            quality_status = "WARNING"

        elif missing_columns or duplicate_count > 0:
            quality_status = "WARNING"

        else:
            quality_status = "PASS"

        result["Data_Quality_Status"] = quality_status

        if warnings:
            result["Data_Quality_Warnings"] = " | ".join(
                dict.fromkeys(warnings)
            )
        else:
            result["Data_Quality_Warnings"] = "None"

        return result

    # ============================================================
    # MODE READINESS
    # ============================================================

    @staticmethod
    def _check_mode_readiness(
        data: pd.DataFrame,
        mode: str,
    ) -> list[str]:
        """
        Check whether the dataset contains the measurements
        expected by the selected operating mode.
        """

        warnings = []

        electronics_early_groups = [
            (
                "Iddq",
                "Iddq_0h_uA",
                "Iddq_24h_uA"
            ),
            (
                "Leakage",
                "Leakage_0h_uA",
                "Leakage_24h_uA"
            ),
            (
                "Delay",
                "Delay_0h_ns",
                "Delay_24h_ns"
            ),
        ]

        if mode in {
            "AUTO",
            "LIVE_FUTURE_SCREENING"
        }:

            available_groups = 0

            for parameter, current, early in electronics_early_groups:

                if (
                    current in data.columns
                    and early in data.columns
                ):
                    available_groups += 1

            if available_groups == 0:
                warnings.append(
                    "No complete 0h + 24h electronics "
                    "measurement pair detected."
                )

        if mode == "TRAINING_EVALUATION":

            target_columns = [
                "Iddq_168h_uA",
                "Leakage_168h_uA",
                "Delay_168h_ns",
            ]

            available_targets = [
                column
                for column in target_columns
                if column in data.columns
            ]

            if not available_targets:
                warnings.append(
                    "No configured 168h target column detected "
                    "for training/evaluation."
                )

        return warnings

    # ============================================================
    # REQUIRED COLUMN CHECK
    # ============================================================

    @staticmethod
    def check_required_columns(
        data: pd.DataFrame,
        required_columns: Iterable[str],
    ) -> dict:
        """
        Return a structured required-column validation result.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Input data must be a pandas DataFrame."
            )

        required = [
            str(column).strip()
            for column in required_columns
        ]

        available = [
            column
            for column in required
            if column in data.columns
        ]

        missing = [
            column
            for column in required
            if column not in data.columns
        ]

        return {
            "valid": len(missing) == 0,
            "required": required,
            "available": available,
            "missing": missing,
        }

    # ============================================================
    # NUMERIC VALIDATION
    # ============================================================

    @staticmethod
    def validate_numeric_columns(
        data: pd.DataFrame,
        columns: Iterable[str],
    ) -> dict:
        """
        Check whether selected columns contain valid numeric
        values.

        Numeric-looking strings are considered convertible,
        while non-convertible values are reported.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Input data must be a pandas DataFrame."
            )

        report = {}

        for column in columns:

            if column not in data.columns:

                report[column] = {
                    "exists": False,
                    "valid": False,
                    "invalid_count": None,
                }

                continue

            converted = pd.to_numeric(
                data[column],
                errors="coerce"
            )

            original_non_null = data[column].notna()

            invalid_mask = (
                original_non_null
                & converted.isna()
            )

            invalid_count = int(
                invalid_mask.sum()
            )

            report[column] = {
                "exists": True,
                "valid": invalid_count == 0,
                "invalid_count": invalid_count,
            }

        return report

    # ============================================================
    # QUALITY REPORT
    # ============================================================

    @staticmethod
    def quality_report(
        data: pd.DataFrame,
    ) -> dict:
        """
        Generate a complete dataset-quality summary.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Input data must be a pandas DataFrame."
            )

        total_cells = (
            data.shape[0] * data.shape[1]
        )

        missing_cells = int(
            data.isna().sum().sum()
        )

        numeric_columns = list(
            data.select_dtypes(
                include=np.number
            ).columns
        )

        infinite_cells = 0

        for column in numeric_columns:

            values = data[column].to_numpy(
                dtype=float,
                copy=False
            )

            infinite_cells += int(
                np.isinf(values).sum()
            )

        return {
            "rows": int(data.shape[0]),
            "columns": int(data.shape[1]),
            "numeric_columns": len(
                numeric_columns
            ),
            "missing_cells": missing_cells,
            "missing_percentage": (
                round(
                    (
                        missing_cells
                        / total_cells
                    ) * 100,
                    4,
                )
                if total_cells
                else 0.0
            ),
            "infinite_cells": infinite_cells,
            "duplicate_rows": int(
                data.duplicated().sum()
            ),
            "column_names": list(
                data.columns
            ),
        }

    # ============================================================
    # DATASET READINESS
    # ============================================================

    def readiness(
        self,
        data: pd.DataFrame,
        mode: str = "AUTO",
    ) -> dict:
        """
        Return a non-destructive readiness assessment.

        Unlike validate(), this method does not modify the data.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Input data must be a pandas DataFrame."
            )

        mode = str(mode).upper()

        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unsupported validation mode '{mode}'."
            )

        quality = self.quality_report(
            data
        )

        warnings = self._check_mode_readiness(
            data,
            mode
        )

        if quality["rows"] == 0:
            status = "INVALID"

        elif warnings:
            status = "WARNING"

        elif (
            quality["missing_cells"] > 0
            or quality["infinite_cells"] > 0
        ):
            status = "WARNING"

        else:
            status = "READY"

        return {
            "status": status,
            "mode": mode,
            "warnings": warnings,
            "quality": quality,
        }

    # ============================================================
    # CLEAN QUALITY WARNINGS
    # ============================================================

    @staticmethod
    def get_warning_columns(
        data: pd.DataFrame,
    ) -> list[str]:
        """
        Return columns containing missing values.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Input data must be a pandas DataFrame."
            )

        return [
            column
            for column in data.columns
            if data[column].isna().any()
        ]