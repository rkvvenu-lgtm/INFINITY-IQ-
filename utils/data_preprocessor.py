from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


class DataPreprocessor:
    """
    Central preprocessing layer for SIH26170.

    Responsibilities:
        - Normalize column names
        - Preserve source data
        - Convert numeric feature columns safely
        - Handle missing numeric values
        - Prepare data for AI modules
        - Provide reusable numeric feature extraction

    This class does NOT:
        - make engineering decisions
        - detect anomalies
        - train prediction models
        - calculate final risk
    """

    def __init__(
        self,
        missing_numeric_strategy: str = "median",
        preserve_original: bool = True,
    ):
        self.missing_numeric_strategy = (
            str(missing_numeric_strategy).lower()
        )
        self.preserve_original = preserve_original

        allowed_strategies = {
            "median",
            "mean",
            "zero",
            "none",
        }

        if self.missing_numeric_strategy not in allowed_strategies:
            raise ValueError(
                "Unsupported missing_numeric_strategy. "
                f"Allowed values: {sorted(allowed_strategies)}"
            )

    # ============================================================
    # MAIN PREPARATION
    # ============================================================

    def prepare(
        self,
        data: pd.DataFrame,
        numeric_columns: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """
        Prepare a DataFrame for downstream AI processing.

        Parameters
        ----------
        data:
            Input dataset.

        numeric_columns:
            Optional explicit list of columns that should be
            treated as numeric.

        Returns
        -------
        pandas.DataFrame
            Prepared dataset.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Input data must be a pandas DataFrame."
            )

        if data.empty:
            raise ValueError(
                "Input dataset contains no rows."
            )

        result = data.copy()

        # --------------------------------------------------------
        # Column normalization
        # --------------------------------------------------------

        result.columns = [
            self._normalize_column_name(column)
            for column in result.columns
        ]

        # --------------------------------------------------------
        # Remove completely empty columns
        # --------------------------------------------------------

        result = result.dropna(
            axis=1,
            how="all",
        )

        if result.empty:
            raise ValueError(
                "Dataset contains no usable columns."
            )

        # --------------------------------------------------------
        # Determine numeric columns
        # --------------------------------------------------------

        if numeric_columns is None:
            numeric_candidates = self._detect_numeric_columns(
                result
            )
        else:
            numeric_candidates = [
                column
                for column in numeric_columns
                if column in result.columns
            ]

        # --------------------------------------------------------
        # Numeric conversion
        # --------------------------------------------------------

        for column in numeric_candidates:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

        # --------------------------------------------------------
        # Missing numeric values
        # --------------------------------------------------------

        result = self._handle_missing_numeric_values(
            result,
            numeric_candidates,
        )

        # --------------------------------------------------------
        # Final cleanup
        # --------------------------------------------------------

        result = result.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        # Re-apply missing-value policy after infinity handling.
        result = self._handle_missing_numeric_values(
            result,
            numeric_candidates,
        )

        return result

    # ============================================================
    # COLUMN NORMALIZATION
    # ============================================================

    @staticmethod
    def _normalize_column_name(column) -> str:
        """
        Normalize a column name without changing its semantic
        structure.

        Examples:
            ' Iddq_0h_uA ' -> 'Iddq_0h_uA'
        """

        return str(column).strip()

    # ============================================================
    # NUMERIC COLUMN DETECTION
    # ============================================================

    @staticmethod
    def _detect_numeric_columns(
        data: pd.DataFrame,
    ) -> list[str]:
        """
        Detect columns that contain meaningful numeric data.

        Existing numeric columns are always selected.

        Object/string columns are selected when a substantial
        portion of their values can be converted to numeric.
        """

        numeric_columns = []

        for column in data.columns:

            series = data[column]

            if pd.api.types.is_numeric_dtype(series):
                numeric_columns.append(column)
                continue

            converted = pd.to_numeric(
                series,
                errors="coerce",
            )

            non_empty = series.notna().sum()

            if non_empty == 0:
                continue

            convertible = converted.notna().sum()

            conversion_ratio = (
                convertible / non_empty
            )

            if conversion_ratio >= 0.80:
                numeric_columns.append(column)

        return numeric_columns

    # ============================================================
    # MISSING VALUE HANDLING
    # ============================================================

    def _handle_missing_numeric_values(
        self,
        data: pd.DataFrame,
        numeric_columns: Iterable[str],
    ) -> pd.DataFrame:
        """
        Apply the configured missing-value strategy.
        """

        result = data.copy()

        for column in numeric_columns:

            if column not in result.columns:
                continue

            series = result[column]

            if not series.isna().any():
                continue

            if self.missing_numeric_strategy == "median":

                value = series.median()

                if pd.notna(value):
                    result[column] = series.fillna(
                        value
                    )

            elif self.missing_numeric_strategy == "mean":

                value = series.mean()

                if pd.notna(value):
                    result[column] = series.fillna(
                        value
                    )

            elif self.missing_numeric_strategy == "zero":

                result[column] = series.fillna(0)

            elif self.missing_numeric_strategy == "none":
                # Leave missing values untouched.
                pass

        return result

    # ============================================================
    # NUMERIC FEATURE EXTRACTION
    # ============================================================

    def get_numeric_features(
        self,
        data: pd.DataFrame,
        columns: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """
        Return numeric features suitable for ML algorithms.

        If columns are supplied, only those existing columns are
        returned.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Input data must be a pandas DataFrame."
            )

        if columns is None:

            selected = self._detect_numeric_columns(
                data
            )

        else:

            selected = [
                column
                for column in columns
                if column in data.columns
            ]

        if not selected:
            return pd.DataFrame(
                index=data.index
            )

        features = data[selected].copy()

        for column in features.columns:
            features[column] = pd.to_numeric(
                features[column],
                errors="coerce",
            )

        return features

    # ============================================================
    # REQUIRED FEATURE CHECK
    # ============================================================

    @staticmethod
    def check_required_columns(
        data: pd.DataFrame,
        required_columns: Iterable[str],
    ) -> dict:
        """
        Check whether required columns are available.

        Returns a structured result so that higher layers can
        decide whether to reject, warn, or route the dataset.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Input data must be a pandas DataFrame."
            )

        required = list(required_columns)

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
            "required_columns": required,
            "available_columns": available,
            "missing_columns": missing,
        }

    # ============================================================
    # DATA QUALITY SUMMARY
    # ============================================================

    @staticmethod
    def get_quality_summary(
        data: pd.DataFrame,
    ) -> dict:
        """
        Generate basic preprocessing/data-quality statistics.
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
            "duplicate_rows": int(
                data.duplicated().sum()
            ),
        }

    # ============================================================
    # COPY / ORIGINAL DATA SUPPORT
    # ============================================================

    def preserve_source(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Return a safe copy of the source DataFrame.

        This ensures preprocessing never mutates the caller's
        original DataFrame.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Input data must be a pandas DataFrame."
            )

        return data.copy(deep=True)