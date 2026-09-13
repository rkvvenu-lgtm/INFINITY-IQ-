"""
SIH26170 - Generic Screening: Feature Discovery
================================================
Classifies every column of an arbitrary dataset and decides whether
the dataset can be screened.

Classification groups
---------------------
parameters       : numeric measurement columns (the "parameters")
group_columns    : categorical columns that can act as lot/group keys
id_columns       : identifier columns excluded from screening
time_columns     : date/time columns used for drift context
limit_columns    : *_min / *_max / limit / threshold columns (spectral)
categorical_other: remaining text columns (informational only)
phase_map        : {base_parameter: [phase columns]} e.g. Iddq -> [Iddq_0h_uA, Iddq_24h_uA, ...]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Column-name hints
# ------------------------------------------------------------------

_ID_HINTS = (
    "id",
    "serial",
    "s_no",
    "sl_no",
    "slno",
    "record",
    "row_no",
    "rownum",
    "row_num",
    "index",
    "code",
    "key",
    "number",
)

_LIMIT_HINTS = (
    "limit",
    "threshold",
    "_max",
    "_min",
    "max_",
    "min_",
    "allowed",
    "tolerance",
    "spec_limit",
    "spec_max",
    "spec_min",
)

_TIME_HINTS = (
    "time",
    "_time",
    "hour",
    "date",
    "day",
    "year",
    "month",
    "week",
    "timestamp",
)

_GROUP_HINTS = (
    "lot",
    "batch",
    "group",
    "category",
    "type",
    "region",
    "plant",
    "unit",
    "class",
    "zone",
    "team",
    "line",
    "station",
    "machine",
    "operator",
    "shift",
)

# Parameter base suffix pattern, e.g. Iddq_0h_uA, Temp_24h, Sales_168h
_PHASE_COLUMN_RE = re.compile(
    r"^(?P<base>.+)_(?P<phase>\d+)"
    r"(?P<unit>h|hr|hrs|hour|hours|min|m|s|ms|ns|us|d|day|days)"
    r"(?P<rest>_.*)?$",
    re.IGNORECASE,
)

# Limit columns that mirror a measurement, e.g. Iddq_Max_Limit_uA
_LIMIT_COLUMN_RE = re.compile(
    r"^(?P<base>.+)_(max|min)(_limit)?"
    r"(?P<rest>_.*)?$",
    re.IGNORECASE,
)


@dataclass
class FeatureDiscovery:
    """Result of analysing an arbitrary dataset."""

    parameters: List[str] = field(default_factory=list)
    group_columns: List[str] = field(default_factory=list)
    id_columns: List[str] = field(default_factory=list)
    time_columns: List[str] = field(default_factory=list)
    limit_columns: List[str] = field(default_factory=list)
    categorical_other: List[str] = field(default_factory=list)
    phase_map: Dict[str, List[str]] = field(default_factory=dict)
    dropped_parameters: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameters": self.parameters,
            "group_columns": self.group_columns,
            "id_columns": self.id_columns,
            "time_columns": self.time_columns,
            "limit_columns": self.limit_columns,
            "categorical_other": self.categorical_other,
            "phase_map": self.phase_map,
            "dropped_parameters": self.dropped_parameters,
        }

    @property
    def has_parameters(self) -> bool:
        return len(self.parameters) > 0


class GenericFeatureDiscovery:
    """Column auto-classification and generic readiness assessment."""

    VERSION = "1.0"

    MINIMUM_ROWS = 10
    MAX_MISSING_PERCENT = 40.0
    MIN_NUMERIC_VALUES = 20

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover(self, data: pd.DataFrame) -> FeatureDiscovery:
        """Classify all columns of the dataframe."""

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        if data.empty:
            return FeatureDiscovery()

        parameters: List[str] = []
        group_columns: List[str] = []
        id_columns: List[str] = []
        time_columns: List[str] = []
        limit_columns: List[str] = []
        categorical_other: List[str] = []
        phase_map: Dict[str, List[str]] = {}
        dropped: List[str] = []

        numeric_hits: Dict[str, List[str]] = {}

        for column in data.columns:
            name = str(column).strip()
            lower = name.lower()
            series = data[name]

            # ------------------------------------------------------
            # Date / time columns
            # ------------------------------------------------------
            if pd.api.types.is_datetime64_any_dtype(series) or self._looks_like_date(series):
                time_columns.append(name)
                continue

            # ------------------------------------------------------
            # Phase column detection (Iddq_0h_uA style)
            # ------------------------------------------------------
            phase_match = _PHASE_COLUMN_RE.match(name)
            if phase_match and self._is_numeric_like(series):
                base = phase_match.group("base").strip()
                numeric_hits.setdefault(base, []).append(name)
                continue

            # ------------------------------------------------------
            # Limit / spec columns
            # ------------------------------------------------------
            if self._looks_like_limit_column(name):
                limit_columns.append(name)
                continue

            # ------------------------------------------------------
            # Numeric columns
            # ------------------------------------------------------
            if self._is_numeric_like(series):
                if self._looks_like_id(name, series, data):
                    id_columns.append(name)
                elif self._low_information(series):
                    dropped.append(name)
                else:
                    parameters.append(name)
                continue

            # ------------------------------------------------------
            # Object / categorical columns
            # ------------------------------------------------------
            if self._looks_like_id(name, series, data):
                id_columns.append(name)
                continue

            if self._looks_like_limit_column(name):
                limit_columns.append(name)
                continue

            if self._is_time_hint(name):
                time_columns.append(name)
                continue

            if self._is_group_hint(name):
                group_columns.append(name)
                continue

            categorical_other.append(name)

        # ----------------------------------------------------------
        # Resolve phase groups into logical parameters.
        # ----------------------------------------------------------
        for base, columns in sorted(numeric_hits.items()):
            base_lower = base.lower()
            if self._looks_like_id(base, None, None):
                id_columns.extend(columns)
                continue
            ordered = sorted(
                columns,
                key=lambda col: self._phase_number(col),
            )
            phase_map[base] = ordered
            if base not in parameters:
                parameters.append(base)

        # ----------------------------------------------------------
        # Limit columns that mirror a parameter (_Max_Limit style)
        # ----------------------------------------------------------
        self._assign_mirror_limits(limit_columns, parameters, data)

        parameters = self._dedupe(parameters)
        group_columns = self._dedupe(group_columns)
        id_columns = self._dedupe(id_columns)
        time_columns = self._dedupe(time_columns)
        limit_columns = self._dedupe(limit_columns)
        categorical_other = self._dedupe(categorical_other)

        return FeatureDiscovery(
            parameters=parameters,
            group_columns=group_columns,
            id_columns=id_columns,
            time_columns=time_columns,
            limit_columns=limit_columns,
            categorical_other=categorical_other,
            phase_map=phase_map,
            dropped_parameters=dropped,
        )

    def readiness(
        self,
        data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Return {ready, warnings, discovery} for arbitrary data."""

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        warnings: List[str] = []
        discovery = self.discover(data)

        if len(data) < self.MINIMUM_ROWS:
            warnings.append(
                f"Only {len(data)} rows found; at least "
                f"{self.MINIMUM_ROWS} rows are recommended "
                "for reliable statistical screening."
            )

        if not discovery.has_parameters:
            warnings.append(
                "No numeric measurement columns were detected. "
                "At least one numeric parameter is required."
            )

        if discovery.dropped_parameters:
            warnings.append(
                "Column(s) dropped as low-information: "
                + ", ".join(discovery.dropped_parameters)
                + "."
            )

        for column in discovery.parameters:
            if column not in data.columns:
                # Phase-group base names are not real columns; skip.
                continue
            missing_percent = float(
                data[column].isna().mean() * 100.0
            )
            if missing_percent > self.MAX_MISSING_PERCENT:
                warnings.append(
                    f"Parameter '{column}' has {missing_percent:.1f}% "
                    "missing values (>40%)."
                )
            elif missing_percent > 0:
                warnings.append(
                    f"Parameter '{column}' contains "
                    f"{missing_percent:.1f}% missing values "
                    "(median-filled during screening)."
                )

        ready = (
            len(data) >= self.MINIMUM_ROWS
            and discovery.has_parameters
        )

        return {
            "ready": ready,
            "warnings": warnings,
            "rows": int(len(data)),
            "parameters": discovery.parameters,
            "discovery": discovery.to_dict(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dedupe(items: List[str]) -> List[str]:
        seen = set()
        out = []
        for item in items:
            key = item.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    @staticmethod
    def _is_numeric_like(series: pd.Series) -> bool:
        return pd.api.types.is_numeric_dtype(series) or (
            series.dropna().map(_try_float).mean() > 0.8
            if len(series.dropna()) > 0
            else False
        )

    @staticmethod
    def _low_information(series: pd.Series) -> bool:
        """Constant or near-constant numeric columns are dropped."""
        values = series.dropna()
        if len(values) == 0:
            return True
        if values.nunique() <= 1:
            return True
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        if len(numeric) < 3:
            return True
        return bool(numeric.std(ddof=0) == 0)

    @classmethod
    def _looks_like_id(
        cls,
        name: str,
        series: pd.Series,
        data: pd.DataFrame,
    ) -> bool:
        lower = name.lower()
        if any(hint in lower for hint in _ID_HINTS):
            return True
        return False

    @staticmethod
    def _looks_like_limit_column(name: str) -> bool:
        lower = name.lower()
        if any(hint in lower for hint in ("limit", "threshold", "tolerance")):
            return True
        return bool(_LIMIT_COLUMN_RE.match(name))

    @staticmethod
    def _is_time_hint(name: str) -> bool:
        lower = name.lower()
        if lower in {"time", "datetime"}:
            return True
        return any(hint in lower for hint in _TIME_HINTS)

    @staticmethod
    def _is_group_hint(name: str) -> bool:
        lower = name.lower()
        return any(hint in lower for hint in _GROUP_HINTS)

    @staticmethod
    def _phase_number(column: str) -> int:
        match = _PHASE_COLUMN_RE.match(column)
        if match:
            try:
                return int(match.group("phase"))
            except ValueError:
                return 0
        return 0

    @staticmethod
    def _looks_like_date(series: pd.Series) -> bool:
        if series.empty:
            return False
        sample = series.dropna().iloc[0]
        return isinstance(sample, (pd.Timestamp,))

    @staticmethod
    def _assign_mirror_limits(
        limit_columns: List[str],
        parameters: List[str],
        data: pd.DataFrame,
    ) -> None:
        """Keep *_Max/_Min limit columns only when a matching parameter exists."""
        base_names = {str(p).lower() for p in parameters}
        retained = []
        for column in limit_columns:
            match = _LIMIT_COLUMN_RE.match(column)
            if match and match.group("base").strip().lower() in base_names:
                retained.append(column)
            elif (
                "limit" in column.lower()
                or "threshold" in column.lower()
                or "tolerance" in column.lower()
            ):
                retained.append(column)
        limit_columns[:] = retained


def _try_float(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False