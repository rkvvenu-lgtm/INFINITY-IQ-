"""
SIH26170 - Judge Dashboard Adapter
Phase 3 / File 11

UI-facing adapter for the final judge/product dashboard.

This module prepares compact, presentation-ready dashboard data from an
already-generated screening DataFrame. It does not modify Phase 1 AI
modules or Phase 2 screening modules.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


class JudgeDashboard:
    """Prepare judge-ready dashboard metrics and investigation views."""

    VERSION = "3.0"

    def __init__(self, project_root: str | Path | None = None) -> None:
        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[1]
        )

    @staticmethod
    def _text(value: Any, default: str = "N/A") -> str:
        if value is None:
            return default
        text = str(value).strip()
        if text.lower() in {"", "nan", "none", "null"}:
            return default
        return text

    @staticmethod
    def _now_utc() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _find(data: pd.DataFrame, names: list[str]) -> Optional[str]:
        for name in names:
            if name in data.columns:
                return name
        return None

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            if isinstance(value, float) and pd.isna(value):
                return None
            return value
        if isinstance(value, dict):
            return {
                str(k): JudgeDashboard._json_safe(v)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [JudgeDashboard._json_safe(v) for v in value]
        if hasattr(value, "item"):
            try:
                return JudgeDashboard._json_safe(value.item())
            except Exception:
                pass
        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except Exception:
                pass
        return str(value)

    def get_info(self) -> Dict[str, Any]:
        return {
            "module": "JudgeDashboard",
            "version": self.VERSION,
            "phase": "PHASE_3",
            "purpose": "Judge-ready AI screening dashboard data adapter",
            "capabilities": [
                "executive_metrics",
                "risk_distribution",
                "decision_distribution",
                "prediction_status",
                "investigation_queue",
                "component_drilldown",
                "quantitative_why",
                "data_quality_status",
                "metadata",
            ],
        }

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def executive_metrics(
        self,
        data: pd.DataFrame,
        *,
        domain: str = "general_unknown",
        screening_status: str = "SCREENED",
        dataset_id: Any = None,
    ) -> Dict[str, Any]:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")

        risk_col = self._find(
            data,
            ["Overall_Risk_Percentage", "Overall_Risk_Score", "Risk_Score"],
        )
        risk_level_col = self._find(
            data,
            ["Risk_Level", "Overall_Risk_Level"],
        )
        decision_col = self._find(
            data,
            ["Risk_Decision", "Final_Decision", "Decision"],
        )
        anomaly_col = self._find(
            data,
            ["Anomaly_Flag", "Anomaly_Risk"],
        )

        predictions = [
            column
            for column in data.columns
            if str(column).startswith("Predicted_")
            and "168h" in str(column)
        ]

        anomaly_count = 0
        if anomaly_col:
            if anomaly_col == "Anomaly_Flag":
                anomaly_count = int(
                    data[anomaly_col]
                    .astype(str)
                    .str.upper()
                    .isin(["TRUE", "1", "YES"])
                    .sum()
                )
            else:
                anomaly_count = int(
                    pd.to_numeric(
                        data[anomaly_col],
                        errors="coerce",
                    ).fillna(0).gt(0).sum()
                )

        risk_distribution: Dict[str, int] = {}
        if risk_level_col:
            values = data[risk_level_col].fillna("UNKNOWN").astype(str).str.upper()
            risk_distribution = {
                str(k): int(v)
                for k, v in values.value_counts().to_dict().items()
            }

        decision_distribution: Dict[str, int] = {}
        if decision_col:
            values = data[decision_col].fillna("UNKNOWN").astype(str).str.upper()
            decision_distribution = {
                str(k): int(v)
                for k, v in values.value_counts().to_dict().items()
            }

        risk_values = (
            pd.to_numeric(data[risk_col], errors="coerce").dropna()
            if risk_col
            else pd.Series(dtype=float)
        )

        return {
            "dashboard_version": self.VERSION,
            "generated_at_utc": self._now_utc(),
            "domain": self._text(domain),
            "screening_status": self._text(screening_status),
            "dataset_id": self._text(dataset_id),
            "components_screened": int(len(data)),
            "anomalies": anomaly_count,
            "predicted_168h_parameters": len(predictions),
            "prediction_status": (
                "AVAILABLE" if predictions else "NOT_AVAILABLE"
            ),
            "average_risk": (
                float(risk_values.mean())
                if not risk_values.empty
                else None
            ),
            "maximum_risk": (
                float(risk_values.max())
                if not risk_values.empty
                else None
            ),
            "risk_distribution": risk_distribution,
            "decision_distribution": decision_distribution,
        }

    # ------------------------------------------------------------------
    # Prediction status
    # ------------------------------------------------------------------

    def prediction_status(self, data: pd.DataFrame) -> Dict[str, Any]:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")

        predicted = [
            column for column in data.columns
            if str(column).startswith("Predicted_")
            and "168h" in str(column)
        ]

        return {
            "target_horizon": "168h",
            "prediction_available": bool(predicted),
            "prediction_columns": predicted,
            "prediction_count": len(predicted),
            "future_actual_168h_required": False,
            "message": (
                "168h values are predictions from earlier measurements; "
                "future 168h actual measurement is not required for live screening."
                if predicted
                else "No 168h prediction columns are available."
            ),
        }

    # ------------------------------------------------------------------
    # Investigation
    # ------------------------------------------------------------------

    def investigation_queue(
        self,
        data: pd.DataFrame,
        limit: int = 50,
    ) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")

        result = data.copy()

        decision_col = self._find(
            result,
            ["Risk_Decision", "Final_Decision", "Decision"],
        )
        risk_col = self._find(
            result,
            ["Overall_Risk_Percentage", "Overall_Risk_Score", "Risk_Score"],
        )

        if decision_col:
            decisions = result[decision_col].astype(str).str.upper()
            mask = decisions.isin(
                ["REVIEW", "INVESTIGATE", "REJECT", "MONITOR"]
            )
            result = result.loc[mask].copy()
        elif risk_col:
            values = pd.to_numeric(result[risk_col], errors="coerce").fillna(0)
            result = result.loc[values >= 50].copy()

        if risk_col and not result.empty:
            numeric = pd.to_numeric(result[risk_col], errors="coerce")
            result = (
                result.assign(_dashboard_sort_risk=numeric)
                .sort_values("_dashboard_sort_risk", ascending=False)
                .drop(columns=["_dashboard_sort_risk"])
            )

        return result.head(max(1, int(limit))).copy()

    def component_drilldown(
        self,
        data: pd.DataFrame,
        component_id: Any,
    ) -> Dict[str, Any]:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")

        component_col = self._find(
            data,
            ["Component_ID", "component_id", "Component"],
        )

        if component_col is None:
            return {
                "status": "COMPONENT_ID_COLUMN_NOT_FOUND",
                "component_id": self._text(component_id),
            }

        target = self._text(component_id, "")
        rows = data[
            data[component_col].astype(str).str.strip() == target
        ]

        if rows.empty:
            return {
                "status": "COMPONENT_NOT_FOUND",
                "component_id": target,
            }

        record = rows.iloc[0].to_dict()

        quantitative_columns = [
            column for column in rows.columns
            if any(
                token in str(column).lower()
                for token in [
                    "z_score",
                    "zscore",
                    "deviation",
                    "drift",
                    "predicted",
                    "limit",
                    "contribution",
                    "why",
                    "explanation",
                ]
            )
        ]

        quantitative = {
            column: self._json_safe(record.get(column))
            for column in quantitative_columns
        }

        return {
            "status": "FOUND",
            "component_id": target,
            "record": self._json_safe(record),
            "quantitative_evidence": quantitative,
        }

    # ------------------------------------------------------------------
    # WHY / explanation view
    # ------------------------------------------------------------------

    def why_view(
        self,
        data: pd.DataFrame,
        component_id: Any,
    ) -> Dict[str, Any]:
        result = self.component_drilldown(data, component_id)

        if result.get("status") != "FOUND":
            return result

        record = result["record"]

        explanation_column = self._find(
            data,
            ["Quantitative_WHY", "AI_Explanation", "Risk_Explanation"],
        )

        reason_column = self._find(
            data,
            ["Primary_Reason", "Primary_Concern"],
        )

        why_text = (
            record.get(explanation_column)
            if explanation_column
            else None
        )

        primary_reason = (
            record.get(reason_column)
            if reason_column
            else None
        )

        return {
            "status": "AVAILABLE",
            "component_id": self._text(component_id),
            "primary_reason": self._json_safe(primary_reason),
            "why": self._json_safe(why_text),
            "quantitative_evidence": result.get(
                "quantitative_evidence",
                {},
            ),
        }

    # ------------------------------------------------------------------
    # Data quality
    # ------------------------------------------------------------------

    def data_quality_status(
        self,
        data: pd.DataFrame,
    ) -> Dict[str, Any]:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")

        missing_cells = int(data.isna().sum().sum())
        duplicate_rows = int(data.duplicated().sum())

        component_col = self._find(
            data,
            ["Component_ID", "component_id", "Component"],
        )
        duplicate_components = 0

        if component_col:
            duplicate_components = int(
                data[component_col].duplicated().sum()
            )

        warning_columns = [
            column for column in data.columns
            if any(
                token in str(column).lower()
                for token in [
                    "warning",
                    "quality_flag",
                    "validation_error",
                    "validation_warning",
                ]
            )
        ]

        return {
            "status": (
                "PASS"
                if missing_cells == 0
                and duplicate_rows == 0
                else "REVIEW"
            ),
            "rows": int(len(data)),
            "columns": int(len(data.columns)),
            "missing_cells": missing_cells,
            "duplicate_rows": duplicate_rows,
            "duplicate_component_ids": duplicate_components,
            "warning_columns": warning_columns,
        }

    # ------------------------------------------------------------------
    # Complete dashboard payload
    # ------------------------------------------------------------------

    def build_dashboard_payload(
        self,
        data: pd.DataFrame,
        *,
        domain: str = "general_unknown",
        screening_status: str = "SCREENED",
        dataset_id: Any = None,
        model_version: Any = None,
        specification_version: Any = None,
        configuration_version: Any = None,
        data_quality: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build the complete UI-ready payload."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")

        quality = (
            data_quality
            if data_quality is not None
            else self.data_quality_status(data)
        )

        metrics = self.executive_metrics(
            data,
            domain=domain,
            screening_status=screening_status,
            dataset_id=dataset_id,
        )

        return {
            "dashboard_version": self.VERSION,
            "generated_at_utc": self._now_utc(),
            "metadata": {
                "dataset_id": self._text(dataset_id),
                "model_version": self._text(model_version),
                "specification_version": self._text(
                    specification_version
                ),
                "configuration_version": self._text(
                    configuration_version
                ),
            },
            "metrics": self._json_safe(metrics),
            "prediction": self._json_safe(
                self.prediction_status(data)
            ),
            "data_quality": self._json_safe(quality),
            "investigation_count": int(
                len(self.investigation_queue(data))
            ),
        }


__all__ = ["JudgeDashboard"]
