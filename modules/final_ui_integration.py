"""
SIH26170 - Final UI Integration Layer
Phase 3 / File 12

Stable Streamlit-facing integration layer for the locked SIH26170 modules.

Design rule:
    Phase 1 and Phase 2 modules are consumed, never modified.

This layer provides:
    - screening-result normalization
    - judge dashboard payload generation
    - investigation queue
    - component WHY view
    - review submission
    - report generation
    - downloadable CSV / XLSX / JSON bytes

The existing user_app.py can call this layer without duplicating business
logic. The module itself does not require Streamlit, so it can also be
verified independently.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from modules.judge_dashboard import JudgeDashboard
from modules.review_report_integration import ReviewReportIntegration


class FinalUIIntegration:
    """Single UI-facing facade over the locked Phase 3 modules."""

    VERSION = "3.0"

    def __init__(
        self,
        project_root: str | Path | None = None,
        report_directory: str | Path = "reports",
        audit_path: str | Path = "reports/engineering_review_audit.jsonl",
    ) -> None:
        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[1]
        )

        self.dashboard = JudgeDashboard(
            project_root=self.project_root,
        )

        self.review_reports = ReviewReportIntegration(
            project_root=self.project_root,
            report_directory=report_directory,
            audit_path=audit_path,
        )

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

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
    def _json_safe(value: Any) -> Any:
        if value is None:
            return None

        if isinstance(value, (str, int, float, bool)):
            if isinstance(value, float) and pd.isna(value):
                return None
            return value

        if isinstance(value, dict):
            return {
                str(k): FinalUIIntegration._json_safe(v)
                for k, v in value.items()
            }

        if isinstance(value, (list, tuple, set)):
            return [
                FinalUIIntegration._json_safe(v)
                for v in value
            ]

        if hasattr(value, "item"):
            try:
                return FinalUIIntegration._json_safe(value.item())
            except Exception:
                pass

        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except Exception:
                pass

        return str(value)

    @staticmethod
    def _require_dataframe(data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("screening result must be a pandas DataFrame.")

    # ------------------------------------------------------------------
    # Module information
    # ------------------------------------------------------------------

    def get_info(self) -> Dict[str, Any]:
        return {
            "module": "FinalUIIntegration",
            "version": self.VERSION,
            "phase": "PHASE_3",
            "purpose": "Final UI facade for judge/product presentation",
            "phase_1_modified": False,
            "phase_2_modified": False,
            "capabilities": [
                "dashboard",
                "investigation",
                "quantitative_why",
                "engineering_review",
                "audit",
                "csv_export",
                "xlsx_export",
                "json_export",
            ],
        }

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def dashboard_payload(
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
        self._require_dataframe(data)

        return self._json_safe(
            self.dashboard.build_dashboard_payload(
                data,
                domain=domain,
                screening_status=screening_status,
                dataset_id=dataset_id,
                model_version=model_version,
                specification_version=specification_version,
                configuration_version=configuration_version,
                data_quality=data_quality,
            )
        )

    def executive_metrics(
        self,
        data: pd.DataFrame,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        self._require_dataframe(data)
        return self._json_safe(
            self.dashboard.executive_metrics(
                data,
                **kwargs,
            )
        )

    def prediction_status(
        self,
        data: pd.DataFrame,
    ) -> Dict[str, Any]:
        self._require_dataframe(data)
        return self._json_safe(
            self.dashboard.prediction_status(data)
        )

    # ------------------------------------------------------------------
    # Investigation
    # ------------------------------------------------------------------

    def investigation_queue(
        self,
        data: pd.DataFrame,
        limit: int = 50,
    ) -> pd.DataFrame:
        self._require_dataframe(data)
        return self.dashboard.investigation_queue(
            data,
            limit=limit,
        )

    def component_drilldown(
        self,
        data: pd.DataFrame,
        component_id: Any,
    ) -> Dict[str, Any]:
        self._require_dataframe(data)
        return self._json_safe(
            self.dashboard.component_drilldown(
                data,
                component_id,
            )
        )

    def why_view(
        self,
        data: pd.DataFrame,
        component_id: Any,
    ) -> Dict[str, Any]:
        self._require_dataframe(data)
        return self._json_safe(
            self.dashboard.why_view(
                data,
                component_id,
            )
        )

    # ------------------------------------------------------------------
    # Engineering review
    # ------------------------------------------------------------------

    def submit_review(
        self,
        action: str,
        reviewer: str,
        *,
        comments: str = "",
        justification: str = "",
        component_id: Any = None,
        lot_id: Any = None,
        dataset_id: Any = None,
        dataset_version: Any = None,
        model_version: Any = None,
        specification_version: Any = None,
        configuration_version: Any = None,
        risk_level: Any = None,
        risk_score: Any = None,
    ) -> Dict[str, Any]:
        result = self.review_reports.record_review(
            action=action,
            reviewer=reviewer,
            comments=comments,
            justification=justification,
            component_id=component_id,
            lot_id=lot_id,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            model_version=model_version,
            specification_version=specification_version,
            configuration_version=configuration_version,
            risk_level=risk_level,
            risk_score=risk_score,
        )

        return self._json_safe(result)

    def review_summary(
        self,
        component_id: Any = None,
        lot_id: Any = None,
    ) -> Dict[str, Any]:
        return self._json_safe(
            self.review_reports.get_review_summary(
                component_id=component_id,
                lot_id=lot_id,
            )
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def integrated_report(
        self,
        data: pd.DataFrame,
        *,
        domain: str = "general_unknown",
        screening_status: str = "SCREENED",
        dataset_id: Any = None,
        dataset_version: Any = None,
        model_version: Any = None,
        specification_version: Any = None,
        configuration_version: Any = None,
        data_quality: Optional[Dict[str, Any]] = None,
        review_summary: Optional[Dict[str, Any]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._require_dataframe(data)

        return self._json_safe(
            self.review_reports.build_integrated_report(
                data,
                domain=domain,
                screening_status=screening_status,
                dataset_id=dataset_id,
                dataset_version=dataset_version,
                model_version=model_version,
                specification_version=specification_version,
                configuration_version=configuration_version,
                data_quality=data_quality,
                review_summary=review_summary,
                extra_metadata=extra_metadata,
            )
        )

    def export_csv_bytes(
        self,
        data: pd.DataFrame,
        summary: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        self._require_dataframe(data)

        prepared = self.review_reports.report_engine.prepare_report_dataframe(
            data,
            summary=summary,
        )

        return self.review_reports.report_engine.to_csv_bytes(
            prepared,
        )

    def export_excel_bytes(
        self,
        data: pd.DataFrame,
        summary: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        self._require_dataframe(data)

        prepared = self.review_reports.report_engine.prepare_report_dataframe(
            data,
            summary=summary,
        )

        return self.review_reports.report_engine.to_excel_bytes(
            prepared,
            summary=summary,
        )

    def export_json_bytes(
        self,
        integrated_report: Dict[str, Any],
    ) -> bytes:
        return self.review_reports.report_engine.to_json_bytes(
            integrated_report,
        )

    # ------------------------------------------------------------------
    # Complete UI state
    # ------------------------------------------------------------------

    def build_ui_state(
        self,
        data: pd.DataFrame,
        *,
        domain: str = "general_unknown",
        screening_status: str = "SCREENED",
        dataset_id: Any = None,
        dataset_version: Any = None,
        model_version: Any = None,
        specification_version: Any = None,
        configuration_version: Any = None,
        data_quality: Optional[Dict[str, Any]] = None,
        investigation_limit: int = 50,
    ) -> Dict[str, Any]:
        """
        Build one serializable object suitable for st.session_state.

        The source DataFrame is never modified.
        """
        self._require_dataframe(data)

        payload = self.dashboard_payload(
            data,
            domain=domain,
            screening_status=screening_status,
            dataset_id=dataset_id,
            model_version=model_version,
            specification_version=specification_version,
            configuration_version=configuration_version,
            data_quality=data_quality,
        )

        integrated = self.integrated_report(
            data,
            domain=domain,
            screening_status=screening_status,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            model_version=model_version,
            specification_version=specification_version,
            configuration_version=configuration_version,
            data_quality=data_quality,
        )

        queue = self.investigation_queue(
            data,
            limit=investigation_limit,
        )

        return {
            "ui_version": self.VERSION,
            "generated_at_utc": self._now_utc(),
            "dashboard": payload,
            "prediction": self.prediction_status(data),
            "data_quality": (
                data_quality
                if data_quality is not None
                else self.dashboard.data_quality_status(data)
            ),
            "investigation_count": int(len(queue)),
            "integrated_report": integrated,
        }


__all__ = ["FinalUIIntegration"]
