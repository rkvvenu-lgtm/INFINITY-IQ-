"""
SIH26170 - Review & Reporting Integration Layer
Phase 3 / File 10

Application-layer integration module connecting screening results,
engineering review/audit records, and report generation.

This file does NOT modify the Phase 1 AI engine or Phase 2 screening
modules. It provides one stable interface for the final UI/report layer.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from modules.engineering_review import EngineeringReview
from modules.report_engine import ReportEngine


class ReviewReportIntegration:
    """Coordinate engineer review, audit information and reports."""

    VERSION = "3.0"

    def __init__(
        self,
        project_root: str | Path | None = None,
        audit_path: str | Path = "reports/engineering_review_audit.jsonl",
        report_directory: str | Path = "reports",
    ) -> None:
        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[1]
        )

        self.engineering_review = EngineeringReview(
            project_root=self.project_root,
            audit_path=audit_path,
        )

        self.report_engine = ReportEngine(
            project_root=self.project_root,
            report_directory=report_directory,
        )

    # ------------------------------------------------------------------
    # Basic information
    # ------------------------------------------------------------------

    @staticmethod
    def _now_utc() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _text(value: Any, default: str = "N/A") -> str:
        if value is None:
            return default
        text = str(value).strip()
        return default if text.lower() in {"", "nan", "none", "null"} else text

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
                str(k): ReviewReportIntegration._json_safe(v)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [
                ReviewReportIntegration._json_safe(v)
                for v in value
            ]
        if hasattr(value, "item"):
            try:
                return ReviewReportIntegration._json_safe(value.item())
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
            "module": "ReviewReportIntegration",
            "version": self.VERSION,
            "phase": "PHASE_3",
            "review_module": "EngineeringReview",
            "report_module": "ReportEngine",
            "supported_review_actions": [
                "ACCEPT",
                "MONITOR",
                "INVESTIGATE",
                "REJECT",
            ],
            "supported_report_formats": [
                "CSV",
                "XLSX",
                "JSON",
            ],
        }

    # ------------------------------------------------------------------
    # Review helpers
    # ------------------------------------------------------------------

    def record_review(
        self,
        action: str,
        reviewer: str,
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
        source: str = "SIH26170",
    ) -> Dict[str, Any]:
        """
        Record one engineering review through EngineeringReview.

        The exact EngineeringReview public method is resolved at runtime
        so this integration layer remains tolerant of small implementation
        differences while keeping one stable UI-facing API.
        """
        normalized_action = self._text(action, "").upper()

        if normalized_action not in {
            "ACCEPT",
            "MONITOR",
            "INVESTIGATE",
            "REJECT",
        }:
            raise ValueError(
                "Invalid review action. Use ACCEPT, MONITOR, "
                "INVESTIGATE or REJECT."
            )

        payload = {
            "action": normalized_action,
            "reviewer": self._text(reviewer),
            "comments": self._text(comments, ""),
            "justification": self._text(justification, ""),
            "component_id": self._text(component_id, ""),
            "lot_id": self._text(lot_id, ""),
            "dataset_id": self._text(dataset_id, ""),
            "dataset_version": self._text(dataset_version, ""),
            "model_version": self._text(model_version, ""),
            "specification_version": self._text(
                specification_version,
                "",
            ),
            "configuration_version": self._text(
                configuration_version,
                "",
            ),
            "risk_level": self._text(risk_level, ""),
            "risk_score": risk_score,
            "source": self._text(source, "SIH26170"),
            "timestamp_utc": self._now_utc(),
        }

        # EngineeringReview versions may expose one of these conventional
        # entry points. Prefer the explicit review method.
        for method_name in (
            "record_review",
            "add_review",
            "create_review",
            "submit_review",
        ):
            method = getattr(self.engineering_review, method_name, None)
            if callable(method):
                try:
                    result = method(**payload)
                    if isinstance(result, dict):
                        return self._json_safe(result)
                    return {
                        "status": "RECORDED",
                        "result": self._json_safe(result),
                        "review": self._json_safe(payload),
                    }
                except TypeError:
                    # Fall through to the next compatible public method.
                    continue

        # Stable fallback: if EngineeringReview exposes an audit writer,
        # use it rather than silently losing the review.
        for method_name in (
            "write_audit",
            "append_audit",
            "save_review",
        ):
            method = getattr(self.engineering_review, method_name, None)
            if callable(method):
                try:
                    result = method(payload)
                    return {
                        "status": "RECORDED",
                        "result": self._json_safe(result),
                        "review": self._json_safe(payload),
                    }
                except TypeError:
                    continue

        raise AttributeError(
            "EngineeringReview does not expose a compatible review "
            "recording method."
        )

    def get_review_summary(
        self,
        component_id: Any = None,
        lot_id: Any = None,
    ) -> Dict[str, Any]:
        """Retrieve the latest review/audit summary."""
        methods = (
            "review_summary",
            "get_review_summary",
            "summary",
        )

        for method_name in methods:
            method = getattr(self.engineering_review, method_name, None)
            if not callable(method):
                continue

            attempts = [
                {
                    "component_id": self._text(component_id, ""),
                    "lot_id": self._text(lot_id, ""),
                },
                {},
            ]

            for kwargs in attempts:
                try:
                    result = method(**kwargs)
                    if isinstance(result, dict):
                        return self._json_safe(result)
                except TypeError:
                    continue

        return {
            "status": "NO_REVIEW_DATA",
            "component_id": self._text(component_id, ""),
            "lot_id": self._text(lot_id, ""),
        }

    # ------------------------------------------------------------------
    # Screening integration
    # ------------------------------------------------------------------

    def build_integrated_report(
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
        """Build the complete structured report object."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")

        summary = self.report_engine.build_summary(
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
        )

        metadata = self.report_engine.build_metadata(
            summary,
            extra_metadata=extra_metadata,
        )

        return {
            "integration_version": self.VERSION,
            "generated_at_utc": self._now_utc(),
            "metadata": self._json_safe(metadata),
            "summary": self._json_safe(summary),
            "review": self._json_safe(
                review_summary or self.get_review_summary()
            ),
            "report_engine": self.report_engine.get_report_info(),
        }

    # ------------------------------------------------------------------
    # Final UI-ready tables
    # ------------------------------------------------------------------

    def build_review_table(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create a compact review queue without changing the source DataFrame.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")

        result = data.copy()

        preferred = [
            "Component_ID",
            "Lot_ID",
            "Component_Type",
            "Risk_Level",
            "Risk_Decision",
            "Final_Decision",
            "Overall_Risk_Score",
            "Overall_Risk_Percentage",
            "Anomaly_Risk",
            "Anomaly_Lot_Risk",
            "Drift_Risk",
            "Predicted_Iddq_168h",
            "Predicted_Leakage_168h",
            "Predicted_Delay_168h",
            "AI_Explanation",
            "Quantitative_WHY",
        ]

        available = [
            column
            for column in preferred
            if column in result.columns
        ]

        if available:
            return result[available].copy()

        return result

    def build_review_queue(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return records requiring human attention first."""
        review_table = self.build_review_table(data)

        if review_table.empty:
            return review_table

        decision_columns = [
            column
            for column in [
                "Risk_Decision",
                "Final_Decision",
                "Decision",
            ]
            if column in review_table.columns
        ]

        if not decision_columns:
            return review_table

        column = decision_columns[0]
        values = review_table[column].astype(str).str.upper()

        attention = values.isin(
            ["REVIEW", "INVESTIGATE", "REJECT", "MONITOR"]
        )

        return review_table.loc[attention].copy()

    # ------------------------------------------------------------------
    # Export orchestration
    # ------------------------------------------------------------------

    def export_csv(
        self,
        data: pd.DataFrame,
        *,
        summary: Optional[Dict[str, Any]] = None,
        filename: str = "sih26170_screening_report.csv",
    ) -> Path:
        report_data = self.report_engine.prepare_report_dataframe(
            data,
            summary=summary,
        )
        return self.report_engine.save_csv(
            report_data,
            filename=filename,
        )

    def export_excel(
        self,
        data: pd.DataFrame,
        *,
        summary: Optional[Dict[str, Any]] = None,
        filename: str = "sih26170_screening_report.xlsx",
    ) -> Path:
        report_data = self.report_engine.prepare_report_dataframe(
            data,
            summary=summary,
        )
        return self.report_engine.save_excel(
            report_data,
            summary=summary,
            filename=filename,
        )

    def export_json(
        self,
        integrated_report: Dict[str, Any],
        *,
        filename: str = "sih26170_screening_report.json",
    ) -> Path:
        path = self.report_engine.report_directory / filename
        path.write_text(
            json.dumps(
                self._json_safe(integrated_report),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    # ------------------------------------------------------------------
    # End-to-end convenience method
    # ------------------------------------------------------------------

    def generate_all(
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
        csv_filename: str = "sih26170_screening_report.csv",
        excel_filename: str = "sih26170_screening_report.xlsx",
        json_filename: str = "sih26170_screening_report.json",
        include_excel: bool = True,
    ) -> Dict[str, Any]:
        """Generate the structured report plus requested report files."""
        integrated = self.build_integrated_report(
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

        summary = integrated["summary"]

        csv_path = self.export_csv(
            data,
            summary=summary,
            filename=csv_filename,
        )

        output: Dict[str, Any] = {
            "status": "GENERATED",
            "integration_version": self.VERSION,
            "csv": str(csv_path),
            "json": None,
            "xlsx": None,
            "report": integrated,
        }

        output["json"] = str(
            self.export_json(
                integrated,
                filename=json_filename,
            )
        )

        if include_excel:
            output["xlsx"] = str(
                self.export_excel(
                    data,
                    summary=summary,
                    filename=excel_filename,
                )
            )

        return self._json_safe(output)


__all__ = ["ReviewReportIntegration"]
