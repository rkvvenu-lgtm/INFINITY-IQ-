"""
SIH26170 - Report & Metadata Engine
Phase 3 / File 9

Application-layer reporting engine. It consumes an already-generated
screening DataFrame/result and produces judge-ready summaries and CSV/Excel
reports without changing the Phase 1 or Phase 2 AI engines.
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


class ReportEngine:
    """Generate screening summaries, metadata and downloadable reports."""

    VERSION = "3.0"

    def __init__(
        self,
        project_root: str | Path | None = None,
        report_directory: str | Path = "reports",
    ) -> None:
        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[1]
        )

        report_dir = Path(report_directory)
        self.report_directory = (
            report_dir
            if report_dir.is_absolute()
            else self.project_root / report_dir
        )
        self.report_directory.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_report_info(self) -> Dict[str, Any]:
        return {
            "module": "ReportEngine",
            "version": self.VERSION,
            "supported_formats": ["CSV", "XLSX", "JSON"],
            "report_directory": str(self.report_directory),
            "phase": "PHASE_3",
        }

    @staticmethod
    def _text(value: Any, default: str = "N/A") -> str:
        if value is None:
            return default
        text = str(value).strip()
        if text.lower() in {"", "nan", "none", "null"}:
            return default
        return text

    @staticmethod
    def _safe_number(value: Any) -> Optional[float]:
        try:
            value = pd.to_numeric(value, errors="coerce")
            if pd.isna(value):
                return None
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            if isinstance(value, float):
                if pd.isna(value) or value in (float("inf"), float("-inf")):
                    return None
            return value
        if isinstance(value, dict):
            return {
                str(k): ReportEngine._json_safe(v)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [ReportEngine._json_safe(v) for v in value]
        if hasattr(value, "item"):
            try:
                return ReportEngine._json_safe(value.item())
            except Exception:
                pass
        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except Exception:
                pass
        return str(value)

    @staticmethod
    def _now_utc() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    # ------------------------------------------------------------------
    # Column helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_column(
        data: pd.DataFrame,
        candidates: list[str],
    ) -> Optional[str]:
        for column in candidates:
            if column in data.columns:
                return column
        return None

    def _risk_column(self, data: pd.DataFrame) -> Optional[str]:
        return self._find_column(
            data,
            [
                "Overall_Risk_Percentage",
                "Overall_Risk_Score",
                "Risk_Score",
                "Risk",
            ],
        )

    def _risk_level_column(self, data: pd.DataFrame) -> Optional[str]:
        return self._find_column(
            data,
            ["Risk_Level", "Overall_Risk_Level"],
        )

    def _decision_column(self, data: pd.DataFrame) -> Optional[str]:
        return self._find_column(
            data,
            ["Risk_Decision", "Final_Decision", "Decision"],
        )

    def _anomaly_column(self, data: pd.DataFrame) -> Optional[str]:
        return self._find_column(
            data,
            ["Anomaly_Flag", "Anomaly_Risk"],
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def build_summary(
        self,
        data: pd.DataFrame,
        domain: str = "general_unknown",
        screening_status: str = "SCREENED",
        dataset_id: Any = None,
        dataset_version: Any = None,
        model_version: Any = None,
        specification_version: Any = None,
        configuration_version: Any = None,
        data_quality: Optional[Dict[str, Any]] = None,
        review_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a structured screening summary from an existing result."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")

        rows = int(len(data))
        columns = int(len(data.columns))

        risk_column = self._risk_column(data)
        risk_level_column = self._risk_level_column(data)
        decision_column = self._decision_column(data)
        anomaly_column = self._anomaly_column(data)

        risk_distribution: Dict[str, int] = {}
        if risk_level_column:
            values = (
                data[risk_level_column]
                .fillna("UNKNOWN")
                .astype(str)
                .str.upper()
            )
            risk_distribution = {
                str(key): int(value)
                for key, value in values.value_counts().to_dict().items()
            }

        decision_distribution: Dict[str, int] = {}
        if decision_column:
            values = (
                data[decision_column]
                .fillna("UNKNOWN")
                .astype(str)
                .str.upper()
            )
            decision_distribution = {
                str(key): int(value)
                for key, value in values.value_counts().to_dict().items()
            }

        anomaly_count = 0
        if anomaly_column:
            if anomaly_column == "Anomaly_Flag":
                values = data[anomaly_column]
                anomaly_count = int(
                    values.astype(str)
                    .str.strip()
                    .str.lower()
                    .isin(["true", "1", "yes"])
                    .sum()
                )
            else:
                numeric = pd.to_numeric(
                    data[anomaly_column],
                    errors="coerce",
                ).fillna(0)
                anomaly_count = int((numeric > 0).sum())

        predicted_columns = [
            column
            for column in data.columns
            if str(column).startswith("Predicted_")
            and str(column).endswith("168h")
        ]

        current_violation_count = 0
        predicted_violation_count = 0

        for column in data.columns:
            normalized = str(column).lower()

            if (
                "current" in normalized
                and "spec" in normalized
                and "violation" in normalized
            ):
                values = data[column].astype(str).str.upper()
                current_violation_count += int(
                    values.isin(["TRUE", "1", "YES", "VIOLATION", "REJECT"]).sum()
                )

            if (
                "predicted" in normalized
                and "spec" in normalized
                and "violation" in normalized
            ):
                values = data[column].astype(str).str.upper()
                predicted_violation_count += int(
                    values.isin(["TRUE", "1", "YES", "VIOLATION", "REJECT"]).sum()
                )

        risk_values = None
        if risk_column:
            risk_values = pd.to_numeric(
                data[risk_column],
                errors="coerce",
            ).dropna()

        risk_average = (
            float(risk_values.mean())
            if risk_values is not None and not risk_values.empty
            else None
        )

        risk_maximum = (
            float(risk_values.max())
            if risk_values is not None and not risk_values.empty
            else None
        )

        quality = data_quality or {}

        return {
            "report_version": self.VERSION,
            "generated_at_utc": self._now_utc(),
            "screening": {
                "status": self._text(screening_status),
                "domain": self._text(domain),
                "rows": rows,
                "columns": columns,
                "prediction_columns": predicted_columns,
                "prediction_count": len(predicted_columns),
                "anomaly_count": anomaly_count,
                "current_spec_violation_count": current_violation_count,
                "predicted_spec_violation_count": predicted_violation_count,
                "average_risk": risk_average,
                "maximum_risk": risk_maximum,
                "risk_distribution": risk_distribution,
                "decision_distribution": decision_distribution,
            },
            "dataset": {
                "dataset_id": self._text(dataset_id),
                "dataset_version": self._text(dataset_version),
            },
            "model": {
                "model_version": self._text(model_version),
            },
            "configuration": {
                "specification_version": self._text(specification_version),
                "configuration_version": self._text(configuration_version),
            },
            "data_quality": self._json_safe(quality),
            "engineering_review": self._json_safe(review_summary or {}),
        }

    # ------------------------------------------------------------------
    # Report metadata
    # ------------------------------------------------------------------

    def build_metadata(
        self,
        summary: Dict[str, Any],
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return a compact report metadata object."""
        metadata = {
            "report_engine_version": self.VERSION,
            "generated_at_utc": self._now_utc(),
            "dataset": summary.get("dataset", {}),
            "model": summary.get("model", {}),
            "configuration": summary.get("configuration", {}),
            "screening": summary.get("screening", {}),
        }

        if extra_metadata:
            metadata["additional_metadata"] = self._json_safe(
                extra_metadata
            )

        return metadata

    # ------------------------------------------------------------------
    # Data report preparation
    # ------------------------------------------------------------------

    def prepare_report_dataframe(
        self,
        data: pd.DataFrame,
        summary: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Return a copy of the screening result with report metadata columns.

        The original DataFrame is never modified.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")

        report = data.copy()

        if summary:
            screening = summary.get("screening", {})
            dataset = summary.get("dataset", {})
            model = summary.get("model", {})
            configuration = summary.get("configuration", {})

            metadata_columns = {
                "Report_Version": self.VERSION,
                "Report_Generated_UTC": summary.get(
                    "generated_at_utc",
                    self._now_utc(),
                ),
                "Report_Domain": screening.get("domain", "N/A"),
                "Dataset_ID": dataset.get("dataset_id", "N/A"),
                "Dataset_Version": dataset.get("dataset_version", "N/A"),
                "Model_Version": model.get("model_version", "N/A"),
                "Specification_Version": configuration.get(
                    "specification_version",
                    "N/A",
                ),
                "Configuration_Version": configuration.get(
                    "configuration_version",
                    "N/A",
                ),
            }

            for column, value in metadata_columns.items():
                report[column] = value

        return report

    # ------------------------------------------------------------------
    # CSV / Excel / JSON
    # ------------------------------------------------------------------

    def to_csv_bytes(
        self,
        data: pd.DataFrame,
    ) -> bytes:
        """Return the report DataFrame as UTF-8 CSV bytes."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")

        return data.to_csv(
            index=False,
            encoding="utf-8-sig",
        ).encode("utf-8-sig")

    def to_excel_bytes(
        self,
        data: pd.DataFrame,
        summary: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        """Return a multi-sheet XLSX report as bytes."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")

        output = io.BytesIO()

        with pd.ExcelWriter(
            output,
            engine="openpyxl",
        ) as writer:

            data.to_excel(
                writer,
                sheet_name="Screening_Result",
                index=False,
            )

            if summary:
                screening = summary.get("screening", {})
                metadata = self.build_metadata(summary)

                summary_rows = [
                    ["Report Version", self.VERSION],
                    ["Generated UTC", summary.get("generated_at_utc", "N/A")],
                    ["Domain", screening.get("domain", "N/A")],
                    ["Status", screening.get("status", "N/A")],
                    ["Rows", screening.get("rows", 0)],
                    ["Columns", screening.get("columns", 0)],
                    ["Predictions", screening.get("prediction_count", 0)],
                    ["Anomalies", screening.get("anomaly_count", 0)],
                    [
                        "Current Spec Violations",
                        screening.get(
                            "current_spec_violation_count",
                            0,
                        ),
                    ],
                    [
                        "Predicted Spec Violations",
                        screening.get(
                            "predicted_spec_violation_count",
                            0,
                        ),
                    ],
                    ["Dataset ID", summary.get("dataset", {}).get("dataset_id", "N/A")],
                    [
                        "Dataset Version",
                        summary.get("dataset", {}).get(
                            "dataset_version",
                            "N/A",
                        ),
                    ],
                    [
                        "Model Version",
                        summary.get("model", {}).get(
                            "model_version",
                            "N/A",
                        ),
                    ],
                    [
                        "Specification Version",
                        summary.get("configuration", {}).get(
                            "specification_version",
                            "N/A",
                        ),
                    ],
                    [
                        "Configuration Version",
                        summary.get("configuration", {}).get(
                            "configuration_version",
                            "N/A",
                        ),
                    ],
                ]

                pd.DataFrame(
                    summary_rows,
                    columns=["Metric", "Value"],
                ).to_excel(
                    writer,
                    sheet_name="Summary",
                    index=False,
                )

                risk_distribution = screening.get(
                    "risk_distribution",
                    {},
                )

                if risk_distribution:
                    pd.DataFrame(
                        list(risk_distribution.items()),
                        columns=["Risk_Level", "Count"],
                    ).to_excel(
                        writer,
                        sheet_name="Risk_Distribution",
                        index=False,
                    )

                decision_distribution = screening.get(
                    "decision_distribution",
                    {},
                )

                if decision_distribution:
                    pd.DataFrame(
                        list(decision_distribution.items()),
                        columns=["Decision", "Count"],
                    ).to_excel(
                        writer,
                        sheet_name="Decision_Distribution",
                        index=False,
                    )

                pd.DataFrame(
                    [
                        {
                            "Metadata": json.dumps(
                                self._json_safe(metadata),
                                ensure_ascii=False,
                            )
                        }
                    ]
                ).to_excel(
                    writer,
                    sheet_name="Metadata",
                    index=False,
                )

        output.seek(0)
        return output.getvalue()

    def to_json_bytes(
        self,
        summary: Dict[str, Any],
    ) -> bytes:
        """Return the structured summary as JSON bytes."""
        return json.dumps(
            self._json_safe(summary),
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")

    # ------------------------------------------------------------------
    # File persistence
    # ------------------------------------------------------------------

    def save_csv(
        self,
        data: pd.DataFrame,
        filename: str = "screening_report.csv",
    ) -> Path:
        path = self.report_directory / filename
        path.write_bytes(self.to_csv_bytes(data))
        return path

    def save_excel(
        self,
        data: pd.DataFrame,
        summary: Optional[Dict[str, Any]] = None,
        filename: str = "screening_report.xlsx",
    ) -> Path:
        path = self.report_directory / filename
        path.write_bytes(
            self.to_excel_bytes(
                data,
                summary=summary,
            )
        )
        return path

    def save_json(
        self,
        summary: Dict[str, Any],
        filename: str = "screening_summary.json",
    ) -> Path:
        path = self.report_directory / filename
        path.write_bytes(self.to_json_bytes(summary))
        return path


__all__ = ["ReportEngine"]
