"""
SIH26170 - Component Intelligence
Phase 2, File 5

Purpose:
    Convert row-level AI screening results into component-level
    engineering intelligence.

Architecture:
    Phase 1 Result
          |
          v
    Component Intelligence
          |
          +--> Current measurements
          +--> Anomaly status
          +--> Lot anomaly
          +--> Drift risk
          +--> 168h prediction
          +--> Specification status
          +--> Overall risk
          +--> Decision
          +--> Primary WHY
          |
          v
    Judge / Engineer UI

Design rule:
    This module operates above Phase 1.
    It does not modify any Phase 1 source files.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from models.model_manager import ModelManager
from modules.explainability_engine import ExplainabilityEngine
from modules.electronics_pipeline import ElectronicsPipeline


class ComponentIntelligence:
    """
    Component-level intelligence adapter.

    It converts the row-level AI output into a structured representation
    suitable for dashboards, reports and engineer review.
    """

    VERSION = "3.0"

    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        explainability_engine: Optional[ExplainabilityEngine] = None,
        electronics_pipeline: Optional[ElectronicsPipeline] = None,
    ) -> None:

        self.model_manager = model_manager or ModelManager()

        self.explainability_engine = (
            explainability_engine or ExplainabilityEngine()
        )

        self.electronics_pipeline = (
            electronics_pipeline or ElectronicsPipeline(
                model_manager=self.model_manager
            )
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_engine_info(self) -> Dict[str, Any]:
        """Return component intelligence metadata."""

        return {
            "component_intelligence_version": self.VERSION,
            "supported_parameters": self.electronics_pipeline.get_parameters(),
            "capabilities": [
                "current_measurements",
                "anomaly_status",
                "lot_anomaly",
                "drift_risk",
                "future_168h_prediction",
                "engineering_specification",
                "overall_risk",
                "decision",
                "quantitative_explanation",
            ],
            "phase": "PHASE_2",
            "phase_1_core_modified": False,
        }

    def analyze(
        self,
        data: pd.DataFrame,
        component_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze components from a Phase 1 result DataFrame.

        Parameters
        ----------
        data:
            DataFrame containing Phase 1 output.

        component_id:
            Optional component ID. If supplied, only that component
            is returned.

        Returns
        -------
        dict
            Structured component intelligence.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        working = data.copy()

        if component_id is not None:
            if "Component_ID" not in working.columns:
                raise ValueError(
                    "Component_ID column is required when component_id "
                    "is supplied."
                )

            working = working[
                working["Component_ID"].astype(str)
                == str(component_id)
            ].copy()

            if working.empty:
                raise ValueError(
                    f"Component '{component_id}' was not found."
                )

        components = [
            self._build_component_record(row)
            for _, row in working.iterrows()
        ]

        return {
            "version": self.VERSION,
            "component_count": len(components),
            "components": components,
            "phase_1_core_modified": False,
        }

    def analyze_component(
        self,
        row: pd.Series,
    ) -> Dict[str, Any]:
        """
        Build intelligence for one component row.
        """

        if not isinstance(row, pd.Series):
            raise TypeError("row must be a pandas Series")

        return self._build_component_record(row)

    def get_component(
        self,
        data: pd.DataFrame,
        component_id: str,
    ) -> Dict[str, Any]:
        """
        Return intelligence for one component.
        """

        result = self.analyze(
            data=data,
            component_id=component_id,
        )

        return result["components"][0]

    # ------------------------------------------------------------------
    # Component record
    # ------------------------------------------------------------------

    def _build_component_record(
        self,
        row: pd.Series,
    ) -> Dict[str, Any]:

        component_id = self._value(
            row,
            "Component_ID",
            default=None,
        )

        lot_id = self._value(
            row,
            "Lot_ID",
            default=None,
        )

        component_type = self._value(
            row,
            "Component_Type",
            default=None,
        )

        parameters = {}

        for parameter in self.electronics_pipeline.get_parameters():
            parameters[parameter] = self._build_parameter_record(
                row,
                parameter,
            )

        anomaly = self._build_anomaly_record(row)
        risk = self._build_risk_record(row)
        decision = self._build_decision_record(row)
        explanation = self._build_explanation_record(
            row=row,
            parameters=parameters,
        )

        return {
            "component_id": component_id,
            "lot_id": lot_id,
            "component_type": component_type,
            "parameters": parameters,
            "anomaly": anomaly,
            "risk": risk,
            "decision": decision,
            "explanation": explanation,
        }

    # ------------------------------------------------------------------
    # Parameter intelligence
    # ------------------------------------------------------------------

    def _build_parameter_record(
        self,
        row: pd.Series,
        parameter: str,
    ) -> Dict[str, Any]:

        profile = self.electronics_pipeline.get_parameter_profile(
            parameter
        )

        mapping = self.electronics_pipeline.get_parameter_mapping(
            parameter
        )

        limits = profile["engineering_limit"]

        current_values = {}

        for time_point, column in mapping["early"].items():
            if column is not None:
                current_values[time_point] = self._numeric_value(
                    row,
                    column,
                )

        future_96h = None

        if mapping["full"].get("96h") is not None:
            future_96h = self._numeric_value(
                row,
                mapping["full"]["96h"],
            )

        actual_168h = None

        if mapping["target"] in row.index:
            actual_168h = self._numeric_value(
                row,
                mapping["target"],
            )

        predicted_168h = None

        if profile["prediction_column"] in row.index:
            predicted_168h = self._numeric_value(
                row,
                profile["prediction_column"],
            )

        current_spec_status = self._find_current_spec_status(
            row,
            parameter,
        )

        predicted_spec_status = self._derive_predicted_spec_status(
            predicted_168h=predicted_168h,
            direction=profile["direction"],
            limits=limits,
        )

        drift_percentage = self._calculate_drift_percentage(
            current_values=current_values,
            predicted_168h=predicted_168h,
        )

        return {
            "display_name": profile["display_name"],
            "unit": profile["unit"],
            "direction": profile["direction"],
            "measurements": {
                "0h": current_values.get("0h"),
                "24h": current_values.get("24h"),
                "96h": future_96h,
                "168h_actual": actual_168h,
                "168h_predicted": predicted_168h,
            },
            "engineering_limit": limits,
            "current_spec_status": current_spec_status,
            "predicted_spec_status": predicted_spec_status,
            "drift_percentage": drift_percentage,
            "target_available": actual_168h is not None,
            "prediction_available": predicted_168h is not None,
        }

    # ------------------------------------------------------------------
    # Anomaly intelligence
    # ------------------------------------------------------------------

    @staticmethod
    def _build_anomaly_record(
        row: pd.Series,
    ) -> Dict[str, Any]:

        anomaly_risk = ComponentIntelligence._numeric_value(
            row,
            "Anomaly_Risk",
        )

        lot_anomaly = ComponentIntelligence._numeric_value(
            row,
            "Anomaly_Lot_Risk",
        )

        global_anomaly = ComponentIntelligence._numeric_value(
            row,
            "Anomaly_Global_Risk",
        )

        isolation_anomaly = ComponentIntelligence._numeric_value(
            row,
            "Anomaly_Isolation_Risk",
        )

        global_z = ComponentIntelligence._numeric_value(
            row,
            "Global_Z_Score",
        )

        lot_z = ComponentIntelligence._numeric_value(
            row,
            "Lot_Z_Score",
        )

        flag = ComponentIntelligence._value(
            row,
            "Anomaly_Flag",
            default=None,
        )

        level = ComponentIntelligence._value(
            row,
            "Anomaly_Level",
            default=None,
        )

        return {
            "risk": anomaly_risk,
            "global_risk": global_anomaly,
            "lot_risk": lot_anomaly,
            "isolation_risk": isolation_anomaly,
            "global_z_score": global_z,
            "lot_z_score": lot_z,
            "flag": flag,
            "level": level,
            "explanation": ComponentIntelligence._value(
                row,
                "Anomaly_Explanation",
                default=None,
            ),
        }

    # ------------------------------------------------------------------
    # Risk intelligence
    # ------------------------------------------------------------------

    @staticmethod
    def _build_risk_record(
        row: pd.Series,
    ) -> Dict[str, Any]:

        return {
            "overall_score": ComponentIntelligence._numeric_value(
                row,
                "Overall_Risk_Score",
            ),
            "percentage": ComponentIntelligence._numeric_value(
                row,
                "Overall_Risk_Percentage",
            ),
            "level": ComponentIntelligence._value(
                row,
                "Risk_Level",
                default=None,
            ),
            "contributors": ComponentIntelligence._value(
                row,
                "Risk_Contributors",
                default=None,
            ),
            "explanation": ComponentIntelligence._value(
                row,
                "Risk_Explanation",
                default=None,
            ),
            "drift_risk": ComponentIntelligence._numeric_value(
                row,
                "Drift_Risk",
            ),
        }

    # ------------------------------------------------------------------
    # Decision intelligence
    # ------------------------------------------------------------------

    @staticmethod
    def _build_decision_record(
        row: pd.Series,
    ) -> Dict[str, Any]:

        decision = ComponentIntelligence._value(
            row,
            "Risk_Decision",
            default="REVIEW",
        )

        if decision is None:
            decision = "REVIEW"

        return {
            "decision": str(decision).upper(),
            "automatic": True,
            "review_required": (
                str(decision).upper() == "REVIEW"
            ),
        }

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------

    def _build_explanation_record(
        self,
        row: pd.Series,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:

        ai_explanation = self._value(
            row,
            "AI_Explanation",
            default=None,
        )

        primary_reason = self._find_primary_reason(
            row,
            parameters,
        )

        return {
            "primary_reason": primary_reason,
            "ai_explanation": ai_explanation,
            "quantitative": self._build_quantitative_explanation(
                row,
                parameters,
            ),
        }

    def _build_quantitative_explanation(
        self,
        row: pd.Series,
        parameters: Dict[str, Any],
    ) -> List[Dict[str, Any]]:

        explanations: List[Dict[str, Any]] = []

        for parameter, information in parameters.items():

            measurements = information["measurements"]

            current_value = measurements.get("24h")

            predicted_value = measurements.get(
                "168h_predicted"
            )

            limit = information[
                "engineering_limit"
            ]

            explanations.append(
                {
                    "parameter": parameter,
                    "current_value": current_value,
                    "unit": information["unit"],
                    "direction": information["direction"],
                    "predicted_168h": predicted_value,
                    "engineering_limit": limit.get("max"),
                    "warning_limit": limit.get(
                        "warning_max"
                    ),
                    "drift_percentage": information[
                        "drift_percentage"
                    ],
                    "current_spec_status": information[
                        "current_spec_status"
                    ],
                    "predicted_spec_status": information[
                        "predicted_spec_status"
                    ],
                    "global_z_score": self._numeric_value(
                        row,
                        "Global_Z_Score",
                    ),
                    "lot_z_score": self._numeric_value(
                        row,
                        "Lot_Z_Score",
                    ),
                }
            )

        return explanations

    # ------------------------------------------------------------------
    # Specification helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_current_spec_status(
        row: pd.Series,
        parameter: str,
    ) -> Optional[str]:

        column = (
            f"Current_{parameter}_Spec_Status"
        )

        if column in row.index:
            value = row[column]

            if pd.notna(value):
                return str(value)

        return None

    @staticmethod
    def _derive_predicted_spec_status(
        predicted_168h: Optional[float],
        direction: str,
        limits: Dict[str, Any],
    ) -> str:

        if predicted_168h is None:
            return "UNKNOWN"

        direction = str(direction).upper()

        minimum = limits.get("min")
        warning_minimum = limits.get("warning_min")
        warning_maximum = limits.get("warning_max")
        maximum = limits.get("max")

        if direction == "HIGH":

            if maximum is not None and predicted_168h > maximum:
                return "VIOLATION"

            if (
                warning_maximum is not None
                and predicted_168h > warning_maximum
            ):
                return "WARNING"

            return "PASS"

        if direction == "LOW":

            if minimum is not None and predicted_168h < minimum:
                return "VIOLATION"

            if (
                warning_minimum is not None
                and predicted_168h < warning_minimum
            ):
                return "WARNING"

            return "PASS"

        if direction == "BOTH":

            if (
                minimum is not None
                and predicted_168h < minimum
            ):
                return "VIOLATION"

            if (
                maximum is not None
                and predicted_168h > maximum
            ):
                return "VIOLATION"

            if (
                warning_minimum is not None
                and predicted_168h < warning_minimum
            ):
                return "WARNING"

            if (
                warning_maximum is not None
                and predicted_168h > warning_maximum
            ):
                return "WARNING"

            return "PASS"

        return "REVIEW"

    # ------------------------------------------------------------------
    # Numeric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _numeric_value(
        row: pd.Series,
        column: str,
    ) -> Optional[float]:

        if column not in row.index:
            return None

        value = row[column]

        if pd.isna(value):
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _value(
        row: pd.Series,
        column: str,
        default: Any = None,
    ) -> Any:

        if column not in row.index:
            return default

        value = row[column]

        if pd.isna(value):
            return default

        return value

    @staticmethod
    def _calculate_drift_percentage(
        current_values: Dict[str, Optional[float]],
        predicted_168h: Optional[float],
    ) -> Optional[float]:

        if predicted_168h is None:
            return None

        baseline = current_values.get("24h")

        if baseline is None:
            baseline = current_values.get("0h")

        if baseline is None or baseline == 0:
            return None

        return float(
            ((predicted_168h - baseline) / abs(baseline))
            * 100.0
        )

    @staticmethod
    def _find_primary_reason(
        row: pd.Series,
        parameters: Dict[str, Any],
    ) -> str:

        decision = str(
            ComponentIntelligence._value(
                row,
                "Risk_Decision",
                default="REVIEW",
            )
        ).upper()

        if decision == "REJECT":
            for parameter, information in parameters.items():
                if (
                    information["predicted_spec_status"]
                    == "VIOLATION"
                ):
                    return (
                        f"{information['display_name']} "
                        f"is predicted to violate the "
                        f"168h engineering limit."
                    )

            return "Risk assessment indicates rejection."

        if decision == "REVIEW":

            anomaly_level = ComponentIntelligence._value(
                row,
                "Anomaly_Level",
                default=None,
            )

            if anomaly_level is not None:
                return (
                    f"Anomaly level is {str(anomaly_level).upper()} "
                    f"and requires engineering review."
                )

            return "Risk assessment requires engineering review."

        for parameter, information in parameters.items():
            if (
                information["predicted_spec_status"]
                == "WARNING"
            ):
                return (
                    f"{information['display_name']} "
                    f"shows predicted movement toward "
                    f"its engineering limit."
                )

        return "No primary engineering concern identified."

    # ------------------------------------------------------------------
    # DataFrame conversion
    # ------------------------------------------------------------------

    def to_dataframe(
        self,
        intelligence: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Flatten component-level intelligence into a dashboard-friendly
        DataFrame.
        """

        records: List[Dict[str, Any]] = []

        for component in intelligence.get(
            "components",
            [],
        ):

            record: Dict[str, Any] = {
                "Component_ID": component["component_id"],
                "Lot_ID": component["lot_id"],
                "Component_Type": component[
                    "component_type"
                ],
                "Overall_Risk_Score": component[
                    "risk"
                ]["overall_score"],
                "Overall_Risk_Percentage": component[
                    "risk"
                ]["percentage"],
                "Risk_Level": component["risk"]["level"],
                "Risk_Decision": component[
                    "decision"
                ]["decision"],
                "Anomaly_Risk": component[
                    "anomaly"
                ]["risk"],
                "Lot_Anomaly_Risk": component[
                    "anomaly"
                ]["lot_risk"],
                "Primary_Reason": component[
                    "explanation"
                ]["primary_reason"],
            }

            for parameter, information in component[
                "parameters"
            ].items():

                prefix = parameter

                measurements = information[
                    "measurements"
                ]

                record[f"{prefix}_0h"] = measurements.get(
                    "0h"
                )

                record[f"{prefix}_24h"] = measurements.get(
                    "24h"
                )

                record[f"{prefix}_96h"] = measurements.get(
                    "96h"
                )

                record[
                    f"{prefix}_168h_Predicted"
                ] = measurements.get(
                    "168h_predicted"
                )

                record[
                    f"{prefix}_Current_Spec"
                ] = information[
                    "current_spec_status"
                ]

                record[
                    f"{prefix}_Predicted_Spec"
                ] = information[
                    "predicted_spec_status"
                ]

                record[
                    f"{prefix}_Drift_Percentage"
                ] = information[
                    "drift_percentage"
                ]

            records.append(record)

        return pd.DataFrame(records)


if __name__ == "__main__":
    engine = ComponentIntelligence()

    print("ComponentIntelligence : PASS")
    print("Version:", engine.VERSION)
    print(
        "Parameters:",
        engine.electronics_pipeline.get_parameters(),
    )