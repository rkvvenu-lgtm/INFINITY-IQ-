from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class ExplainabilityEngine:
    """
    Quantitative explanation engine for AI risk decisions.

    Generates numerical, engineer-readable explanations from
    existing pipeline outputs.

    It does not train models and does not modify predictions.
    """

    ENGINE_VERSION = "3.0"

    def __init__(
        self,
        lot_group_columns: Optional[list[str]] = None,
    ):
        self.lot_group_columns = (
            lot_group_columns
            if lot_group_columns is not None
            else [
                "Lot_ID",
                "Component_Type",
            ]
        )

    # ============================================================
    # BASIC NUMERIC HELPERS
    # ============================================================

    @staticmethod
    def _to_float(
        value: Any,
    ) -> Optional[float]:

        try:
            numeric = float(value)

            if np.isfinite(numeric):
                return numeric

        except (
            TypeError,
            ValueError,
        ):
            pass

        return None

    @staticmethod
    def _safe_round(
        value: Any,
        digits: int = 4,
    ) -> Optional[float]:

        numeric = ExplainabilityEngine._to_float(
            value
        )

        if numeric is None:
            return None

        return round(
            numeric,
            digits,
        )

    # ============================================================
    # LOT STATISTICS
    # ============================================================

    def calculate_lot_statistics(
        self,
        data: pd.DataFrame,
        value_column: str,
        group_columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Calculate lot-relative mean, median and standard
        deviation for a parameter.
        """

        if value_column not in data.columns:
            raise ValueError(
                f"Column '{value_column}' not found."
            )

        groups = (
            group_columns
            if group_columns is not None
            else self.lot_group_columns
        )

        valid_groups = [
            column
            for column in groups
            if column in data.columns
        ]

        output = data.copy()

        numeric_values = pd.to_numeric(
            output[value_column],
            errors="coerce",
        )

        if not valid_groups:

            mean_value = numeric_values.mean()
            median_value = numeric_values.median()
            std_value = numeric_values.std(
                ddof=0
            )

            output[
                "Lot_Mean"
            ] = mean_value

            output[
                "Lot_Median"
            ] = median_value

            output[
                "Lot_Std"
            ] = std_value

        else:

            temporary = output.copy()

            temporary[
                "__numeric_value"
            ] = numeric_values

            grouped = temporary.groupby(
                valid_groups,
                dropna=False,
            )[
                "__numeric_value"
            ]

            output[
                "Lot_Mean"
            ] = grouped.transform(
                "mean"
            )

            output[
                "Lot_Median"
            ] = grouped.transform(
                "median"
            )

            output[
                "Lot_Std"
            ] = grouped.transform(
                lambda values: values.std(
                    ddof=0
                )
            )

        return output

    # ============================================================
    # Z-SCORE
    # ============================================================

    @staticmethod
    def calculate_z_score(
        value: Any,
        mean: Any,
        std: Any,
    ) -> Optional[float]:

        value_num = (
            ExplainabilityEngine._to_float(
                value
            )
        )

        mean_num = (
            ExplainabilityEngine._to_float(
                mean
            )
        )

        std_num = (
            ExplainabilityEngine._to_float(
                std
            )
        )

        if (
            value_num is None
            or mean_num is None
            or std_num is None
            or std_num <= 0
        ):
            return None

        return (
            value_num - mean_num
        ) / std_num

    # ============================================================
    # DRIFT CALCULATION
    # ============================================================

    @staticmethod
    def calculate_drift_percentage(
        current_value: Any,
        predicted_value: Any,
    ) -> Optional[float]:

        current = (
            ExplainabilityEngine._to_float(
                current_value
            )
        )

        predicted = (
            ExplainabilityEngine._to_float(
                predicted_value
            )
        )

        if (
            current is None
            or predicted is None
        ):
            return None

        if abs(current) < 1e-12:
            return None

        return (
            (
                predicted - current
            )
            / abs(current)
        ) * 100.0

    # ============================================================
    # SINGLE EXPLANATION
    # ============================================================

    def explain(
        self,
        parameter: str,
        current_value: Any = None,
        lot_mean: Any = None,
        lot_median: Any = None,
        lot_std: Any = None,
        anomaly_risk: Any = 0.0,
        lot_anomaly_risk: Any = 0.0,
        predicted_value: Any = None,
        engineering_limit: Any = None,
        warning_limit: Any = None,
        direction: str = "HIGH",
        current_spec_status: str = "PASS",
        predicted_spec_status: str = "PASS",
        risk_score: Any = 0.0,
        risk_level: str = "LOW",
        decision: str = "PASS",
    ) -> Dict[str, Any]:
        """
        Generate a quantitative explanation for one parameter.
        """

        current = self._to_float(
            current_value
        )

        predicted = self._to_float(
            predicted_value
        )

        mean = self._to_float(
            lot_mean
        )

        median = self._to_float(
            lot_median
        )

        std = self._to_float(
            lot_std
        )

        z_score = self.calculate_z_score(
            current,
            mean,
            std,
        )

        drift_percentage = (
            self.calculate_drift_percentage(
                current,
                predicted,
            )
        )

        anomaly = self._safe_round(
            anomaly_risk
        )

        lot_anomaly = self._safe_round(
            lot_anomaly_risk
        )

        risk = self._safe_round(
            risk_score
        )

        limit = self._to_float(
            engineering_limit
        )

        warning = self._to_float(
            warning_limit
        )

        explanation_points = []

        # --------------------------------------------------------
        # Current value
        # --------------------------------------------------------

        if current is not None:
            explanation_points.append(
                f"Current {parameter} = "
                f"{current:.4f}"
            )

        # --------------------------------------------------------
        # Lot comparison
        # --------------------------------------------------------

        if (
            current is not None
            and mean is not None
            and z_score is not None
        ):

            explanation_points.append(
                f"lot mean = {mean:.4f}, "
                f"z-score = {z_score:.2f}"
            )

        elif (
            current is not None
            and mean is not None
        ):

            explanation_points.append(
                f"lot mean = {mean:.4f}"
            )

        # --------------------------------------------------------
        # Anomaly
        # --------------------------------------------------------

        if anomaly is not None and anomaly > 0:
            explanation_points.append(
                f"global anomaly contribution = "
                f"{anomaly:.2f}"
            )

        if (
            lot_anomaly is not None
            and lot_anomaly > 0
        ):
            explanation_points.append(
                f"lot-relative anomaly contribution = "
                f"{lot_anomaly:.2f}"
            )

        # --------------------------------------------------------
        # Future prediction
        # --------------------------------------------------------

        if predicted is not None:

            explanation_points.append(
                f"predicted 168h value = "
                f"{predicted:.4f}"
            )

        if drift_percentage is not None:

            explanation_points.append(
                f"projected drift = "
                f"{drift_percentage:.2f}%"
            )

        # --------------------------------------------------------
        # Engineering limits
        # --------------------------------------------------------

        if warning is not None:

            explanation_points.append(
                f"warning threshold = "
                f"{warning:.4f}"
            )

        if limit is not None:

            explanation_points.append(
                f"engineering limit = "
                f"{limit:.4f}"
            )

        # --------------------------------------------------------
        # Specification result
        # --------------------------------------------------------

        explanation_points.append(
            f"current specification = "
            f"{str(current_spec_status).upper()}"
        )

        explanation_points.append(
            f"predicted specification = "
            f"{str(predicted_spec_status).upper()}"
        )

        # --------------------------------------------------------
        # Final risk
        # --------------------------------------------------------

        if risk is not None:

            explanation_points.append(
                f"overall risk = "
                f"{risk * 100.0:.2f}%"
            )

        explanation_points.append(
            f"risk level = "
            f"{str(risk_level).upper()}"
        )

        explanation_points.append(
            f"decision = "
            f"{str(decision).upper()}"
        )

        # --------------------------------------------------------
        # Primary reason
        # --------------------------------------------------------

        primary_reason = (
            self._primary_reason(
                current=current,
                predicted=predicted,
                z_score=z_score,
                limit=limit,
                warning=warning,
                direction=direction,
                current_spec_status=(
                    current_spec_status
                ),
                predicted_spec_status=(
                    predicted_spec_status
                ),
                anomaly=anomaly,
                lot_anomaly=lot_anomaly,
            )
        )

        return {
            "engine_version": (
                self.ENGINE_VERSION
            ),
            "parameter": parameter,
            "current_value": (
                self._safe_round(current)
            ),
            "lot_mean": (
                self._safe_round(mean)
            ),
            "lot_median": (
                self._safe_round(median)
            ),
            "lot_std": (
                self._safe_round(std)
            ),
            "z_score": (
                self._safe_round(z_score)
            ),
            "anomaly_contribution": anomaly,
            "lot_anomaly_contribution": (
                lot_anomaly
            ),
            "predicted_168h": (
                self._safe_round(predicted)
            ),
            "drift_percentage": (
                self._safe_round(
                    drift_percentage,
                    digits=2,
                )
            ),
            "warning_limit": (
                self._safe_round(warning)
            ),
            "engineering_limit": (
                self._safe_round(limit)
            ),
            "direction": str(
                direction
            ).upper(),
            "current_spec_status": str(
                current_spec_status
            ).upper(),
            "predicted_spec_status": str(
                predicted_spec_status
            ).upper(),
            "risk_score": risk,
            "risk_level": str(
                risk_level
            ).upper(),
            "decision": str(
                decision
            ).upper(),
            "primary_reason": primary_reason,
            "explanation": (
                ". ".join(
                    explanation_points
                )
                + "."
            ),
        }

    # ============================================================
    # PRIMARY REASON
    # ============================================================

    @staticmethod
    def _primary_reason(
        current: Optional[float],
        predicted: Optional[float],
        z_score: Optional[float],
        limit: Optional[float],
        warning: Optional[float],
        direction: str,
        current_spec_status: str,
        predicted_spec_status: str,
        anomaly: Optional[float],
        lot_anomaly: Optional[float],
    ) -> str:

        current_status = str(
            current_spec_status
        ).upper()

        predicted_status = str(
            predicted_spec_status
        ).upper()

        if predicted_status == "FAIL":
            return (
                "Predicted 168h value violates "
                "the engineering limit."
            )

        if current_status == "FAIL":
            return (
                "Current value violates "
                "the engineering limit."
            )

        if (
            z_score is not None
            and abs(z_score) >= 3.0
        ):
            return (
                "Current value is statistically "
                "abnormal relative to its lot."
            )

        if (
            lot_anomaly is not None
            and lot_anomaly >= 0.5
        ):
            return (
                "Lot-relative behaviour is a "
                "significant risk contributor."
            )

        if (
            anomaly is not None
            and anomaly >= 0.5
        ):
            return (
                "Global anomaly detection identifies "
                "unusual behaviour."
            )

        if (
            predicted is not None
            and limit is not None
        ):

            if direction == "HIGH":
                if predicted > limit:
                    return (
                        "Predicted value crosses "
                        "the engineering limit."
                    )

                if (
                    warning is not None
                    and predicted > warning
                ):
                    return (
                        "Predicted value enters "
                        "the engineering warning region."
                    )

            elif direction == "LOW":

                if predicted < limit:
                    return (
                        "Predicted value crosses "
                        "the engineering limit."
                    )

                if (
                    warning is not None
                    and predicted < warning
                ):
                    return (
                        "Predicted value enters "
                        "the engineering warning region."
                    )

        if (
            predicted is not None
            and current is not None
        ):

            if direction == "HIGH" and predicted > current:
                return (
                    "Future value shows upward drift."
                )

            if direction == "LOW" and predicted < current:
                return (
                    "Future value shows downward drift."
                )

        return (
            "No dominant engineering or statistical "
            "risk contributor was detected."
        )

    # ============================================================
    # DATAFRAME EXPLANATIONS
    # ============================================================

    def explain_dataframe(
        self,
        data: pd.DataFrame,
        parameter: str,
        current_column: str,
        predicted_column: Optional[str] = None,
        anomaly_column: str = "Anomaly_Risk",
        lot_anomaly_column: str = "Lot_Risk",
        current_status_column: Optional[str] = None,
        predicted_status_column: Optional[str] = None,
        risk_score_column: str = "Overall_Risk_Score",
        risk_level_column: str = "Risk_Level",
        decision_column: str = "Risk_Decision",
        direction: str = "HIGH",
        engineering_limit: Any = None,
        warning_limit: Any = None,
    ) -> pd.DataFrame:
        """
        Add quantitative explanations to a dataframe.
        """

        if current_column not in data.columns:
            raise ValueError(
                f"Column '{current_column}' not found."
            )

        output = self.calculate_lot_statistics(
            data,
            current_column,
        )

        explanations = []

        for index in output.index:

            current = output.loc[
                index,
                current_column,
            ]

            predicted = None

            if (
                predicted_column is not None
                and predicted_column
                in output.columns
            ):
                predicted = output.loc[
                    index,
                    predicted_column,
                ]

            anomaly = (
                output.loc[
                    index,
                    anomaly_column,
                ]
                if anomaly_column
                in output.columns
                else 0.0
            )

            lot_anomaly = (
                output.loc[
                    index,
                    lot_anomaly_column,
                ]
                if lot_anomaly_column
                in output.columns
                else 0.0
            )

            current_status = (
                output.loc[
                    index,
                    current_status_column,
                ]
                if (
                    current_status_column
                    and current_status_column
                    in output.columns
                )
                else "PASS"
            )

            predicted_status = (
                output.loc[
                    index,
                    predicted_status_column,
                ]
                if (
                    predicted_status_column
                    and predicted_status_column
                    in output.columns
                )
                else "PASS"
            )

            risk_score = (
                output.loc[
                    index,
                    risk_score_column,
                ]
                if risk_score_column
                in output.columns
                else 0.0
            )

            risk_level = (
                output.loc[
                    index,
                    risk_level_column,
                ]
                if risk_level_column
                in output.columns
                else "LOW"
            )

            decision = (
                output.loc[
                    index,
                    decision_column,
                ]
                if decision_column
                in output.columns
                else "PASS"
            )

            explanation = self.explain(
                parameter=parameter,
                current_value=current,
                lot_mean=output.loc[
                    index,
                    "Lot_Mean",
                ],
                lot_median=output.loc[
                    index,
                    "Lot_Median",
                ],
                lot_std=output.loc[
                    index,
                    "Lot_Std",
                ],
                anomaly_risk=anomaly,
                lot_anomaly_risk=lot_anomaly,
                predicted_value=predicted,
                engineering_limit=(
                    engineering_limit
                ),
                warning_limit=(
                    warning_limit
                ),
                direction=direction,
                current_spec_status=(
                    current_status
                ),
                predicted_spec_status=(
                    predicted_status
                ),
                risk_score=risk_score,
                risk_level=risk_level,
                decision=decision,
            )

            explanations.append(
                explanation
            )

        output[
            "AI_Explanation"
        ] = [
            item["explanation"]
            for item in explanations
        ]

        output[
            "AI_Primary_Reason"
        ] = [
            item["primary_reason"]
            for item in explanations
        ]

        output[
            "AI_Z_Score"
        ] = [
            item["z_score"]
            for item in explanations
        ]

        output[
            "AI_Drift_Percentage"
        ] = [
            item["drift_percentage"]
            for item in explanations
        ]

        return output

    # ============================================================
    # ENGINE INFORMATION
    # ============================================================

    def get_engine_info(
        self,
    ) -> Dict[str, Any]:

        return {
            "engine_version": (
                self.ENGINE_VERSION
            ),
            "lot_group_columns": (
                self.lot_group_columns.copy()
            ),
            "capabilities": [
                "lot_mean",
                "lot_median",
                "z_score",
                "anomaly_contribution",
                "lot_anomaly_contribution",
                "drift_percentage",
                "predicted_168h",
                "engineering_limit",
                "primary_reason",
                "quantitative_explanation",
            ],
        }