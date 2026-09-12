from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class RiskFusion:
    """
    Multi-signal engineering risk fusion engine.

    Combines:
        - Global anomaly risk
        - Lot-relative anomaly risk
        - Predicted drift risk
        - Current specification status
        - Predicted 168h specification status

    Output:
        - Risk score
        - Risk level
        - Decision
        - Risk contributors
        - Explanation

    This module does not train or predict models.
    """

    FUSION_VERSION = "3.0"

    def __init__(
        self,
        anomaly_weight: float = 0.30,
        lot_anomaly_weight: float = 0.20,
        drift_weight: float = 0.20,
        current_spec_weight: float = 0.15,
        predicted_spec_weight: float = 0.15,
    ):
        weights = {
            "anomaly": anomaly_weight,
            "lot_anomaly": lot_anomaly_weight,
            "drift": drift_weight,
            "current_spec": current_spec_weight,
            "predicted_spec": predicted_spec_weight,
        }

        total = sum(weights.values())

        if total <= 0:
            raise ValueError(
                "Risk-fusion weights must have "
                "a positive total."
            )

        self.weights = {
            key: float(value) / total
            for key, value in weights.items()
        }

    # ============================================================
    # NORMALIZATION
    # ============================================================

    @staticmethod
    def _clip_score(
        value: Any,
    ) -> float:

        try:
            numeric = float(value)
        except (
            TypeError,
            ValueError,
        ):
            numeric = 0.0

        return float(
            np.clip(
                numeric,
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _status_score(
        status: Any,
    ) -> float:

        normalized = str(
            status
        ).strip().upper()

        mapping = {
            "PASS": 0.0,
            "WARNING": 0.5,
            "FAIL": 1.0,
            "REVIEW": 0.75,
            "INVALID": 1.0,
            "UNKNOWN": 0.75,
        }

        return mapping.get(
            normalized,
            0.0,
        )

    # ============================================================
    # DRIFT SCORE
    # ============================================================

    @staticmethod
    def drift_score(
        current_value: Any,
        predicted_value: Any,
        direction: str = "HIGH",
        warning_limit: Optional[float] = None,
        maximum_limit: Optional[float] = None,
    ) -> float:
        """
        Convert future drift into a normalized 0-1 risk score.

        HIGH:
            Increasing value = increasing risk.

        LOW:
            Decreasing value = increasing risk.

        BOTH:
            Absolute movement away from current value
            contributes to risk.
        """

        try:
            current = float(
                current_value
            )

            predicted = float(
                predicted_value
            )

        except (
            TypeError,
            ValueError,
        ):
            return 1.0

        if not np.isfinite(
            current
        ) or not np.isfinite(
            predicted
        ):
            return 1.0

        direction = str(
            direction
        ).upper()

        if direction == "HIGH":

            change = predicted - current

        elif direction == "LOW":

            change = current - predicted

        elif direction == "BOTH":

            change = abs(
                predicted - current
            )

        else:

            change = abs(
                predicted - current
            )

        if change <= 0:
            return 0.0

        if (
            warning_limit is not None
            and maximum_limit is not None
        ):

            warning_distance = abs(
                float(warning_limit)
                - current
            )

            maximum_distance = abs(
                float(maximum_limit)
                - current
            )

            if maximum_distance > 0:

                score = (
                    change
                    / maximum_distance
                )

            elif warning_distance > 0:

                score = (
                    change
                    / warning_distance
                )

            else:

                score = 1.0

        else:

            denominator = max(
                abs(current),
                1e-9,
            )

            score = (
                change
                / denominator
            )

        return RiskFusion._clip_score(
            score
        )

    # ============================================================
    # SINGLE-ROW FUSION
    # ============================================================

    def fuse(
        self,
        anomaly_risk: Any = 0.0,
        lot_anomaly_risk: Any = 0.0,
        drift_risk: Any = 0.0,
        current_spec_status: str = "PASS",
        predicted_spec_status: str = "PASS",
    ) -> Dict[str, Any]:
        """
        Fuse all available risk signals for one observation.
        """

        components = {
            "anomaly": self._clip_score(
                anomaly_risk
            ),
            "lot_anomaly": self._clip_score(
                lot_anomaly_risk
            ),
            "drift": self._clip_score(
                drift_risk
            ),
            "current_spec": self._status_score(
                current_spec_status
            ),
            "predicted_spec": self._status_score(
                predicted_spec_status
            ),
        }

        weighted = {
            key: (
                components[key]
                * self.weights[key]
            )
            for key in components
        }

        risk_score = self._clip_score(
            sum(weighted.values())
        )

        risk_level = (
            self._risk_level(
                risk_score
            )
        )

        decision = (
            self._decision(
                risk_score,
                current_spec_status,
                predicted_spec_status,
            )
        )

        contributors = (
            self._contributors(
                weighted
            )
        )

        explanation = (
            self._build_explanation(
                components,
                risk_level,
                decision,
                contributors,
            )
        )

        return {
            "fusion_version": (
                self.FUSION_VERSION
            ),
            "risk_score": round(
                risk_score,
                6,
            ),
            "risk_percentage": round(
                risk_score * 100.0,
                2,
            ),
            "risk_level": risk_level,
            "decision": decision,
            "components": components,
            "weighted_components": weighted,
            "contributors": contributors,
            "current_spec_status": (
                str(
                    current_spec_status
                ).upper()
            ),
            "predicted_spec_status": (
                str(
                    predicted_spec_status
                ).upper()
            ),
            "explanation": explanation,
        }

    # ============================================================
    # RISK LEVEL
    # ============================================================

    @staticmethod
    def _risk_level(
        score: float,
    ) -> str:

        if score >= 0.75:
            return "CRITICAL"

        if score >= 0.50:
            return "HIGH"

        if score >= 0.25:
            return "MEDIUM"

        return "LOW"

    # ============================================================
    # DECISION
    # ============================================================

    @staticmethod
    def _decision(
        score: float,
        current_status: str,
        predicted_status: str,
    ) -> str:

        current = str(
            current_status
        ).upper()

        predicted = str(
            predicted_status
        ).upper()

        if (
            current == "FAIL"
            or predicted == "FAIL"
        ):
            return "REJECT"

        if (
            current == "INVALID"
            or predicted == "INVALID"
        ):
            return "REVIEW"

        if (
            current == "REVIEW"
            or predicted == "REVIEW"
        ):
            return "REVIEW"

        if score >= 0.75:
            return "REJECT"

        if score >= 0.50:
            return "REVIEW"

        if score >= 0.25:
            return "MONITOR"

        return "PASS"

    # ============================================================
    # CONTRIBUTORS
    # ============================================================

    @staticmethod
    def _contributors(
        weighted_components: Dict[str, float],
    ) -> list[str]:

        ordered = sorted(
            weighted_components.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            name
            for name, value in ordered
            if value > 0
        ]

    @staticmethod
    def _build_explanation(
        components: Dict[str, float],
        risk_level: str,
        decision: str,
        contributors: list[str],
    ) -> str:

        if contributors:

            contributor_text = ", ".join(
                contributors
            )

        else:

            contributor_text = "none"

        return (
            f"Overall risk is {risk_level} "
            f"with decision {decision}. "
            f"Primary contributors: "
            f"{contributor_text}."
        )

    # ============================================================
    # DATAFRAME FUSION
    # ============================================================

    def fuse_dataframe(
        self,
        data: pd.DataFrame,
        anomaly_column: str = "Anomaly_Risk",
        lot_anomaly_column: str = "Lot_Risk",
        drift_column: str = "Drift_Risk",
        current_status_column: str = (
            "Current_Spec_Status"
        ),
        predicted_status_column: str = (
            "Predicted_Spec_Status"
        ),
    ) -> pd.DataFrame:
        """
        Apply risk fusion row-by-row.

        Missing optional risk columns are treated as zero.
        Missing specification columns default to PASS.
        """

        output = data.copy()

        anomaly = (
            output[anomaly_column]
            if anomaly_column in output.columns
            else pd.Series(
                0.0,
                index=output.index,
            )
        )

        lot_anomaly = (
            output[lot_anomaly_column]
            if lot_anomaly_column in output.columns
            else pd.Series(
                0.0,
                index=output.index,
            )
        )

        drift = (
            output[drift_column]
            if drift_column in output.columns
            else pd.Series(
                0.0,
                index=output.index,
            )
        )

        current_status = (
            output[current_status_column]
            if current_status_column in output.columns
            else pd.Series(
                "PASS",
                index=output.index,
            )
        )

        predicted_status = (
            output[
                predicted_status_column
            ]
            if predicted_status_column
            in output.columns
            else pd.Series(
                "PASS",
                index=output.index,
            )
        )

        results = []

        for index in output.index:

            result = self.fuse(
                anomaly_risk=anomaly.loc[
                    index
                ],
                lot_anomaly_risk=lot_anomaly.loc[
                    index
                ],
                drift_risk=drift.loc[
                    index
                ],
                current_spec_status=(
                    current_status.loc[
                        index
                    ]
                ),
                predicted_spec_status=(
                    predicted_status.loc[
                        index
                    ]
                ),
            )

            results.append(
                (
                    index,
                    result,
                )
            )

        output[
            "Overall_Risk_Score"
        ] = 0.0

        output[
            "Overall_Risk_Percentage"
        ] = 0.0

        output[
            "Risk_Level"
        ] = "LOW"

        output[
            "Risk_Decision"
        ] = "PASS"

        output[
            "Risk_Contributors"
        ] = ""

        output[
            "Risk_Explanation"
        ] = ""

        for index, result in results:

            output.loc[
                index,
                "Overall_Risk_Score",
            ] = result[
                "risk_score"
            ]

            output.loc[
                index,
                "Overall_Risk_Percentage",
            ] = result[
                "risk_percentage"
            ]

            output.loc[
                index,
                "Risk_Level",
            ] = result[
                "risk_level"
            ]

            output.loc[
                index,
                "Risk_Decision",
            ] = result[
                "decision"
            ]

            output.loc[
                index,
                "Risk_Contributors",
            ] = ", ".join(
                result[
                    "contributors"
                ]
            )

            output.loc[
                index,
                "Risk_Explanation",
            ] = result[
                "explanation"
            ]

        return output

    # ============================================================
    # SUMMARY
    # ============================================================

    def summarize(
        self,
        data: pd.DataFrame,
    ) -> Dict[str, Any]:

        if "Risk_Level" not in data.columns:
            raise ValueError(
                "Risk_Level column not found."
            )

        if "Risk_Decision" not in data.columns:
            raise ValueError(
                "Risk_Decision column not found."
            )

        levels = (
            data["Risk_Level"]
            .astype(str)
            .str.upper()
        )

        decisions = (
            data["Risk_Decision"]
            .astype(str)
            .str.upper()
        )

        return {
            "total": int(
                len(data)
            ),
            "low": int(
                (levels == "LOW").sum()
            ),
            "medium": int(
                (levels == "MEDIUM").sum()
            ),
            "high": int(
                (levels == "HIGH").sum()
            ),
            "critical": int(
                (levels == "CRITICAL").sum()
            ),
            "pass": int(
                (decisions == "PASS").sum()
            ),
            "monitor": int(
                (decisions == "MONITOR").sum()
            ),
            "review": int(
                (decisions == "REVIEW").sum()
            ),
            "reject": int(
                (decisions == "REJECT").sum()
            ),
        }

    # ============================================================
    # CONFIGURATION INFORMATION
    # ============================================================

    def get_fusion_info(
        self,
    ) -> Dict[str, Any]:

        return {
            "fusion_version": (
                self.FUSION_VERSION
            ),
            "weights": self.weights.copy(),
            "weight_sum": round(
                sum(
                    self.weights.values()
                ),
                6,
            ),
        }