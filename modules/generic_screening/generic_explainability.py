"""
SIH26170 - Generic Screening: Explainability
=============================================
Produces a transparent, quantitative WHY message for every record.

The message is assembled purely from already-computed signals:
    - maximum robust z-score per parameter and which parameter it is
    - lot-relative z-score and group name
    - drift percentage for phase-aware parameters
    - specification limit violations
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


class GenericExplainer:
    """Per-record quantitative explanations."""

    VERSION = "1.0"

    GLOBAL_Z_THRESHOLD = 3.5
    LOT_Z_THRESHOLD = 2.5

    def explain_dataframe(
        self,
        data: pd.DataFrame,
        parameters: List[str],
        phase_map: Dict[str, List[str]],
    ) -> pd.DataFrame:
        """
        Return a copy of data with an 'AI_Explanation' column added.
        """

        output = data.copy()
        explanations = []

        for _, row in output.iterrows():
            explanations.append(
                self._explain_row(
                    row,
                    parameters,
                    phase_map,
                )
            )

        output["AI_Explanation"] = explanations
        return output

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _explain_row(
        self,
        row: pd.Series,
        parameters: List[str],
        phase_map: Dict[str, List[str]],
    ) -> str:
        reasons: List[str] = []

        z_value = _num(row.get("Global_Z_Score")) or 0.0

        if z_value >= self.GLOBAL_Z_THRESHOLD:
            culprit = self._worst_parameter(
                row,
                parameters,
            )
            if culprit:
                reasons.append(
                    f"{culprit} deviates strongly from the dataset "
                    f"(robust z={z_value:.2f})"
                )
            else:
                reasons.append(
                    f"row deviates strongly from the dataset "
                    f"(robust z={z_value:.2f})"
                )
        elif z_value >= 2.0:
            culprit = self._worst_parameter(
                row,
                parameters,
            )
            reasons.append(
                (
                    f"{culprit} shows the highest deviation "
                    f"(robust z={z_value:.2f}, anomaly threshold 3.5)"
                    if culprit
                    else (
                        f"highest deviation observed "
                        f"(robust z={z_value:.2f}, threshold 3.5)"
                    )
                )
            )

        isolation = _num(row.get("Anomaly_Isolation_Risk")) or 0.0
        if isolation >= 0.5:
            reasons.append(
                "multivariate outlier pattern detected "
                f"(Isolation Forest risk={isolation:.2f})"
            )

        lot_z = _num(row.get("Lot_Z_Score"))
        if lot_z is not None and lot_z >= self.LOT_Z_THRESHOLD:
            group = row.get("Lot_ID_Used")
            if group is not None and str(group) not in ("", "N/A"):
                reasons.append(
                    f"lot-relative outlier in group '{group}' "
                    f"(lot z={lot_z:.2f})"
                )
            else:
                reasons.append(
                    f"lot-relative outlier detected (lot z={lot_z:.2f})"
                )

        drift_names = list(phase_map.keys()) or parameters
        for drift_parameter in drift_names:
            drift_column = f"{drift_parameter}_Drift_Percent"
            if drift_column in row.index:
                drift = _num(row.get(drift_column))
                if drift is not None and abs(drift) >= 5.0:
                    reasons.append(
                        f"{drift_parameter} drifts {drift:.1f}% "
                        "between earliest and latest measurement"
                    )

        if "Generic_Spec_Violation" in row.index and row.get(
            "Generic_Spec_Violation"
        ):
            reasons.append(
                "specification limit exceeded on this record"
            )

        decision = str(
            row.get("Risk_Decision", "REVIEW")
        ).strip().upper()
        if decision in {"", "NAN", "NONE"}:
            decision = "REVIEW"

        if reasons:
            return (
                f"{decision} because "
                + "; ".join(dict.fromkeys(reasons))
                + "."
            )

        risk_pct = _num(row.get("Overall_Risk_Percentage"))
        risk_text = (
            f" (risk {risk_pct:.1f}%)"
            if risk_pct is not None
            else ""
        )
        return (
            f"{decision}{risk_text}: no strong statistical deviation, "
            "lot anomaly or spec violation was identified in "
            "the available screenable signals."
        )

    @staticmethod
    def _worst_parameter(
        row: pd.Series,
        parameters: List[str],
    ) -> str:
        best = None
        best_z = 0.0
        for parameter in parameters:
            value = _num(row.get(f"Z_{parameter}"))
            if value is None:
                continue
            if abs(value) > abs(best_z):
                best = parameter
                best_z = float(value)
        return best


def _num(value: Any):
    try:
        converted = pd.to_numeric(value, errors="coerce")
        if pd.isna(converted):
            return None
        return float(converted)
    except Exception:
        return None