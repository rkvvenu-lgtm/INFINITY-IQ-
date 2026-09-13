"""
SIH26170 - Generic Screening: Risk Rules
========================================
Adaptive risk fusion and the documented decision table.

Risk fusion (weights auto-renormalise to the signals available)
--------------------------------------------------------------
    base weights:
        anomaly      0.45   (always present)
        lot_relative 0.25   (only when a group column exists)
        drift        0.20   (only when time/phase information exists)
        spec         0.10   (only when matching limit columns exist)

    overall = sum(w_i * risk_i) / sum(w_i)      (present signals only)

Decision table (fixed, documented thresholds)
---------------------------------------------
    Risk %          Risk_Level    Risk_Decision
    < 15            LOW           PASS
    15 .. < 35      MEDIUM        MONITOR
    35 .. < 65      HIGH          REVIEW
    >= 65           CRITICAL      REJECT
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import pandas as pd

# Fixed decision table (single source of truth for the UI + reports).
DECISION_TABLE = [
    {"maximum": 15.0, "level": "LOW", "decision": "PASS"},
    {"maximum": 35.0, "level": "MEDIUM", "decision": "MONITOR"},
    {"maximum": 65.0, "level": "HIGH", "decision": "REVIEW"},
]

DEFAULT_WEIGHTS = {
    "anomaly": 0.45,
    "lot": 0.25,
    "drift": 0.20,
    "spec": 0.10,
}


def decide(
    overall_risk_percentage: Any,
) -> Tuple[str, str]:
    """Map an overall risk percentage to (level, decision)."""

    try:
        value = float(overall_risk_percentage)
    except (TypeError, ValueError):
        value = 0.0

    if value >= 65.0:
        return "CRITICAL", "REJECT"

    for row in DECISION_TABLE:
        if value < row["maximum"]:
            return row["level"], row["decision"]

    return "CRITICAL", "REJECT"


class GenericRiskFusion:
    """Weighted, adaptive risk fusion."""

    VERSION = "1.0"

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fuse_static(
        self,
        anomaly_risk: float,
        lot_risk: float = 0.0,
        drift_risk: float = 0.0,
        spec_violation: bool = False,
        lot_available: bool = False,
        drift_available: bool = False,
        spec_available: bool = False,
    ) -> float:
        """Produce overall risk 0..1 from individual signals."""

        signals = {
            "anomaly": float(anomaly_risk or 0.0),
        }

        if lot_available:
            signals["lot"] = float(lot_risk or 0.0)
        if drift_available:
            signals["drift"] = float(drift_risk or 0.0)
        if spec_available:
            signals["spec"] = 1.0 if spec_violation else 0.0

        total_weight = sum(
            self.weights[key]
            for key in signals
        )

        if total_weight <= 0:
            return 0.0

        overall = (
            sum(
                self.weights[key] * signals[key]
                for key in signals
            )
            / total_weight
        )

        # Documented override: a hard specification violation always
        # warrants at least a REVIEW (risk floor of 0.40).
        if spec_available and spec_violation:
            overall = max(overall, 0.40)

        return min(max(float(overall), 0.0), 1.0)

    def fuse_dataframe(
        self,
        data: pd.DataFrame,
        drift_available: bool = True,
        spec_available: bool = False,
    ) -> pd.DataFrame:
        """
        Add Overall_Risk_Score / Percentage / Level / Decision columns
        to the result dataframe.
        """

        output = data.copy()

        anomaly = pd.to_numeric(
            output.get("Anomaly_Risk", 0),
            errors="coerce",
        ).fillna(0.0).to_numpy()

        lot_available = "Anomaly_Lot_Risk" in output.columns
        lot = (
            pd.to_numeric(
                output["Anomaly_Lot_Risk"],
                errors="coerce",
            ).fillna(0.0).to_numpy()
            if lot_available
            else np_zero(len(output))
        )

        drift = (
            pd.to_numeric(
                output["Drift_Risk"],
                errors="coerce",
            ).fillna(0.0).to_numpy()
            if drift_available
            else np_zero(len(output))
        )

        spec = (
            output["Generic_Spec_Violation"]
            .astype(bool)
            .to_numpy()
            if "Generic_Spec_Violation" in output.columns
            else np_zero_bool(len(output))
        )

        scores = []
        for i in range(len(output)):
            scores.append(
                self.fuse_static(
                    anomaly_risk=anomaly[i],
                    lot_risk=lot[i],
                    drift_risk=drift[i],
                    spec_violation=bool(spec[i]),
                    lot_available=lot_available,
                    drift_available=drift_available,
                    spec_available="Generic_Spec_Violation"
                    in output.columns,
                )
            )

        overall = pd.Series(scores, index=output.index)
        percents = overall * 100.0

        levels = []
        decisions = []
        for value in percents:
            level, decision = decide(value)
            levels.append(level)
            decisions.append(decision)

        output["Overall_Risk_Score"] = overall.round(4)
        output["Overall_Risk_Percentage"] = percents.round(2)
        output["Risk_Level"] = levels
        output["Risk_Decision"] = decisions

        return output


def np_zero(length: int):
    import numpy as np

    return np.zeros(length, dtype=float)


def np_zero_bool(length: int):
    import numpy as np

    return np.zeros(length, dtype=bool)