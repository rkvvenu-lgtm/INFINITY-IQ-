"""
SIH26170 - Generic Screening: Drift Detection
==============================================
Detects temporal drift when the dataset provides time information.

Rules
-----
1. Phase columns (e.g. Iddq_0h_uA, Iddq_24h_uA, Iddq_168h_uA):
   For each logical parameter with at least two phase columns,
   per-row drift % is computed between the FIRST and the LAST phase:

       drift_percent = (last - first) / abs(first) * 100

   Drift_Risk contribution = clip(|drift_percent| / 20.0, 0, 1).

2. Single explicit time column (hours / days / index) with one
   measurement column: no per-row drift is computed (single snapshot);
   Drift_Risk = 0 and an honest note is emitted.

3. No time information: Drift_Risk = 0, note emitted.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from modules.generic_screening.generic_feature_discovery import (
    FeatureDiscovery,
    _PHASE_COLUMN_RE,
)


class GenericDriftDetector:
    """Feature-agnostic drift detector."""

    VERSION = "1.0"

    DRIFT_RISK_PERCENT = 20.0

    def detect(
        self,
        data: pd.DataFrame,
        discovery: FeatureDiscovery,
    ) -> Dict[str, Any]:
        """
        Return drift signals.

        Returns dict with keys:
            drift_columns : list of '<parameter>_Drift_Percent'
            Drift_Risk    : per-row 0..1 series
            note          : human-readable method explanation
        """

        drift_columns = []
        risk = pd.Series(
            np.zeros(len(data)),
            index=data.index,
            dtype=float,
        )

        phase_map = discovery.phase_map or {}
        phase_params = [
            base
            for base, columns in phase_map.items()
            if len(columns) >= 2
        ]

        if phase_params:
            for base in phase_params:
                columns = sorted(
                    phase_map[base],
                    key=_phase_number,
                )
                first_col = columns[0]
                last_col = columns[-1]

                first = pd.to_numeric(
                    data[first_col],
                    errors="coerce",
                )
                last = pd.to_numeric(
                    data[last_col],
                    errors="coerce",
                )

                denominator = first.abs().replace(0, np.nan)
                drift_percent = (
                    (last - first)
                    .abs()
                    / denominator
                    * 100.0
                )

                drift_column = f"{base}_Drift_Percent"
                drift_columns.append(drift_column)
                data[drift_column] = drift_percent.fillna(0.0).round(4)

                contribution = (
                    drift_percent.abs()
                    / self.DRIFT_RISK_PERCENT
                ).clip(0, 1).fillna(0.0)

                risk = np.maximum(risk, contribution)

            note = (
                f"Phase-aware drift computed for: "
                + ", ".join(phase_params)
                + "."
            )

        elif discovery.time_columns:
            note = (
                "Time columns present, but no repeated phase "
                "measurements were found; per-row drift is not "
                "computable (single snapshot per parameter)."
            )

        else:
            note = (
                "No time or phase information detected; "
                "drift risk set to 0 (honestly reported)."
            )

        return {
            "drift_columns": drift_columns,
            "Drift_Risk": risk.clip(0, 1).round(4),
            "note": note,
        }


def _phase_number(column: str) -> int:
    match = _PHASE_COLUMN_RE.match(column)
    if match:
        try:
            return int(match.group("phase"))
        except ValueError:
            return 0
    return 0