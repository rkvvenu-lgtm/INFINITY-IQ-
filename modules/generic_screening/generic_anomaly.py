"""
SIH26170 - Generic Screening: Anomaly Detection
================================================
Statistical anomaly detection that works on ANY numeric parameter
set (no electronics-specific knowledge required).

Methods (documented, deterministic)
-----------------------------------
1. Robust z-score (median / MAD):
       robust_z = 0.6745 * (value - median) / MAD
   A row is a global anomaly candidate when its maximum robust
   z-score across parameters is >= 3.5 sigma.

2. Isolation Forest (interaction detector):
       Single-class occupancy on standardised parameters; used to
       complement the robust z-score, never to replace it.

3. Lot / group-relative analysis:
       When a group column exists, within-group z-scores identify
       rows that deviate from their OWN lot (not the global mean).

Output scaling
--------------
Anomaly_Risk (0..1):
    global  -> clip((max_robust_z - 3.0) / 3.0, 0, 1)
    blended -> 0.6 * global + 0.4 * isolation_risk
Anomaly_Lot_Risk (0..1):
    clip((max_lot_z - 2.5) / 2.5, 0, 1)
Anomaly_Flag:
    True when blended risk >= 0.5 OR max robust z >= 3.5.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class GenericAnomalyDetector:
    """Feature-agnostic anomaly detector."""

    VERSION = "1.0"

    GLOBAL_Z_THRESHOLD = 3.5
    LOT_Z_THRESHOLD = 2.5
    FLAG_RISK_THRESHOLD = 0.5
    IF_CONTAMINATION = 0.05
    IF_RANDOM_STATE = 42

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_global(
        self,
        data: pd.DataFrame,
        parameters: List[str],
    ) -> Dict[str, pd.Series]:
        """
        Return per-row global anomaly signals.

        Returns dict with keys:
            Global_Z_Score, Anomaly_Risk, Anomaly_Flag, Anomaly_Model_Version
        """

        numeric = self._to_numeric_matrix(
            data,
            parameters,
        )

        available_columns = [
            parameter
            for parameter in parameters
            if parameter in data.columns
        ] or []

        z_scores, available = self._robust_z_matrix(numeric)

        per_column_z = {}
        for index, column in enumerate(available_columns):
            per_column_z[f"Z_{column}"] = pd.Series(
                z_scores[:, index] if z_scores.size else np.zeros(len(data)),
                index=data.index,
            ).round(4)

        global_z = pd.Series(
            np.max(np.abs(z_scores), axis=1)
            if z_scores.size
            else np.zeros(len(data)),
            index=data.index,
        ).fillna(0.0)

        robust_risk = self._z_to_risk(
            global_z,
            threshold=3.0,
            full=1.0,
        )

        isolation_risk = self._isolation_forest_risk(numeric)

        if isolation_risk is not None:
            anomaly_risk = (
                0.6 * robust_risk
                + 0.4 * isolation_risk
            )
        else:
            anomaly_risk = robust_risk

        anomaly_flag = (
            (anomaly_risk >= self.FLAG_RISK_THRESHOLD)
            | (global_z >= self.GLOBAL_Z_THRESHOLD)
        )

        return {
            "Global_Z_Score": global_z.round(4),
            "Anomaly_Risk": anomaly_risk.clip(0, 1).round(4),
            "Anomaly_Robust_Risk": robust_risk.clip(0, 1).round(4),
            "Anomaly_Isolation_Risk": (
                isolation_risk.clip(0, 1).round(4)
                if isolation_risk is not None
                else pd.Series(
                    np.zeros(len(data)),
                    index=data.index,
                    dtype=float,
                )
            ),
            "Anomaly_Flag": anomaly_flag,
            "Anomaly_Model_Version": pd.Series(
                f"generic-robust-z+if-v{self.VERSION}",
                index=data.index,
            ),
            **per_column_z,
        }

    def detect_lot(
        self,
        data: pd.DataFrame,
        parameters: List[str],
        group_column: Optional[str],
    ) -> Dict[str, pd.Series]:
        """
        Return per-row lot/group-relative anomaly signals.

        Returns dict with keys:
            Anomaly_Lot_Risk, Lot_Z_Score, Lot_ID_Used
        """

        default_lot_risk = pd.Series(
            np.zeros(len(data)),
            index=data.index,
        )

        if not group_column or group_column not in data.columns:
            return {
                "Anomaly_Lot_Risk": default_lot_risk,
                "Lot_Z_Score": pd.Series(
                    np.zeros(len(data)),
                    index=data.index,
                ),
                "Lot_ID_Used": pd.Series(
                    "N/A",
                    index=data.index,
                ),
            }

        lot_labels = data[group_column].astype(str).fillna("UNKNOWN")
        lot_used = lot_labels.copy()

        lot_z = pd.Series(
            np.zeros(len(data)),
            index=data.index,
            dtype=float,
        )

        for lot in lot_labels.unique():
            mask = lot_labels == lot
            if int(mask.sum()) < 3:
                continue
            matrix = self._to_numeric_matrix(data.loc[mask], parameters)
            group_z, _ = self._robust_z_matrix(matrix)
            if group_z.size:
                lot_z.loc[mask] = np.max(np.abs(group_z), axis=1)

        lot_risk = self._z_to_risk(
            lot_z,
            threshold=self.LOT_Z_THRESHOLD - 0.5,
            full=self.LOT_Z_THRESHOLD + 2.5,
        )

        flag_mask = lot_z >= self.LOT_Z_THRESHOLD
        final_risk = pd.Series(np.zeros(len(data)), index=data.index, dtype=float)
        final_risk[flag_mask] = 1.0
        blend = (
            0.5 * lot_risk
            + 0.5 * pd.Series(flag_mask.astype(float), index=data.index)
        )
        final_risk = np.maximum(final_risk, blend)

        return {
            "Anomaly_Lot_Risk": final_risk.clip(0, 1).round(4),
            "Lot_Z_Score": lot_z.round(4),
            "Lot_ID_Used": lot_used,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _to_numeric_matrix(
        self,
        data: pd.DataFrame,
        parameters: List[str],
    ) -> np.ndarray:
        available = [p for p in parameters if p in data.columns]
        if not available:
            return np.zeros((len(data), 1))
        matrix = data[available].apply(
            lambda col: pd.to_numeric(col, errors="coerce")
        )
        # Median fill per column.
        matrix = matrix.fillna(matrix.median())
        matrix = matrix.fillna(0.0)
        return matrix.to_numpy(dtype=float)

    @staticmethod
    def _robust_z_matrix(
        matrix: np.ndarray,
    ) -> tuple:
        if matrix.size == 0:
            return np.zeros_like(matrix), 0
        median = np.nanmedian(matrix, axis=0)
        mad = np.nanmedian(
            np.abs(matrix - median),
            axis=0,
        )
        std = np.nanstd(matrix, axis=0)
        scale = np.where(
            mad > 0,
            mad,
            np.where(std > 0, std, 1.0),
        )
        z = 0.6745 * (matrix - median) / scale
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0), matrix.shape[0]

    @staticmethod
    def _z_to_risk(
        z: pd.Series,
        threshold: float,
        full: float,
    ) -> pd.Series:
        values = z.astype(float)
        span = max(float(full - threshold), 1e-9)
        return ((values - threshold) / span).clip(0.0, 1.0).fillna(0.0)

    def _isolation_forest_risk(
        self,
        matrix: np.ndarray,
    ) -> Optional[pd.Series]:
        if matrix.shape[0] < self.IF_RANDOM_STATE:
            return None
        if matrix.shape[1] < 1:
            return None
        try:
            std = matrix.std(axis=0)
            std[std == 0] = 1.0
            scaled = (matrix - matrix.mean(axis=0)) / std
            model = IsolationForest(
                n_estimators=100,
                contamination=self.IF_CONTAMINATION,
                random_state=self.IF_RANDOM_STATE,
            )
            model.fit(scaled)
            scores = model.score_samples(scaled)
            # Normalise to 0..1 using min-max of the model scores.
            score_min = float(scores.min())
            score_max = float(scores.max())
            if score_max - score_min < 1e-12:
                return None
            risk = (scores - score_min) / (score_max - score_min)
            # Larger risk = more anomalous. score_samples is positive for
            # inliers, so invert.
            risk = 1.0 - risk
            return pd.Series(risk, dtype=float)
        except Exception:
            return None