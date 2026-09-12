from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    """
    SIH26170 anomaly-detection engine.

    Detection layers:
        1. Isolation Forest
        2. Global statistical Z-score
        3. Lot-relative Z-score

    Final anomaly risk is a weighted combination of the
    three signals.

    This module does not:
        - predict future values
        - apply engineering specifications
        - make PASS/REJECT decisions
    """

    MODEL_VERSION = "anomaly_detector_v3"

    DEFAULT_CONFIG = {
        "enabled": True,
        "model": {
            "algorithm": "ISOLATION_FOREST",
            "n_estimators": 200,
            "contamination": 0.15,
            "random_state": 42,
        },
        "global_statistics": {
            "enabled": True,
            "method": "Z_SCORE",
            "threshold": 3.0,
        },
        "lot_aware": {
            "enabled": True,
            "group_columns": [
                "Lot_ID",
                "Component_Type",
            ],
            "minimum_group_size": 5,
            "z_score_threshold": 3.0,
        },
        "risk_weights": {
            "isolation_forest": 0.50,
            "global_statistical": 0.25,
            "lot_relative": 0.25,
        },
        "risk_thresholds": {
            "low": 0.25,
            "medium": 0.50,
            "high": 0.70,
            "critical": 0.85,
        },
    }

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(self, config: Optional[dict] = None):

        self.config = self._merge_config(
            self.DEFAULT_CONFIG,
            config or {},
        )

        self.enabled = bool(
            self.config.get("enabled", True)
        )

        model_config = self.config.get(
            "model",
            {},
        )

        self.n_estimators = int(
            model_config.get(
                "n_estimators",
                200,
            )
        )

        self.contamination = model_config.get(
            "contamination",
            0.15,
        )

        self.random_state = int(
            model_config.get(
                "random_state",
                42,
            )
        )

        global_config = self.config.get(
            "global_statistics",
            {},
        )

        self.global_enabled = bool(
            global_config.get(
                "enabled",
                True,
            )
        )

        self.global_z_threshold = float(
            global_config.get(
                "threshold",
                3.0,
            )
        )

        lot_config = self.config.get(
            "lot_aware",
            {},
        )

        self.lot_enabled = bool(
            lot_config.get(
                "enabled",
                True,
            )
        )

        self.group_columns = list(
            lot_config.get(
                "group_columns",
                [
                    "Lot_ID",
                    "Component_Type",
                ],
            )
        )

        self.minimum_group_size = int(
            lot_config.get(
                "minimum_group_size",
                5,
            )
        )

        self.lot_z_threshold = float(
            lot_config.get(
                "z_score_threshold",
                3.0,
            )
        )

        risk_weights = self.config.get(
            "risk_weights",
            {},
        )

        self.weight_isolation = float(
            risk_weights.get(
                "isolation_forest",
                0.50,
            )
        )

        self.weight_global = float(
            risk_weights.get(
                "global_statistical",
                0.25,
            )
        )

        self.weight_lot = float(
            risk_weights.get(
                "lot_relative",
                0.25,
            )
        )

        self._normalize_weights()

    # ============================================================
    # MAIN DETECTION
    # ============================================================

    def detect(
        self,
        data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Run complete anomaly analysis.

        Returns the original dataset plus anomaly-related
        columns.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Input data must be a pandas DataFrame."
            )

        if data.empty:
            raise ValueError(
                "Input dataset contains no rows."
            )

        result = data.copy()

        if not self.enabled:
            return self._disabled_result(
                result
            )

        features = self._select_features(
            result,
            feature_columns,
        )

        if not features:
            return self._no_feature_result(
                result
            )

        numeric_data = self._prepare_numeric_features(
            result,
            features,
        )

        isolation_risk = self._isolation_forest_risk(
            numeric_data
        )

        global_risk, global_z = (
            self._global_statistical_risk(
                numeric_data
            )
        )

        lot_risk, lot_z = (
            self._lot_relative_risk(
                result,
                numeric_data,
            )
        )

        final_risk = (
            self.weight_isolation
            * isolation_risk
            + self.weight_global
            * global_risk
            + self.weight_lot
            * lot_risk
        )

        final_risk = np.clip(
            final_risk,
            0.0,
            1.0,
        )

        result["Anomaly_Isolation_Risk"] = (
            np.round(
                isolation_risk,
                6,
            )
        )

        result["Anomaly_Global_Risk"] = (
            np.round(
                global_risk,
                6,
            )
        )

        result["Anomaly_Lot_Risk"] = (
            np.round(
                lot_risk,
                6,
            )
        )

        result["Global_Z_Score"] = (
            np.round(
                global_z,
                6,
            )
        )

        result["Lot_Z_Score"] = (
            np.round(
                lot_z,
                6,
            )
        )

        result["Anomaly_Risk"] = (
            np.round(
                final_risk,
                6,
            )
        )

        result["Anomaly_Level"] = [
            self._risk_level(value)
            for value in final_risk
        ]

        result["Anomaly_Flag"] = [
            value >= 0.50
            for value in final_risk
        ]

        result["Anomaly_Model_Version"] = (
            self.MODEL_VERSION
        )

        result["Anomaly_Explanation"] = (
            self._build_explanations(
                isolation_risk,
                global_risk,
                lot_risk,
                global_z,
                lot_z,
                final_risk,
            )
        )

        return result

    # ============================================================
    # FEATURE SELECTION
    # ============================================================

    def _select_features(
        self,
        data: pd.DataFrame,
        feature_columns: Optional[List[str]],
    ) -> List[str]:

        if feature_columns is not None:

            return [
                column
                for column in feature_columns
                if column in data.columns
            ]

        excluded = set(
            self.group_columns
        )

        features = []

        for column in data.columns:

            if column in excluded:
                continue

            if pd.api.types.is_numeric_dtype(
                data[column]
            ):
                features.append(column)

        return features

    # ============================================================
    # NUMERIC PREPARATION
    # ============================================================

    @staticmethod
    def _prepare_numeric_features(
        data: pd.DataFrame,
        features: List[str],
    ) -> pd.DataFrame:

        numeric = data[features].copy()

        for column in numeric.columns:

            numeric[column] = pd.to_numeric(
                numeric[column],
                errors="coerce",
            )

        numeric = numeric.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        # Median imputation is used only for the ML matrix.
        # Original source columns remain unchanged.
        for column in numeric.columns:

            median = numeric[column].median()

            if pd.notna(median):

                numeric[column] = (
                    numeric[column]
                    .fillna(median)
                )

            else:

                numeric[column] = (
                    numeric[column]
                    .fillna(0.0)
                )

        return numeric

    # ============================================================
    # ISOLATION FOREST
    # ============================================================

    def _isolation_forest_risk(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:

        if features.empty:
            return np.zeros(
                len(features)
            )

        if len(features) < 2:
            return np.zeros(
                len(features)
            )

        try:

            model = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=self.random_state,
            )

            model.fit(features)

            decision = model.decision_function(
                features
            )

            # Lower Isolation Forest decision values
            # represent more anomalous observations.
            risk = 1.0 - self._minmax_normalize(
                decision
            )

            return np.clip(
                risk,
                0.0,
                1.0,
            )

        except Exception:

            return np.zeros(
                len(features)
            )

    # ============================================================
    # GLOBAL STATISTICAL RISK
    # ============================================================

    def _global_statistical_risk(
        self,
        features: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:

        n_rows = len(features)

        if (
            not self.global_enabled
            or n_rows == 0
        ):

            return (
                np.zeros(n_rows),
                np.zeros(n_rows),
            )

        z_scores = np.zeros(
            n_rows,
            dtype=float,
        )

        for column in features.columns:

            values = features[column].to_numpy(
                dtype=float
            )

            mean = np.mean(values)
            std = np.std(values)

            if std <= 1e-12:
                continue

            column_z = np.abs(
                (values - mean) / std
            )

            z_scores = np.maximum(
                z_scores,
                column_z,
            )

        threshold = max(
            self.global_z_threshold,
            1e-12,
        )

        risk = np.clip(
            z_scores / threshold,
            0.0,
            1.0,
        )

        return risk, z_scores

    # ============================================================
    # LOT-RELATIVE RISK
    # ============================================================

    def _lot_relative_risk(
        self,
        original_data: pd.DataFrame,
        features: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:

        n_rows = len(original_data)

        if (
            not self.lot_enabled
            or n_rows == 0
            or not all(
                column in original_data.columns
                for column in self.group_columns
            )
        ):

            return (
                np.zeros(n_rows),
                np.zeros(n_rows),
            )

        combined = original_data[
            self.group_columns
        ].copy()

        for column in features.columns:
            combined[column] = features[
                column
            ].to_numpy()

        lot_risk = np.zeros(
            n_rows,
            dtype=float,
        )

        lot_z = np.zeros(
            n_rows,
            dtype=float,
        )

        grouped = combined.groupby(
            self.group_columns,
            dropna=False,
            sort=False,
        )

        for _, group in grouped:

            if len(group) < self.minimum_group_size:
                continue

            group_indices = group.index

            group_features = group[
                features.columns
            ]

            for column in features.columns:

                values = group_features[
                    column
                ].to_numpy(
                    dtype=float
                )

                mean = np.mean(values)
                std = np.std(values)

                if std <= 1e-12:
                    continue

                row_values = group_features[
                    column
                ].to_numpy(
                    dtype=float
                )

                abs_z = np.abs(
                    (row_values - mean) / std
                )

                for position, index in enumerate(
                    group_indices
                ):

                    lot_z[
                        original_data.index.get_loc(
                            index
                        )
                    ] = max(
                        lot_z[
                            original_data.index.get_loc(
                                index
                            )
                        ],
                        abs_z[position],
                    )

        threshold = max(
            self.lot_z_threshold,
            1e-12,
        )

        lot_risk = np.clip(
            lot_z / threshold,
            0.0,
            1.0,
        )

        return lot_risk, lot_z

    # ============================================================
    # RISK LEVEL
    # ============================================================

    def _risk_level(
        self,
        risk: float,
    ) -> str:

        thresholds = self.config.get(
            "risk_thresholds",
            {},
        )

        low = float(
            thresholds.get(
                "low",
                0.25,
            )
        )

        medium = float(
            thresholds.get(
                "medium",
                0.50,
            )
        )

        high = float(
            thresholds.get(
                "high",
                0.70,
            )
        )

        critical = float(
            thresholds.get(
                "critical",
                0.85,
            )
        )

        if risk >= critical:
            return "CRITICAL"

        if risk >= high:
            return "HIGH"

        if risk >= medium:
            return "MEDIUM"

        if risk >= low:
            return "LOW"

        return "NORMAL"

    # ============================================================
    # EXPLANATION
    # ============================================================

    @staticmethod
    def _build_explanations(
        isolation_risk: np.ndarray,
        global_risk: np.ndarray,
        lot_risk: np.ndarray,
        global_z: np.ndarray,
        lot_z: np.ndarray,
        final_risk: np.ndarray,
    ) -> List[str]:

        explanations = []

        for i in range(
            len(final_risk)
        ):

            reasons = []

            if isolation_risk[i] >= 0.70:
                reasons.append(
                    "strong multivariate anomaly"
                )
            elif isolation_risk[i] >= 0.50:
                reasons.append(
                    "elevated multivariate anomaly"
                )

            if global_z[i] >= 3.0:
                reasons.append(
                    f"global deviation Z={global_z[i]:.2f}"
                )

            if lot_z[i] >= 3.0:
                reasons.append(
                    f"lot deviation Z={lot_z[i]:.2f}"
                )

            if not reasons:
                reasons.append(
                    "no significant anomaly signal"
                )

            explanations.append(
                "; ".join(reasons)
                + f". Combined anomaly risk={final_risk[i]:.2f}."
            )

        return explanations

    # ============================================================
    # DISABLED RESULT
    # ============================================================

    def _disabled_result(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:

        result = data.copy()

        result["Anomaly_Isolation_Risk"] = 0.0
        result["Anomaly_Global_Risk"] = 0.0
        result["Anomaly_Lot_Risk"] = 0.0
        result["Global_Z_Score"] = 0.0
        result["Lot_Z_Score"] = 0.0
        result["Anomaly_Risk"] = 0.0
        result["Anomaly_Level"] = "DISABLED"
        result["Anomaly_Flag"] = False
        result["Anomaly_Model_Version"] = (
            self.MODEL_VERSION
        )
        result["Anomaly_Explanation"] = (
            "Anomaly detection is disabled."
        )

        return result

    # ============================================================
    # NO FEATURE RESULT
    # ============================================================

    def _no_feature_result(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:

        result = data.copy()

        n_rows = len(result)

        result["Anomaly_Isolation_Risk"] = 0.0
        result["Anomaly_Global_Risk"] = 0.0
        result["Anomaly_Lot_Risk"] = 0.0
        result["Global_Z_Score"] = 0.0
        result["Lot_Z_Score"] = 0.0
        result["Anomaly_Risk"] = 0.0
        result["Anomaly_Level"] = [
            "UNAVAILABLE"
        ] * n_rows
        result["Anomaly_Flag"] = False
        result["Anomaly_Model_Version"] = (
            self.MODEL_VERSION
        )
        result["Anomaly_Explanation"] = (
            "No usable numeric features were available "
            "for anomaly analysis."
        )

        return result

    # ============================================================
    # WEIGHT NORMALIZATION
    # ============================================================

    def _normalize_weights(self):

        total = (
            self.weight_isolation
            + self.weight_global
            + self.weight_lot
        )

        if total <= 0:

            self.weight_isolation = 0.50
            self.weight_global = 0.25
            self.weight_lot = 0.25

            return

        self.weight_isolation /= total
        self.weight_global /= total
        self.weight_lot /= total

    # ============================================================
    # CONFIG MERGE
    # ============================================================

    @classmethod
    def _merge_config(
        cls,
        base: dict,
        override: dict,
    ) -> dict:

        result = {}

        for key, value in base.items():

            if isinstance(value, dict):
                result[key] = cls._merge_config(
                    value,
                    override.get(
                        key,
                        {}
                    )
                    if isinstance(
                        override.get(
                            key,
                            {}
                        ),
                        dict,
                    )
                    else {},
                )

            else:
                result[key] = value

        for key, value in override.items():

            if (
                isinstance(value, dict)
                and isinstance(
                    result.get(key),
                    dict,
                )
            ):

                result[key] = cls._merge_config(
                    result[key],
                    value,
                )

            else:

                result[key] = value

        return result

    # ============================================================
    # MODEL INFORMATION
    # ============================================================

    def get_model_info(self) -> Dict:
        """
        Return detector configuration for UI/reporting.
        """

        return {
            "model_version": self.MODEL_VERSION,
            "algorithm": "ISOLATION_FOREST",
            "n_estimators": self.n_estimators,
            "contamination": self.contamination,
            "global_z_threshold": (
                self.global_z_threshold
            ),
            "lot_z_threshold": (
                self.lot_z_threshold
            ),
            "lot_aware": self.lot_enabled,
            "group_columns": list(
                self.group_columns
            ),
            "minimum_group_size": (
                self.minimum_group_size
            ),
            "weights": {
                "isolation_forest": (
                    self.weight_isolation
                ),
                "global_statistical": (
                    self.weight_global
                ),
                "lot_relative": (
                    self.weight_lot
                ),
            },
        }