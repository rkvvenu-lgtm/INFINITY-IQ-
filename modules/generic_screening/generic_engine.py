"""
SIH26170 - Generic Screening: Orchestration Engine
===================================================
Feature-agnostic end-to-end screening for ANY dataset.

Produces the same output vocabulary used by the rest of the
application (Risk_Level, Risk_Decision, AI_Explanation, ...) so the
results remain interpretable and comparable with the electronics path.

The electronics Phase-1 engine is never modified or invoked here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from modules.generic_screening.generic_feature_discovery import (
    FeatureDiscovery,
    GenericFeatureDiscovery,
)
from modules.generic_screening.generic_anomaly import (
    GenericAnomalyDetector,
)
from modules.generic_screening.generic_drift import (
    GenericDriftDetector,
)
from modules.generic_screening.generic_rules import (
    GenericRiskFusion,
    DECISION_TABLE,
)
from modules.generic_screening.generic_explainability import (
    GenericExplainer,
)


class GenericScreeningEngine:
    """Universal / domain-agnostic screening engine."""

    VERSION = "1.0"

    def __init__(
        self,
        discovery: Optional[GenericFeatureDiscovery] = None,
        anomaly_detector: Optional[GenericAnomalyDetector] = None,
        drift_detector: Optional[GenericDriftDetector] = None,
        risk_fusion: Optional[GenericRiskFusion] = None,
        explainer: Optional[GenericExplainer] = None,
    ) -> None:
        self.discovery = discovery or GenericFeatureDiscovery()
        self.anomaly = anomaly_detector or GenericAnomalyDetector()
        self.drift = drift_detector or GenericDriftDetector()
        self.fusion = risk_fusion or GenericRiskFusion()
        self.explainer = explainer or GenericExplainer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def readiness(
        self,
        data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Generic readiness assessment (no electronics columns required)."""
        return self.discovery.readiness(data)

    def get_engine_info(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "capabilities": [
                "auto_feature_discovery",
                "robust_z_anomaly",
                "isolation_forest",
                "lot_relative_anomaly",
                "phase_drift",
                "adaptive_risk_fusion",
                "decision_table",
                "quantitative_why",
            ],
            "decision_table": DECISION_TABLE,
            "phase": "PHASE_3_GENERIC",
            "phase_1_core_modified": False,
        }

    def screen(
        self,
        data: pd.DataFrame,
        mode: str = "LIVE_FUTURE_SCREENING",
    ) -> Dict[str, Any]:
        """
        Run the full generic screening pipeline.

        Returns a screen dict with the same top-level structure used
        by the electronics path (result, screening_status, decision,
        summary, parameters, domain, ...).
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        readiness = self.discovery.readiness(data)

        if not readiness["ready"]:
            return {
                "engine_version": self.VERSION,
                "screening_status": "NOT_READY",
                "decision": "BLOCKED",
                "mode": mode,
                "domain": "general",
                "domain_display_name": "Universal / Generic",
                "parameters": [],
                "summary": {"readiness": readiness},
                "result": data.copy(),
                "manual_review_required": True,
                "review_reason": (
                    " | ".join(readiness["warnings"])
                    or "Dataset cannot be screened generically."
                ),
                "phase_1_core_modified": False,
            }

        discovery_map = readiness["discovery"]
        parameters = list(readiness["parameters"])
        phase_map = dict(discovery_map.get("phase_map") or {})
        group_column = self._select_group_column(discovery_map)

        # ----------------------------------------------------------
        # Working copy with cleaned numeric parameters.
        # ----------------------------------------------------------
        result = data.copy()

        # Map each logical parameter to an actual numeric column:
        # a direct column when it exists, otherwise the LAST phase column.
        effective_columns: List[str] = []
        column_to_parameter: Dict[str, str] = {}
        for parameter in parameters:
            if parameter in result.columns:
                column = parameter
            elif parameter in phase_map and phase_map[parameter]:
                column = phase_map[parameter][-1]
            else:
                continue
            effective_columns.append(column)
            column_to_parameter[column] = parameter

        for column in effective_columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            ).fillna(result[column].median())
            result[column] = result[column].fillna(0.0)

        # ----------------------------------------------------------
        # 1. Global anomaly detection
        # ----------------------------------------------------------
        global_signals = self.anomaly.detect_global(
            result,
            effective_columns,
        )
        run = pd.concat([result, pd.DataFrame(global_signals)], axis=1)

        # ----------------------------------------------------------
        # 2. Lot / group-relative anomaly detection
        # ----------------------------------------------------------
        lot_signals = self.anomaly.detect_lot(
            run,
            effective_columns,
            group_column,
        )
        run = pd.concat([run, pd.DataFrame(lot_signals)], axis=1)

        # ----------------------------------------------------------
        # 3. Drift detection
        # ----------------------------------------------------------
        drift_signals = self.drift.detect(
            run,
            FeatureDiscovery(**discovery_map),
        )
        run["Drift_Risk"] = drift_signals["Drift_Risk"]
        drift_available = bool(drift_signals["drift_columns"])

        # ----------------------------------------------------------
        # 4. Specification limit check (only when limits present)
        # ----------------------------------------------------------
        spec_available, violations = self._check_limits(
            run,
            parameters,
            discovery_map,
            phase_map,
        )
        if spec_available:
            run["Generic_Spec_Violation"] = violations

        # ----------------------------------------------------------
        # 5. Risk fusion + decision table
        # ----------------------------------------------------------
        run = self.fusion.fuse_dataframe(
            run,
            drift_available=drift_available,
            spec_available=spec_available,
        )

        # ----------------------------------------------------------
        # 6. Quantitative WHY (uses effective columns for naming)
        # ----------------------------------------------------------
        run = self.explainer.explain_dataframe(
            run,
            effective_columns,
            phase_map,
        )

        summary = self._summarize(run, discovery_map, drift_signals, spec_available)

        return {
            "engine_version": self.VERSION,
            "screening_status": "SCREENED",
            "decision": summary["primary_decision"],
            "mode": mode,
            "domain": "general",
            "domain_display_name": "Universal / Generic",
            "domain_detection": {
                "domain": "general",
                "display_name": "Universal / Generic",
                "requires_manual_review": False,
            },
            "parameters": parameters,
            "effective_columns": effective_columns,
            "column_to_parameter": column_to_parameter,
            "phase_map": phase_map,
            "group_column": group_column,
            "feature_discovery": discovery_map,
            "drift_note": drift_signals["note"],
            "summary": summary,
            "result": run,
            "manual_review_required": False,
            "phase_1_core_modified": False,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _select_group_column(
        discovery_map: Dict[str, Any],
    ) -> Optional[str]:
        groups = discovery_map.get("group_columns") or []
        return groups[0] if groups else None

    @staticmethod
    def _check_limits(
        data: pd.DataFrame,
        parameters: List[str],
        discovery_map: Dict[str, Any],
        phase_map: Dict[str, List[str]],
    ) -> tuple:
        """
        Detect rows violating *_max / *_min limit columns present in
        the dataset. Returns (available, violation_series).
        """

        parameter_names = {str(p).lower(): p for p in parameters}
        limits: Dict[str, Dict[str, Optional[str]]] = {}

        for column in discovery_map.get("limit_columns") or []:
            lower = str(column).lower()

            base = None
            for name in parameters:
                if name.lower() in lower:
                    base = name
                    break

            if base is None:
                continue

            is_max = (
                "_max" in lower
                or lower.startswith("max_")
                or "max_limit" in lower
            )
            is_min = (
                "_min" in lower
                or lower.startswith("min_")
                or "min_limit" in lower
            )

            limits.setdefault(base, {"max": None, "min": None})
            if is_max:
                limits[base]["max"] = column
            if is_min:
                limits[base]["min"] = column

        if not limits:
            return False, pd.Series(
                [False] * len(data),
                index=data.index,
            )

        violated = pd.Series(False, index=data.index)

        for base, mapping in limits.items():
            if base in data.columns:
                param_column = base
            elif base in phase_map and phase_map[base]:
                param_column = phase_map[base][-1]
            else:
                continue

            values = pd.to_numeric(
                data[param_column],
                errors="coerce",
            )

            max_column = mapping.get("max")
            if max_column and max_column in data.columns:
                limit = pd.to_numeric(
                    data[max_column],
                    errors="coerce",
                )
                violated |= (values > limit).fillna(False)

            min_column = mapping.get("min")
            if min_column and min_column in data.columns:
                limit = pd.to_numeric(
                    data[min_column],
                    errors="coerce",
                )
                violated |= (values < limit).fillna(False)

        return True, violated

    def _summarize(
        self,
        run: pd.DataFrame,
        discovery_map: Dict[str, Any],
        drift_signals: Dict[str, Any],
        spec_available: bool,
    ) -> Dict[str, Any]:
        decisions = (
            run["Risk_Decision"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        counts = {
            decision: int((decisions == decision).sum())
            for decision in ("PASS", "MONITOR", "REVIEW", "REJECT")
        }

        risk_level = (
            run["Risk_Level"]
            .astype(str)
            .str.strip()
            .str.upper()
            if "Risk_Level" in run.columns
            else pd.Series(["UNKNOWN"] * len(run), index=run.index)
        )

        level_counts = {
            level: int((risk_level == level).sum())
            for level in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
        }

        anomaly_count = int(
            pd.to_numeric(
                run.get("Anomaly_Flag", 0),
                errors="coerce",
            ).fillna(0).astype(bool).sum()
        )

        primary_decision = "PASS"
        for decision in ("REJECT", "REVIEW", "MONITOR", "PASS"):
            if counts.get(decision, 0) > 0:
                primary_decision = decision
                break

        return {
            "rows": int(len(run)),
            "columns": int(len(run.columns)),
            "decision_counts": counts,
            "risk_level_counts": level_counts,
            "anomaly_count": anomaly_count,
            "primary_decision": primary_decision,
            "parameters": discovery_map.get("parameters") or [],
            "group_columns": discovery_map.get("group_columns") or [],
            "phase_map": discovery_map.get("phase_map") or {},
            "drift_note": drift_signals.get("note"),
            "spec_available": spec_available,
            "average_risk": float(
                pd.to_numeric(
                    run["Overall_Risk_Percentage"],
                    errors="coerce",
                ).mean()
                or 0.0
            ),
        }