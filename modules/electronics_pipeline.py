"""
SIH26170 - Electronics Specialized Pipeline
Phase 2, File 3

Purpose:
    Provides an electronics-domain adapter above the locked Phase 1 core.

Design rule:
    This module must not modify Phase 1 files.

Responsibilities:
    - Identify registered electronics parameters.
    - Map measurement columns.
    - Expose engineering-limit information.
    - Describe parameter direction and units.
    - Prepare an electronics-domain screening specification.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from models.model_manager import ModelManager
from modules.specification_engine import SpecificationEngine


class ElectronicsPipeline:
    """
    Electronics-domain intelligence layer.

    This class is intentionally independent of the Phase 1 orchestrator.
    It prepares domain-specific information that higher-level application
    components can use without changing the Phase 1 AI core.
    """

    VERSION = "3.0"
    DOMAIN = "electronics"

    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        specification_engine: Optional[SpecificationEngine] = None,
    ) -> None:
        self.model_manager = model_manager or ModelManager()
        self.specification_engine = specification_engine or SpecificationEngine(
            domain=self.DOMAIN
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Return pipeline metadata and supported electronics parameters."""
        parameters = self.get_parameters()

        return {
            "pipeline_version": self.VERSION,
            "domain": self.DOMAIN,
            "supported_parameters": parameters,
            "parameter_count": len(parameters),
            "phase": "PHASE_2",
            "phase_1_core_modified": False,
        }

    def get_parameters(self) -> List[str]:
        """Return enabled electronics parameters registered in Phase 1."""
        parameters = self.model_manager.get_enabled_parameters()

        return [
            parameter
            for parameter in parameters
            if self.model_manager.is_parameter_registered(parameter)
        ]

    def get_parameter_profile(self, parameter: str) -> Dict[str, Any]:
        """
        Return the complete electronics profile for one registered parameter.
        """
        self._validate_parameter(parameter)

        early_features = self.model_manager.get_early_features(parameter)
        full_features = self.model_manager.get_full_features(parameter)
        target_column = self.model_manager.get_target_column(parameter)
        prediction_column = self.model_manager.get_prediction_column(parameter)

        return {
            "parameter": parameter,
            "display_name": self.model_manager.get_display_name(parameter),
            "unit": self.model_manager.get_unit(parameter),
            "direction": self.model_manager.get_direction(parameter),
            "early_features": early_features,
            "full_features": full_features,
            "target_column": target_column,
            "prediction_column": prediction_column,
            "engineering_limit": self.get_engineering_limit(parameter),
        }

    def get_engineering_limit(self, parameter: str) -> Dict[str, Any]:
        """Return the configured engineering limit for a parameter."""
        self._validate_parameter(parameter)

        return self.specification_engine.get_limits(parameter)

    def get_parameter_mapping(self, parameter: str) -> Dict[str, Any]:
        """
        Return the time-point mapping for an electronics parameter.

        The mapping is derived from the registered Phase 1 feature definitions,
        so the domain layer does not duplicate hard-coded model configuration.
        """
        self._validate_parameter(parameter)

        early_features = self.model_manager.get_early_features(parameter)
        full_features = self.model_manager.get_full_features(parameter)

        return {
            "parameter": parameter,
            "early": self._classify_measurement_columns(early_features),
            "full": self._classify_measurement_columns(full_features),
            "target": self.model_manager.get_target_column(parameter),
            "prediction": self.model_manager.get_prediction_column(parameter),
        }

    def inspect_dataset(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Inspect whether a dataset contains the expected electronics fields.

        This method does not modify the supplied DataFrame.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        profiles: Dict[str, Any] = {}

        for parameter in self.get_parameters():
            profile = self.get_parameter_profile(parameter)

            early_features = profile["early_features"]
            full_features = profile["full_features"]

            early_available = [
                column for column in early_features if column in data.columns
            ]
            full_available = [
                column for column in full_features if column in data.columns
            ]

            profiles[parameter] = {
                "early_required": early_features,
                "early_available": early_available,
                "early_complete": len(early_available) == len(early_features),
                "full_required": full_features,
                "full_available": full_available,
                "full_complete": len(full_available) == len(full_features),
                "target_available": profile["target_column"] in data.columns,
                "prediction_column": profile["prediction_column"],
                "engineering_limit": profile["engineering_limit"],
            }

        return {
            "domain": self.DOMAIN,
            "pipeline_version": self.VERSION,
            "rows": int(len(data)),
            "columns": int(len(data.columns)),
            "parameters": profiles,
        }

    def prepare_screening_context(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Build a read-only screening context for the application layer.

        The returned object contains domain, parameter, measurement and
        specification information. The original DataFrame is untouched.
        """
        inspection = self.inspect_dataset(data)

        return {
            "pipeline_version": self.VERSION,
            "domain": self.DOMAIN,
            "dataset_rows": inspection["rows"],
            "dataset_columns": inspection["columns"],
            "parameters": {
                parameter: {
                    "profile": self.get_parameter_profile(parameter),
                    "mapping": self.get_parameter_mapping(parameter),
                    "availability": inspection["parameters"][parameter],
                }
                for parameter in self.get_parameters()
            },
        }

    def get_screening_rules(self) -> Dict[str, Any]:
        """
        Return engineering screening rules for all registered parameters.
        """
        rules: Dict[str, Any] = {}

        for parameter in self.get_parameters():
            limits = self.get_engineering_limit(parameter)

            rules[parameter] = {
                "direction": self.model_manager.get_direction(parameter),
                "unit": self.model_manager.get_unit(parameter),
                "warning": limits.get("warning_max"),
                "maximum": limits.get("max"),
                "minimum": limits.get("min"),
                "warning_minimum": limits.get("warning_min"),
                "automatic_rejection_allowed": (
                    not self.model_manager.unknown_direction_requires_review(
                        parameter
                    )
                ),
            }

        return {
            "domain": self.DOMAIN,
            "rules": rules,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_parameter(self, parameter: str) -> None:
        """Validate that a parameter is registered for this domain."""
        if not isinstance(parameter, str) or not parameter.strip():
            raise ValueError("parameter must be a non-empty string")

        if not self.model_manager.is_parameter_registered(parameter):
            raise ValueError(
                f"Parameter '{parameter}' is not registered in the model registry."
            )

    @staticmethod
    def _classify_measurement_columns(
        columns: List[str],
    ) -> Dict[str, Optional[str]]:
        """
        Classify measurement columns by their time point.

        Expected naming convention:
            *_0h_*
            *_24h_*
            *_96h_*
            *_168h_*
        """
        mapping: Dict[str, Optional[str]] = {
            "0h": None,
            "24h": None,
            "96h": None,
            "168h": None,
        }

        for column in columns:
            text = str(column).lower()

            for time_point in mapping:
                if f"_{time_point}_" in text:
                    mapping[time_point] = column
                    break

        return mapping


if __name__ == "__main__":
    pipeline = ElectronicsPipeline()

    print("ElectronicsPipeline : PASS")
    print("Version:", pipeline.VERSION)
    print("Domain:", pipeline.DOMAIN)
    print("Parameters:", pipeline.get_parameters())

    for parameter in pipeline.get_parameters():
        profile = pipeline.get_parameter_profile(parameter)
        print(
            f"{parameter}: "
            f"direction={profile['direction']}, "
            f"unit={profile['unit']}"
        )