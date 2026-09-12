from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib


class ModelManager:
    """
    Central model lifecycle manager for SIH26170.

    Responsibilities:
        - Load the parameter/model registry
        - Resolve model paths
        - Save trained models
        - Load trained models
        - Store model metadata
        - Check model availability
        - Expose registry information to other modules

    The manager does not train models and does not make
    engineering decisions.
    """

    MANAGER_VERSION = "3.0"

    def __init__(
        self,
        registry_path: str = "models/model_registry.json",
        project_root: Optional[str] = None,
    ):
        if project_root is None:
            self.project_root = (
                Path(__file__).resolve().parent.parent
            )
        else:
            self.project_root = Path(
                project_root
            ).resolve()

        self.registry_path = self._resolve_path(
            registry_path
        )

        self.registry = self._load_registry()

        self.model_directory = self._resolve_path(
            self.registry.get(
                "model_policy",
                {}
            ).get(
                "model_directory",
                "models",
            )
        )

        self.model_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ============================================================
    # PATH MANAGEMENT
    # ============================================================

    def _resolve_path(
        self,
        path: str | Path,
    ) -> Path:
        """
        Resolve project-relative paths.
        """

        path = Path(path)

        if path.is_absolute():
            return path

        return self.project_root / path

    # ============================================================
    # REGISTRY
    # ============================================================

    def _load_registry(self) -> Dict[str, Any]:
        """
        Load and validate the model registry.
        """

        if not self.registry_path.exists():
            raise FileNotFoundError(
                "Model registry not found: "
                f"{self.registry_path}"
            )

        try:

            with self.registry_path.open(
                "r",
                encoding="utf-8",
            ) as file:

                registry = json.load(file)

        except json.JSONDecodeError as exc:

            raise ValueError(
                "Invalid JSON in model registry: "
                f"{self.registry_path}"
            ) from exc

        if not isinstance(
            registry,
            dict,
        ):
            raise ValueError(
                "Model registry root must be a JSON object."
            )

        if "parameters" not in registry:
            raise ValueError(
                "Model registry does not contain "
                "'parameters'."
            )

        return registry

    def reload_registry(self) -> None:
        """
        Reload the registry from disk.
        """

        self.registry = self._load_registry()

    def get_registry(self) -> Dict[str, Any]:
        """
        Return a copy of the loaded registry.
        """

        return self.registry.copy()

    def get_registry_version(self) -> str:
        """
        Return the registry version.
        """

        return str(
            self.registry.get(
                "registry_version",
                "unknown",
            )
        )

    # ============================================================
    # PARAMETER INFORMATION
    # ============================================================

    def get_parameter(
        self,
        parameter: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Return registry information for a parameter.
        """

        parameters = self.registry.get(
            "parameters",
            {}
        )

        value = parameters.get(
            parameter
        )

        if value is None:
            return None

        return value.copy()

    def is_parameter_registered(
        self,
        parameter: str,
    ) -> bool:
        """
        Check whether a parameter exists in the registry.
        """

        return (
            parameter
            in self.registry.get(
                "parameters",
                {}
            )
        )

    def is_parameter_enabled(
        self,
        parameter: str,
    ) -> bool:
        """
        Check whether a registered parameter is enabled.
        """

        config = self.get_parameter(
            parameter
        )

        if config is None:
            return False

        return bool(
            config.get(
                "enabled",
                False,
            )
        )

    def get_enabled_parameters(self) -> list[str]:
        """
        Return all enabled registered parameters.
        """

        parameters = self.registry.get(
            "parameters",
            {}
        )

        return [
            parameter
            for parameter, config in parameters.items()
            if bool(
                config.get(
                    "enabled",
                    False,
                )
            )
        ]

    # ============================================================
    # FEATURE INFORMATION
    # ============================================================

    def get_columns(
        self,
        parameter: str,
    ) -> Dict[str, Any]:
        """
        Return column definitions for a parameter.
        """

        config = self.get_parameter(
            parameter
        )

        if config is None:
            raise KeyError(
                f"Parameter '{parameter}' "
                "is not registered."
            )

        return config.get(
            "columns",
            {}
        ).copy()

    def get_current_column(
        self,
        parameter: str,
    ) -> Optional[str]:
        return self.get_columns(
            parameter
        ).get(
            "current"
        )

    def get_early_features(
        self,
        parameter: str,
    ) -> list[str]:
        return list(
            self.get_columns(
                parameter
            ).get(
                "early_features",
                []
            )
        )

    def get_full_features(
        self,
        parameter: str,
    ) -> list[str]:
        return list(
            self.get_columns(
                parameter
            ).get(
                "full_features",
                []
            )
        )

    def get_target_column(
        self,
        parameter: str,
    ) -> Optional[str]:
        return self.get_columns(
            parameter
        ).get(
            "target"
        )

    def get_prediction_column(
        self,
        parameter: str,
    ) -> Optional[str]:
        return self.get_columns(
            parameter
        ).get(
            "prediction"
        )

    # ============================================================
    # PARAMETER METADATA
    # ============================================================

    def get_unit(
        self,
        parameter: str,
    ) -> Optional[str]:

        config = self.get_parameter(
            parameter
        )

        if config is None:
            return None

        return config.get(
            "unit"
        )

    def get_direction(
        self,
        parameter: str,
    ) -> str:

        config = self.get_parameter(
            parameter
        )

        if config is None:
            return "UNKNOWN"

        return str(
            config.get(
                "direction",
                "UNKNOWN",
            )
        ).upper()

    def get_display_name(
        self,
        parameter: str,
    ) -> str:

        config = self.get_parameter(
            parameter
        )

        if config is None:
            return parameter

        return str(
            config.get(
                "display_name",
                parameter,
            )
        )

    # ============================================================
    # ENGINEERING LIMITS
    # ============================================================

    def get_engineering_limit(
        self,
        parameter: str,
    ) -> Dict[str, Any]:

        config = self.get_parameter(
            parameter
        )

        if config is None:
            return {}

        return config.get(
            "engineering_limit",
            {}
        ).copy()

    # ============================================================
    # MODEL PATH
    # ============================================================

    def get_model_path(
        self,
        parameter: str,
        feature_set: str,
        model_name: Optional[str] = None,
    ) -> Path:
        """
        Return the canonical path for a trained model.

        Format:
            <parameter>_<feature_set>_<model_name>.joblib

        If model_name is omitted:
            <parameter>_<feature_set>.joblib
        """

        parameter_safe = self._safe_name(
            parameter
        )

        feature_set_safe = self._safe_name(
            feature_set
        )

        if model_name:

            model_safe = self._safe_name(
                model_name
            )

            filename = (
                f"{parameter_safe}_"
                f"{feature_set_safe}_"
                f"{model_safe}.joblib"
            )

        else:

            filename = (
                f"{parameter_safe}_"
                f"{feature_set_safe}.joblib"
            )

        return (
            self.model_directory
            / filename
        )

    @staticmethod
    def _safe_name(
        value: str,
    ) -> str:
        """
        Convert a value into a filesystem-safe identifier.
        """

        text = str(value).strip()

        safe = "".join(
            character
            if (
                character.isalnum()
                or character in {
                    "_",
                    "-",
                    ".",
                }
            )
            else "_"
            for character in text
        )

        return safe.strip(
            "_."
        ) or "model"

    # ============================================================
    # MODEL SAVE
    # ============================================================

    def save_model(
        self,
        model: Any,
        parameter: str,
        feature_set: str,
        metadata: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
    ) -> Path:
        """
        Save a trained model and its metadata.

        A separate JSON metadata file is stored beside the model.
        """

        if model is None:
            raise ValueError(
                "Cannot save an empty model."
            )

        if not self.is_parameter_registered(
            parameter
        ):
            raise ValueError(
                f"Parameter '{parameter}' "
                "is not registered."
            )

        path = self.get_model_path(
            parameter=parameter,
            feature_set=feature_set,
            model_name=model_name,
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            model,
            path,
        )

        model_metadata = {
            "manager_version": (
                self.MANAGER_VERSION
            ),
            "registry_version": (
                self.get_registry_version()
            ),
            "parameter": parameter,
            "feature_set": feature_set,
            "model_name": model_name,
            "saved_at": (
                datetime.now().isoformat()
            ),
            "model_path": str(path),
        }

        if metadata:
            model_metadata.update(
                metadata
            )

        metadata_path = (
            path.with_suffix(
                ".json"
            )
        )

        with metadata_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                model_metadata,
                file,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        return path

    # ============================================================
    # MODEL LOAD
    # ============================================================

    def load_model(
        self,
        parameter: str,
        feature_set: str,
        model_name: Optional[str] = None,
    ) -> Any:
        """
        Load a previously saved model.
        """

        path = self.get_model_path(
            parameter=parameter,
            feature_set=feature_set,
            model_name=model_name,
        )

        if not path.exists():
            raise FileNotFoundError(
                "Model not found: "
                f"{path}"
            )

        try:

            return joblib.load(
                path
            )

        except Exception as exc:

            raise RuntimeError(
                f"Unable to load model '{path}': "
                f"{exc}"
            ) from exc

    # ============================================================
    # MODEL METADATA
    # ============================================================

    def get_model_metadata(
        self,
        parameter: str,
        feature_set: str,
        model_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Load metadata associated with a trained model.
        """

        model_path = self.get_model_path(
            parameter=parameter,
            feature_set=feature_set,
            model_name=model_name,
        )

        metadata_path = (
            model_path.with_suffix(
                ".json"
            )
        )

        if not metadata_path.exists():
            return None

        try:

            with metadata_path.open(
                "r",
                encoding="utf-8",
            ) as file:

                metadata = json.load(
                    file
                )

            if isinstance(
                metadata,
                dict,
            ):
                return metadata

        except (
            json.JSONDecodeError,
            OSError,
        ):
            return None

        return None

    # ============================================================
    # MODEL STATUS
    # ============================================================

    def model_exists(
        self,
        parameter: str,
        feature_set: str,
        model_name: Optional[str] = None,
    ) -> bool:
        """
        Check whether a trained model exists.
        """

        return self.get_model_path(
            parameter=parameter,
            feature_set=feature_set,
            model_name=model_name,
        ).is_file()

    def get_model_status(
        self,
        parameter: str,
    ) -> Dict[str, Any]:
        """
        Return early/full model availability and metadata.
        """

        status = {
            "parameter": parameter,
            "registered": (
                self.is_parameter_registered(
                    parameter
                )
            ),
            "enabled": (
                self.is_parameter_enabled(
                    parameter
                )
            ),
            "early": {},
            "full": {},
        }

        for feature_set in (
            "early",
            "full",
        ):

            path = self.get_model_path(
                parameter,
                feature_set,
            )

            status[feature_set] = {
                "exists": path.is_file(),
                "path": str(path),
                "metadata": (
                    self.get_model_metadata(
                        parameter,
                        feature_set,
                    )
                ),
            }

        return status

    # ============================================================
    # MODEL POLICY
    # ============================================================

    def get_model_policy(self) -> Dict[str, Any]:
        """
        Return model-selection policy.
        """

        return self.registry.get(
            "model_policy",
            {}
        ).copy()

    def get_allowed_models(self) -> list[str]:
        """
        Return models allowed by the registry.
        """

        return list(
            self.get_model_policy().get(
                "allowed_models",
                []
            )
        )

    def get_selection_metric(self) -> str:
        """
        Return model selection metric.
        """

        return str(
            self.get_model_policy().get(
                "selection_metric",
                "MAE",
            )
        )

    def get_minimum_training_samples(self) -> int:
        """
        Return minimum samples required for training.
        """

        validation = self.get_model_policy().get(
            "validation",
            {}
        )

        return int(
            validation.get(
                "minimum_training_samples",
                10,
            )
        )

    # ============================================================
    # DECISION POLICY
    # ============================================================

    def get_decision_policy(self) -> Dict[str, Any]:
        """
        Return engineering decision policy.
        """

        return self.registry.get(
            "decision_policy",
            {}
        ).copy()

    def unknown_direction_requires_review(
        self,
    ) -> bool:
        """
        Check whether UNKNOWN direction requires engineering
        review.
        """

        return bool(
            self.get_decision_policy().get(
                "unknown_direction_requires_review",
                True,
            )
        )

    # ============================================================
    # REGISTRY SUMMARY
    # ============================================================

    def get_registry_status(self) -> Dict[str, Any]:
        """
        Return a concise registry summary for dashboards
        and reports.
        """

        parameters = (
            self.registry.get(
                "parameters",
                {}
            )
        )

        return {
            "manager_version": (
                self.MANAGER_VERSION
            ),
            "registry_version": (
                self.get_registry_version()
            ),
            "parameter_count": len(
                parameters
            ),
            "enabled_parameters": (
                self.get_enabled_parameters()
            ),
            "model_directory": str(
                self.model_directory
            ),
            "allowed_models": (
                self.get_allowed_models()
            ),
            "selection_metric": (
                self.get_selection_metric()
            ),
        }

    # ============================================================
    # MODEL INVENTORY
    # ============================================================

    def inventory(self) -> list[Dict[str, Any]]:
        """
        Return the current model inventory.

        Useful for the future AI Intelligence dashboard
        and reporting layer.
        """

        inventory = []

        for parameter in self.get_enabled_parameters():

            for feature_set in (
                "early",
                "full",
            ):

                path = self.get_model_path(
                    parameter,
                    feature_set,
                )

                inventory.append({
                    "parameter": parameter,
                    "feature_set": feature_set,
                    "exists": path.is_file(),
                    "model_path": str(path),
                    "metadata": (
                        self.get_model_metadata(
                            parameter,
                            feature_set,
                        )
                    ),
                })

        return inventory

    # ============================================================
    # SUMMARY
    # ============================================================

    def summary(self) -> Dict[str, Any]:
        """
        Return complete manager information.
        """

        return {
            "manager_version": (
                self.MANAGER_VERSION
            ),
            "project_root": str(
                self.project_root
            ),
            "registry_path": str(
                self.registry_path
            ),
            "registry_version": (
                self.get_registry_version()
            ),
            "registry_status": (
                self.get_registry_status()
            ),
            "model_inventory": (
                self.inventory()
            ),
        }