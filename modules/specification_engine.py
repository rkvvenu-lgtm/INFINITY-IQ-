from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


class SpecificationEngine:
    """
    Engineering specification and limit evaluation engine.

    Responsibilities:
        - Load engineering specifications
        - Support domain-based specification configuration
        - Evaluate current measurements
        - Evaluate predicted 168h measurements
        - Classify PASS / WARNING / FAIL / REVIEW
        - Handle HIGH / LOW / BOTH / UNKNOWN directions
        - Provide engineering-limit explanations
    """

    ENGINE_VERSION = "3.0"

    def __init__(
        self,
        specification_path: str = "config/specifications.json",
        project_root: Optional[str] = None,
        domain: str = "electronics",
    ):
        if project_root is None:
            self.project_root = (
                Path(__file__).resolve().parent.parent
            )
        else:
            self.project_root = Path(
                project_root
            ).resolve()

        self.specification_path = self._resolve_path(
            specification_path
        )

        self.domain = str(
            domain
        ).strip().lower()

        self.specifications = (
            self._load_specifications()
        )

    # ============================================================
    # PATH / CONFIGURATION
    # ============================================================

    def _resolve_path(
        self,
        path: str | Path,
    ) -> Path:

        path = Path(path)

        if path.is_absolute():
            return path

        return self.project_root / path

    def _load_specifications(
        self,
    ) -> Dict[str, Any]:

        if not self.specification_path.exists():
            raise FileNotFoundError(
                "Specification file not found: "
                f"{self.specification_path}"
            )

        try:
            with self.specification_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

        except json.JSONDecodeError as exc:
            raise ValueError(
                "Invalid JSON in specification file."
            ) from exc

        if not isinstance(
            data,
            dict,
        ):
            raise ValueError(
                "Specification root must be a JSON object."
            )

        return data

    def reload(self) -> None:
        """Reload specifications from disk."""
        self.specifications = (
            self._load_specifications()
        )

    # ============================================================
    # DOMAIN
    # ============================================================

    def set_domain(
        self,
        domain: str,
    ) -> None:
        """Change the active engineering domain."""
        self.domain = str(
            domain
        ).strip().lower()

    def get_domain(self) -> str:
        """Return the active engineering domain."""
        return self.domain

    def get_domain_specifications(
        self,
    ) -> Dict[str, Any]:

        domain_config = self.specifications.get(
            self.domain,
            {}
        )

        if not isinstance(
            domain_config,
            dict,
        ):
            return {}

        # Electronics stores parameters directly.
        if self.domain == "electronics":
            return domain_config

        # Other domains currently use a generic
        # 'parameters' container.
        parameters = domain_config.get(
            "parameters",
            {}
        )

        if isinstance(
            parameters,
            dict,
        ):
            return parameters

        return {}

    # ============================================================
    # PARAMETER INFORMATION
    # ============================================================

    def get_parameter_spec(
        self,
        parameter: str,
    ) -> Dict[str, Any]:

        parameters = (
            self.get_domain_specifications()
        )

        spec = parameters.get(
            parameter
        )

        if spec is None:
            raise KeyError(
                f"No specification registered for "
                f"parameter '{parameter}' "
                f"in domain '{self.domain}'."
            )

        if not isinstance(
            spec,
            dict,
        ):
            raise ValueError(
                f"Invalid specification for "
                f"parameter '{parameter}'."
            )

        return spec.copy()

    def is_registered(
        self,
        parameter: str,
    ) -> bool:

        return (
            parameter
            in self.get_domain_specifications()
        )

    def is_enabled(
        self,
        parameter: str,
    ) -> bool:

        if not self.is_registered(
            parameter
        ):
            return False

        spec = self.get_parameter_spec(
            parameter
        )

        return bool(
            spec.get(
                "enabled",
                False,
            )
        )

    def get_enabled_parameters(
        self,
    ) -> list[str]:

        parameters = (
            self.get_domain_specifications()
        )

        return [
            parameter
            for parameter, spec
            in parameters.items()
            if isinstance(spec, dict)
            and bool(
                spec.get(
                    "enabled",
                    False,
                )
            )
        ]

    def get_display_name(
        self,
        parameter: str,
    ) -> str:

        spec = self.get_parameter_spec(
            parameter
        )

        return str(
            spec.get(
                "display_name",
                parameter,
            )
        )

    def get_direction(
        self,
        parameter: str,
    ) -> str:

        spec = self.get_parameter_spec(
            parameter
        )

        return str(
            spec.get(
                "direction",
                "UNKNOWN",
            )
        ).upper()

    def get_unit(
        self,
        parameter: str,
    ) -> str:

        spec = self.get_parameter_spec(
            parameter
        )

        return str(
            spec.get(
                "unit",
                "",
            )
        )

    # ============================================================
    # LIMITS
    # ============================================================

    def get_limits(
        self,
        parameter: str,
    ) -> Dict[str, Optional[float]]:

        spec = self.get_parameter_spec(
            parameter
        )

        limits = spec.get(
            "limits",
            {}
        )

        if not isinstance(
            limits,
            dict,
        ):
            limits = {}

        return {
            "min": self._to_float(
                limits.get("minimum")
            ),
            "warning_min": None,
            "warning_max": self._to_float(
                limits.get("warning")
            ),
            "max": self._to_float(
                limits.get("maximum")
            ),
        }

    @staticmethod
    def _to_float(
        value: Any,
    ) -> Optional[float]:

        if value is None:
            return None

        try:
            return float(value)

        except (
            TypeError,
            ValueError,
        ):
            return None

    # ============================================================
    # SINGLE VALUE EVALUATION
    # ============================================================

    def evaluate_value(
        self,
        parameter: str,
        value: Any,
    ) -> Dict[str, Any]:
        """
        Evaluate one engineering measurement.
        """

        if not self.is_registered(
            parameter
        ):
            return {
                "parameter": parameter,
                "value": value,
                "status": "UNKNOWN",
                "violation": False,
                "warning": False,
                "reason": (
                    "Parameter is not registered "
                    f"for domain '{self.domain}'."
                ),
                "limit_type": None,
                "limit_value": None,
            }

        if not self.is_enabled(
            parameter
        ):
            return {
                "parameter": parameter,
                "value": value,
                "status": "REVIEW",
                "violation": False,
                "warning": False,
                "reason": (
                    "Parameter is disabled in "
                    "the specification."
                ),
                "limit_type": None,
                "limit_value": None,
            }

        numeric_value = self._to_float(
            value
        )

        if numeric_value is None:
            return {
                "parameter": parameter,
                "value": value,
                "status": "INVALID",
                "violation": True,
                "warning": False,
                "reason": (
                    "Value is not numeric."
                ),
                "limit_type": None,
                "limit_value": None,
            }

        limits = self.get_limits(
            parameter
        )

        direction = self.get_direction(
            parameter
        )

        result = {
            "parameter": parameter,
            "value": numeric_value,
            "unit": self.get_unit(
                parameter
            ),
            "direction": direction,
            "status": "PASS",
            "violation": False,
            "warning": False,
            "reason": (
                "Within engineering limits."
            ),
            "limit_type": None,
            "limit_value": None,
        }

        # --------------------------------------------------------
        # HIGH
        # --------------------------------------------------------

        if direction == "HIGH":

            maximum = limits["max"]
            warning = limits["warning_max"]

            if (
                maximum is not None
                and numeric_value > maximum
            ):
                result.update({
                    "status": "FAIL",
                    "violation": True,
                    "reason": (
                        "Value exceeds the maximum "
                        "engineering limit."
                    ),
                    "limit_type": "maximum",
                    "limit_value": maximum,
                })

            elif (
                warning is not None
                and numeric_value > warning
            ):
                result.update({
                    "status": "WARNING",
                    "warning": True,
                    "reason": (
                        "Value exceeds the engineering "
                        "warning threshold."
                    ),
                    "limit_type": "warning",
                    "limit_value": warning,
                })

        # --------------------------------------------------------
        # LOW
        # --------------------------------------------------------

        elif direction == "LOW":

            minimum = limits["min"]
            warning = limits["warning_min"]

            if (
                minimum is not None
                and numeric_value < minimum
            ):
                result.update({
                    "status": "FAIL",
                    "violation": True,
                    "reason": (
                        "Value is below the minimum "
                        "engineering limit."
                    ),
                    "limit_type": "minimum",
                    "limit_value": minimum,
                })

            elif (
                warning is not None
                and numeric_value < warning
            ):
                result.update({
                    "status": "WARNING",
                    "warning": True,
                    "reason": (
                        "Value is below the engineering "
                        "warning threshold."
                    ),
                    "limit_type": "warning",
                    "limit_value": warning,
                })

        # --------------------------------------------------------
        # BOTH
        # --------------------------------------------------------

        elif direction == "BOTH":

            minimum = limits["min"]
            maximum = limits["max"]
            warning_min = limits["warning_min"]
            warning_max = limits["warning_max"]

            if (
                minimum is not None
                and numeric_value < minimum
            ):
                result.update({
                    "status": "FAIL",
                    "violation": True,
                    "reason": (
                        "Value is below the minimum "
                        "engineering limit."
                    ),
                    "limit_type": "minimum",
                    "limit_value": minimum,
                })

            elif (
                maximum is not None
                and numeric_value > maximum
            ):
                result.update({
                    "status": "FAIL",
                    "violation": True,
                    "reason": (
                        "Value exceeds the maximum "
                        "engineering limit."
                    ),
                    "limit_type": "maximum",
                    "limit_value": maximum,
                })

            elif (
                warning_min is not None
                and numeric_value < warning_min
            ):
                result.update({
                    "status": "WARNING",
                    "warning": True,
                    "reason": (
                        "Value is below the warning "
                        "threshold."
                    ),
                    "limit_type": "warning_min",
                    "limit_value": warning_min,
                })

            elif (
                warning_max is not None
                and numeric_value > warning_max
            ):
                result.update({
                    "status": "WARNING",
                    "warning": True,
                    "reason": (
                        "Value exceeds the warning "
                        "threshold."
                    ),
                    "limit_type": "warning_max",
                    "limit_value": warning_max,
                })

        # --------------------------------------------------------
        # UNKNOWN
        # --------------------------------------------------------

        else:

            result.update({
                "status": "REVIEW",
                "reason": (
                    "Engineering direction is UNKNOWN. "
                    "Manual engineering review is required."
                ),
            })

        return result

    # ============================================================
    # DATAFRAME EVALUATION
    # ============================================================

    def evaluate_dataframe(
        self,
        data: pd.DataFrame,
        parameter: str,
        column: str,
        prefix: str = "",
    ) -> pd.DataFrame:

        if column not in data.columns:
            raise ValueError(
                f"Column '{column}' not found "
                "in dataframe."
            )

        results = data[column].apply(
            lambda value: self.evaluate_value(
                parameter,
                value,
            )
        )

        output = data.copy()

        output[
            f"{prefix}{parameter}_Spec_Status"
        ] = results.apply(
            lambda item: item["status"]
        )

        output[
            f"{prefix}{parameter}_Spec_Violation"
        ] = results.apply(
            lambda item: item["violation"]
        )

        output[
            f"{prefix}{parameter}_Spec_Warning"
        ] = results.apply(
            lambda item: item["warning"]
        )

        output[
            f"{prefix}{parameter}_Spec_Reason"
        ] = results.apply(
            lambda item: item["reason"]
        )

        output[
            f"{prefix}{parameter}_Spec_Limit_Type"
        ] = results.apply(
            lambda item: item["limit_type"]
        )

        output[
            f"{prefix}{parameter}_Spec_Limit_Value"
        ] = results.apply(
            lambda item: item["limit_value"]
        )

        return output

    # ============================================================
    # CURRENT / PREDICTED
    # ============================================================

    def evaluate_current(
        self,
        data: pd.DataFrame,
        parameter: str,
        column: str,
    ) -> pd.DataFrame:

        return self.evaluate_dataframe(
            data=data,
            parameter=parameter,
            column=column,
            prefix="Current_",
        )

    def evaluate_predicted(
        self,
        data: pd.DataFrame,
        parameter: str,
        prediction_column: str,
    ) -> pd.DataFrame:

        return self.evaluate_dataframe(
            data=data,
            parameter=parameter,
            column=prediction_column,
            prefix="Predicted_",
        )

    # ============================================================
    # SUMMARY
    # ============================================================

    def summarize(
        self,
        data: pd.DataFrame,
        parameter: str,
        status_column: str,
    ) -> Dict[str, Any]:

        if status_column not in data.columns:
            raise ValueError(
                f"Status column '{status_column}' "
                "not found."
            )

        statuses = (
            data[status_column]
            .astype(str)
            .str.upper()
        )

        return {
            "parameter": parameter,
            "total": int(len(statuses)),
            "pass": int(
                (statuses == "PASS").sum()
            ),
            "warning": int(
                (statuses == "WARNING").sum()
            ),
            "fail": int(
                (statuses == "FAIL").sum()
            ),
            "invalid": int(
                (statuses == "INVALID").sum()
            ),
            "review": int(
                (statuses == "REVIEW").sum()
            ),
        }

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
            "specification_version": (
                self.specifications.get(
                    "specification_version",
                    "unknown",
                )
            ),
            "specification_path": str(
                self.specification_path
            ),
            "domain": self.domain,
            "parameters": (
                self.get_enabled_parameters()
            ),
        }