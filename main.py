"""
SIH26170 - Core AI Engine
Version: 3.0

Phase 1 Core Pipeline
---------------------
1. Data Loading
2. Preprocessing
3. Data Validation
4. Current Specification Screening
5. Global + Lot-relative Anomaly Detection
6. Model Training / Evaluation
7. Future 168h Prediction
8. Predicted Specification Screening
9. Drift Risk Calculation
10. Risk Fusion
11. Quantitative Explainability
12. Metadata / Result Summary
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

from utils.config_loader import ConfigLoader
from utils.data_loader import DataLoader
from utils.data_preprocessor import DataPreprocessor

from modules.data_validator import DataValidator
from modules.anomaly_detector import AnomalyDetector
from modules.drift_predictor import DriftPredictor
from modules.specification_engine import SpecificationEngine
from modules.risk_fusion import RiskFusion
from modules.explainability_engine import ExplainabilityEngine

from models.model_manager import ModelManager


class SIH26170Engine:
    """Main orchestration engine for SIH26170 Phase 1."""

    ENGINE_VERSION = "3.0"

    def __init__(
        self,
        project_root: Optional[str] = None
    ):

        # =========================================================
        # PROJECT PATHS
        # =========================================================

        if project_root is None:
            self.project_root = Path(
                __file__
            ).resolve().parent
        else:
            self.project_root = Path(
                project_root
            ).resolve()

        self.config_dir = (
            self.project_root / "config"
        )

        self.models_dir = (
            self.project_root / "models"
        )

        self.data_dir = (
            self.project_root / "data"
        )

        # =========================================================
        # CONFIG LOADER
        # =========================================================

        self.config_loader = ConfigLoader(
            project_root=self.project_root
        )

        self.pipeline_config = (
            self._load_pipeline_config()
        )

        # =========================================================
        # DATA MODULES
        # =========================================================

        self.data_loader = DataLoader()

        self.preprocessor = DataPreprocessor(
            missing_numeric_strategy="median",
            preserve_original=True
        )

        self.validator = DataValidator()

        # =========================================================
        # MODEL MANAGEMENT
        # =========================================================

        self.model_manager = ModelManager(
            registry_path=str(
                self.models_dir /
                "model_registry.json"
            ),
            project_root=str(
                self.project_root
            )
        )

        # =========================================================
        # ANOMALY DETECTION
        # =========================================================

        self.anomaly_detector = AnomalyDetector(
            config=self.pipeline_config
        )

        # =========================================================
        # DRIFT PREDICTION
        # =========================================================

        self.drift_predictor = DriftPredictor(
            model_manager=self.model_manager
        )

        # =========================================================
        # SPECIFICATION ENGINE
        # =========================================================

        self.specification_engine = SpecificationEngine(
            specification_path=str(
                self.config_dir /
                "specifications.json"
            ),
            project_root=str(
                self.project_root
            ),
            domain="electronics"
        )

        # =========================================================
        # RISK FUSION
        # =========================================================

        self.risk_fusion = RiskFusion(
            anomaly_weight=0.30,
            lot_anomaly_weight=0.20,
            drift_weight=0.20,
            current_spec_weight=0.15,
            predicted_spec_weight=0.15
        )

        # =========================================================
        # EXPLAINABILITY
        # =========================================================

        self.explainability_engine = (
            ExplainabilityEngine(
                lot_group_columns=[
                    "Lot_ID",
                    "Component_Type"
                ]
            )
        )

        # =========================================================
        # RUNTIME STATE
        # =========================================================

        self.raw_data = None
        self.prepared_data = None
        self.validated_data = None
        self.results = None

        self.dataset_metadata = {}
        self.validation_report = {}
        self.training_results = {}
        self.prediction_results = {}

    # =============================================================
    # CONFIGURATION
    # =============================================================

    def _load_pipeline_config(
        self
    ) -> Dict[str, Any]:

        path = (
            self.config_dir /
            "pipeline_config.json"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {path}"
            )

        return self.config_loader.load(
            str(path)
        )

    # =============================================================
    # DATA LOADING
    # =============================================================

    def load_data(
        self,
        source: Any
    ) -> pd.DataFrame:
        """Load input dataset."""

        loaded = self.data_loader.load(
            source
        )

        if isinstance(
            loaded,
            pd.DataFrame
        ):

            data = loaded

        elif isinstance(
            loaded,
            dict
        ):

            data = loaded.get(
                "data"
            )

            if data is None:
                data = loaded.get(
                    "dataframe"
                )

            if data is None:
                raise ValueError(
                    "DataLoader did not return a DataFrame."
                )

        else:

            raise TypeError(
                "Unsupported DataLoader result type: "
                f"{type(loaded).__name__}"
            )

        self.raw_data = data.copy()

        # Metadata
        try:

            metadata = (
                self.data_loader.get_metadata()
            )

            if isinstance(
                metadata,
                dict
            ):
                self.dataset_metadata = metadata

        except Exception:

            self.dataset_metadata = {}

        return self.raw_data

    # =============================================================
    # PREPROCESSING
    # =============================================================

    def prepare_data(
        self,
        data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Prepare and clean dataset."""

        if data is None:

            if self.raw_data is None:
                raise ValueError(
                    "No raw data available. "
                    "Run load_data() first."
                )

            data = self.raw_data

        self.prepared_data = (
            self.preprocessor.prepare(
                data.copy()
            )
        )

        return self.prepared_data

    # =============================================================
    # VALIDATION
    # =============================================================

    def validate_data(
        self,
        data: Optional[pd.DataFrame] = None,
        mode: str = "AUTO"
    ) -> Dict[str, Any]:
        """Validate data quality."""

        if data is None:

            if self.prepared_data is None:
                raise ValueError(
                    "No prepared data available."
                )

            data = self.prepared_data

        self.validation_report = (
            self.validator.validate(
                data,
                mode=mode
            )
        )

        self.validated_data = data.copy()

        return self.validation_report

    # =============================================================
    # CURRENT SPECIFICATION SCREENING
    # =============================================================

    def evaluate_current_specifications(
        self,
        data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Evaluate current measured values."""

        if data is None:
            data = self.validated_data

        if data is None:
            raise ValueError(
                "No validated data available."
            )

        output = data.copy()

        parameters = (
            self.model_manager
            .get_enabled_parameters()
        )

        for parameter in parameters:

            current_column = (
                self.model_manager
                .get_current_column(
                    parameter
                )
            )

            if current_column not in output.columns:
                continue

            statuses = []

            for value in output[
                current_column
            ]:

                try:

                    result = (
                        self.specification_engine
                        .evaluate_value(
                            parameter,
                            value
                        )
                    )

                    if isinstance(
                        result,
                        dict
                    ):

                        status = result.get(
                            "status",
                            result.get(
                                "decision",
                                "UNKNOWN"
                            )
                        )

                    else:

                        status = str(
                            result
                        )

                except Exception:

                    status = "UNKNOWN"

                statuses.append(
                    status
                )

            output[
                f"Current_{parameter}_Spec_Status"
            ] = statuses

        return output

    # =============================================================
    # ANOMALY DETECTION
    # =============================================================

    def detect_anomalies(
        self,
        data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Detect global and lot-relative anomalies."""

        if data is None:
            data = self.validated_data

        if data is None:
            raise ValueError(
                "No validated data available."
            )

        output = data.copy()

        try:

            result = (
                self.anomaly_detector
                .detect(
                    output
                )
            )

        except AttributeError:

            result = (
                self.anomaly_detector
                .fit_predict(
                    output
                )
            )

        if isinstance(
            result,
            pd.DataFrame
        ):
            return result

        if isinstance(
            result,
            dict
        ):

            result_data = result.get(
                "data"
            )

            if isinstance(
                result_data,
                pd.DataFrame
            ):
                return result_data

        return output

    # =============================================================
    # MODEL TRAINING
    # =============================================================

    def train_models(
        self,
        data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Train and evaluate drift models."""

        if data is None:
            data = self.validated_data

        if data is None:
            raise ValueError(
                "No validated data available."
            )

        self.training_results = (
            self.drift_predictor.train_all(
                data
            )
        )

        return self.training_results

    # =============================================================
    # FUTURE 168H PREDICTION
    # =============================================================

    def predict_future(
        self,
        data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Predict future 168h values."""

        if data is None:
            data = self.validated_data

        if data is None:
            raise ValueError(
                "No validated data available."
            )

        output = data.copy()

        try:

            predictions = (
                self.drift_predictor
                .predict_all(
                    output
                )
            )

            if isinstance(
                predictions,
                pd.DataFrame
            ):

                output = predictions

            elif isinstance(
                predictions,
                dict
            ):

                prediction_data = (
                    predictions.get(
                        "data"
                    )
                )

                if isinstance(
                    prediction_data,
                    pd.DataFrame
                ):
                    output = prediction_data

        except (
            FileNotFoundError,
            ValueError
        ):

            # Models may not exist during
            # validation-only execution.
            pass

        return output

    # =============================================================
    # PREDICTED SPECIFICATION SCREENING
    # =============================================================

    def evaluate_predicted_specifications(
        self,
        data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Evaluate predicted 168h values."""

        if data is None:
            data = self.validated_data

        if data is None:
            raise ValueError(
                "No validated data available."
            )

        output = data.copy()

        parameters = (
            self.model_manager
            .get_enabled_parameters()
        )

        for parameter in parameters:

            prediction_column = (
                self.model_manager
                .get_prediction_column(
                    parameter
                )
            )

            if prediction_column not in output.columns:
                continue

            statuses = []

            for value in output[
                prediction_column
            ]:

                try:

                    result = (
                        self.specification_engine
                        .evaluate_value(
                            parameter,
                            value
                        )
                    )

                    if isinstance(
                        result,
                        dict
                    ):

                        status = result.get(
                            "status",
                            result.get(
                                "decision",
                                "UNKNOWN"
                            )
                        )

                    else:

                        status = str(
                            result
                        )

                except Exception:

                    status = "UNKNOWN"

                statuses.append(
                    status
                )

            output[
                f"Predicted_{parameter}_Spec_Status"
            ] = statuses

        return output

    # =============================================================
    # DRIFT RISK
    # =============================================================

    def calculate_drift_risk(
        self,
        data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Calculate future drift percentage and risk."""

        if data is None:
            data = self.validated_data

        if data is None:
            raise ValueError(
                "No data available."
            )

        output = data.copy()

        drift_columns = []

        parameters = (
            self.model_manager
            .get_enabled_parameters()
        )

        for parameter in parameters:

            current_column = (
                self.model_manager
                .get_current_column(
                    parameter
                )
            )

            prediction_column = (
                self.model_manager
                .get_prediction_column(
                    parameter
                )
            )

            if (
                current_column
                not in output.columns
                or
                prediction_column
                not in output.columns
            ):
                continue

            current = pd.to_numeric(
                output[current_column],
                errors="coerce"
            )

            predicted = pd.to_numeric(
                output[prediction_column],
                errors="coerce"
            )

            denominator = (
                current.abs()
                .replace(
                    0,
                    pd.NA
                )
            )

            drift_percent = (
                (predicted - current)
                .abs()
                / denominator
                * 100.0
            )

            drift_column = (
                f"{parameter}_Drift_Percent"
            )

            output[drift_column] = (
                drift_percent
                .fillna(0.0)
            )

            drift_columns.append(
                drift_column
            )

        if drift_columns:

            output["Drift_Risk"] = (
                output[
                    drift_columns
                ]
                .max(axis=1)
                .clip(0, 100)
                / 100.0
            )

        else:

            output["Drift_Risk"] = 0.0

        return output

    # =============================================================
    # RISK FUSION
    # =============================================================

    def fuse_risk(
        self,
        data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Fuse all major risk signals."""

        if data is None:
            data = self.validated_data

        if data is None:
            raise ValueError(
                "No data available."
            )

        output = data.copy()

        if "Drift_Risk" not in output.columns:

            output = (
                self.calculate_drift_risk(
                    output
                )
            )

        try:

            fused = (
                self.risk_fusion
                .fuse_dataframe(
                    output
                )
            )

            if isinstance(
                fused,
                pd.DataFrame
            ):
                output = fused

        except AttributeError:

            scores = []

            for _, row in output.iterrows():

                result = (
                    self.risk_fusion
                    .fuse(
                        row.to_dict()
                    )
                )

                if isinstance(
                    result,
                    dict
                ):

                    score = result.get(
                        "risk_score",
                        result.get(
                            "score",
                            0.0
                        )
                    )

                else:

                    score = float(
                        result
                    )

                scores.append(
                    score
                )

            output["Risk_Score"] = scores

            output[
                "Risk_Percentage"
            ] = (
                output["Risk_Score"]
                * 100.0
            )

        return output

    # =============================================================
    # EXPLAINABILITY
    # =============================================================

    def add_explainability(
        self,
        data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Add quantitative WHY explanations."""

        if data is None:
            data = self.validated_data

        if data is None:
            raise ValueError(
                "No data available."
            )

        output = data.copy()

        explanations = []

        parameters = (
            self.model_manager
            .get_enabled_parameters()
        )

        for _, row in output.iterrows():

            row_df = pd.DataFrame(
                [row.to_dict()]
            )

            parameter_explanations = []

            for parameter in parameters:

                try:

                    result = (
                        self.explainability_engine
                        .explain_dataframe(
                            row_df,
                            parameter=parameter
                        )
                    )

                    explanation = None

                    if isinstance(
                        result,
                        pd.DataFrame
                    ):

                        if (
                            "AI_Explanation"
                            in result.columns
                        ):

                            explanation = (
                                result.iloc[0][
                                    "AI_Explanation"
                                ]
                            )

                        elif (
                            "Explanation"
                            in result.columns
                        ):

                            explanation = (
                                result.iloc[0][
                                    "Explanation"
                                ]
                            )

                    elif isinstance(
                        result,
                        dict
                    ):

                        explanation = (
                            result.get(
                                "explanation",
                                result.get(
                                    "primary_reason"
                                )
                            )
                        )

                    if explanation:

                        parameter_explanations.append(
                            f"{parameter}: "
                            f"{explanation}"
                        )

                except Exception:

                    continue

            if parameter_explanations:

                explanations.append(
                    " | ".join(
                        parameter_explanations
                    )
                )

            else:

                explanations.append(
                    "No quantitative explanation available."
                )

        output[
            "AI_Explanation"
        ] = explanations

        return output

    # =============================================================
    # COMPLETE PHASE-1 PIPELINE
    # =============================================================

    def run_core_pipeline(
        self,
        source: Any,
        mode: str = "AUTO",
        train_models: bool = False
    ) -> pd.DataFrame:
        """Execute complete Phase-1 pipeline."""

        # 1. Load
        data = self.load_data(
            source
        )

        # 2. Preprocess
        data = self.prepare_data(
            data
        )

        # 3. Validate
        self.validate_data(
            data,
            mode=mode
        )

        data = (
            self.validated_data.copy()
        )

        # 4. Current specification
        data = (
            self.evaluate_current_specifications(
                data
            )
        )

        # 5. Anomaly detection
        data = (
            self.detect_anomalies(
                data
            )
        )

        # 6. Optional model training
        if train_models:

            self.training_results = (
                self.train_models(
                    data
                )
            )

        # 7. Future prediction
        data = (
            self.predict_future(
                data
            )
        )

        # 8. Predicted specification
        data = (
            self.evaluate_predicted_specifications(
                data
            )
        )

        # 9. Drift risk
        data = (
            self.calculate_drift_risk(
                data
            )
        )

        # 10. Risk fusion
        data = (
            self.fuse_risk(
                data
            )
        )

        # 11. Explainability
        data = (
            self.add_explainability(
                data
            )
        )

        self.results = data

        return self.results

    # =============================================================
    # PUBLIC RUN
    # =============================================================

    def run(
        self,
        source: Any,
        mode: str = "AUTO",
        train_models: bool = False
    ) -> pd.DataFrame:
        """Public pipeline entry point."""

        return self.run_core_pipeline(
            source=source,
            mode=mode,
            train_models=train_models
        )

    # =============================================================
    # METADATA
    # =============================================================

    def _get_specification_version(
        self
    ) -> str:
        """Read the specification version safely."""

        engine = self.specification_engine

        value = getattr(
            engine,
            "specification_version",
            None
        )

        if value:
            return str(value)

        specs = getattr(
            engine,
            "specifications",
            None
        )

        if isinstance(specs, dict):

            value = specs.get(
                "specification_version"
            )

            if value:
                return str(value)

        return "UNKNOWN"

    def build_metadata(
        self
    ) -> Dict[str, Any]:
        """Build audit/report metadata."""

        return {
            "engine_version":
                self.ENGINE_VERSION,

            "pipeline_version":
                self.pipeline_config.get(
                    "project_version",
                    self.ENGINE_VERSION
                ),

            "specification_version":
                self._get_specification_version(),

            "dataset_id":
                self.dataset_metadata.get(
                    "dataset_id",
                    "UNKNOWN"
                ),

            "timestamp_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "enabled_parameters":
                self.model_manager
                .get_enabled_parameters(),

            "model_policy":
                self.model_manager
                .get_model_policy()
        }

    # =============================================================
    # RESULT SUMMARY
    # =============================================================

    def get_result_summary(
        self
    ) -> Dict[str, Any]:
        """Return high-level result summary."""

        if self.results is None:

            return {
                "status": "NO_RESULTS"
            }

        data = self.results

        summary = {
            "status": "COMPLETED",
            "engine_version":
                self.ENGINE_VERSION,
            "rows":
                len(data),
            "columns":
                len(data.columns)
        }

        risk_column = None
        for column in (
            "Overall_Risk_Score",
            "Risk_Score",
            "Overall_Risk_Percentage",
        ):
            if column in data.columns:
                risk_column = column
                break

        if risk_column is not None:

            risk_values = pd.to_numeric(
                data[risk_column],
                errors="coerce"
            )

            summary[
                "average_risk_score"
            ] = float(
                risk_values.mean()
            )

        if "Anomaly_Flag" in data.columns:

            summary[
                "anomaly_count"
            ] = int(
                data["Anomaly_Flag"]
                .fillna(False)
                .astype(bool)
                .sum()
            )

        elif "Anomaly_Risk" in data.columns:

            summary[
                "anomaly_count"
            ] = int(
                (
                    pd.to_numeric(
                        data["Anomaly_Risk"],
                        errors="coerce"
                    ).fillna(0)
                    >= 0.5
                ).sum()
            )

        lot_column = None
        for column in (
            "Anomaly_Lot_Risk",
            "Lot_Risk",
        ):
            if column in data.columns:
                lot_column = column
                break

        if lot_column is not None:

            summary[
                "lot_anomaly_count"
            ] = int(
                (
                    pd.to_numeric(
                        data[lot_column],
                        errors="coerce"
                    ).fillna(0)
                    >= 0.5
                ).sum()
            )

        summary[
            "metadata"
        ] = self.build_metadata()

        return summary


# =================================================================
# DIRECT EXECUTION
# =================================================================

if __name__ == "__main__":

    engine = SIH26170Engine()

    print(
        "SIH26170 Core Engine"
    )
    print(
        "--------------------"
    )
    print(
        "Version:",
        engine.ENGINE_VERSION
    )
    print(
        "Parameters:",
        engine.model_manager
        .get_enabled_parameters()
    )
    print(
        "Status: READY"
    )