from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models.model_manager import ModelManager


class DriftPredictor:
    """
    Future drift prediction engine.

    Training:
        Historical 168h measurements are used as targets.

    Live screening:
        Available earlier measurements are used to predict
        the unseen 168h value.

    The class supports:
        - Linear Regression
        - Random Forest
        - Gradient Boosting
        - Early feature set
        - Full feature set
        - MAE
        - RMSE
        - R2
        - Model selection
        - Model persistence through ModelManager
    """

    PREDICTOR_VERSION = "3.0"

    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        random_state: int = 42,
    ):
        self.model_manager = (
            model_manager
            if model_manager is not None
            else ModelManager()
        )

        self.random_state = random_state

    # ============================================================
    # MODEL CREATION
    # ============================================================

    def _create_model(
        self,
        model_name: str,
    ) -> Any:

        name = str(
            model_name
        ).strip().lower()

        if name == "linear regression":
            return LinearRegression()

        if name == "random forest":
            return RandomForestRegressor(
                n_estimators=200,
                random_state=self.random_state,
                n_jobs=-1,
                max_depth=None,
            )

        if name == "gradient boosting":
            return GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=3,
                random_state=self.random_state,
            )

        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    # ============================================================
    # DATA PREPARATION
    # ============================================================

    @staticmethod
    def _numeric_frame(
        data: pd.DataFrame,
        columns: list[str],
    ) -> pd.DataFrame:

        frame = data.copy()

        missing = [
            column
            for column in columns
            if column not in frame.columns
        ]

        if missing:
            raise ValueError(
                "Missing feature columns: "
                + ", ".join(missing)
            )

        for column in columns:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

        return frame

    def _prepare_training_data(
        self,
        data: pd.DataFrame,
        parameter: str,
        feature_set: str,
    ):

        features = self._get_features(
            parameter,
            feature_set,
        )

        target = self.model_manager.get_target_column(
            parameter
        )

        if not target:
            raise ValueError(
                f"No target column registered for {parameter}"
            )

        frame = self._numeric_frame(
            data,
            features + [target],
        )

        frame = frame.dropna(
            subset=features + [target]
        )

        if frame.empty:
            raise ValueError(
                f"No valid training rows for {parameter}"
            )

        X = frame[features]
        y = frame[target]

        return X, y

    def _prepare_prediction_data(
        self,
        data: pd.DataFrame,
        parameter: str,
        feature_set: str,
    ) -> pd.DataFrame:

        features = self._get_features(
            parameter,
            feature_set,
        )

        frame = self._numeric_frame(
            data,
            features,
        )

        return frame[features]

    def _get_features(
        self,
        parameter: str,
        feature_set: str,
    ) -> list[str]:

        normalized = str(
            feature_set
        ).strip().lower()

        if normalized == "early":
            features = (
                self.model_manager
                .get_early_features(parameter)
            )

        elif normalized == "full":
            features = (
                self.model_manager
                .get_full_features(parameter)
            )

        else:
            raise ValueError(
                "feature_set must be "
                "'early' or 'full'."
            )

        if not features:
            raise ValueError(
                f"No {feature_set} features registered "
                f"for {parameter}"
            )

        return features

    # ============================================================
    # METRICS
    # ============================================================

    @staticmethod
    def _calculate_metrics(
        y_true,
        y_pred,
    ) -> Dict[str, float]:

        mae = mean_absolute_error(
            y_true,
            y_pred,
        )

        rmse = float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        )

        if len(
            np.unique(y_true)
        ) > 1:

            r2 = r2_score(
                y_true,
                y_pred,
            )

        else:
            r2 = float("nan")

        return {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
        }

    # ============================================================
    # MODEL EVALUATION
    # ============================================================

    def evaluate_model(
        self,
        data: pd.DataFrame,
        parameter: str,
        feature_set: str,
        model_name: str,
        validation_fraction: float = 0.2,
    ) -> Dict[str, Any]:

        X, y = self._prepare_training_data(
            data,
            parameter,
            feature_set,
        )

        minimum_samples = (
            self.model_manager
            .get_minimum_training_samples()
        )

        if len(X) < minimum_samples:
            raise ValueError(
                f"{parameter}/{feature_set} has "
                f"{len(X)} samples; "
                f"minimum required is "
                f"{minimum_samples}."
            )

        if not (
            0.05
            <= validation_fraction
            <= 0.5
        ):
            raise ValueError(
                "validation_fraction must "
                "be between 0.05 and 0.50."
            )

        split_index = int(
            len(X)
            * (1.0 - validation_fraction)
        )

        split_index = max(
            1,
            min(
                split_index,
                len(X) - 1,
            ),
        )

        X_train = X.iloc[
            :split_index
        ]

        X_valid = X.iloc[
            split_index:
        ]

        y_train = y.iloc[
            :split_index
        ]

        y_valid = y.iloc[
            split_index:
        ]

        model = self._create_model(
            model_name
        )

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_valid
        )

        metrics = self._calculate_metrics(
            y_valid,
            predictions,
        )

        return {
            "parameter": parameter,
            "feature_set": feature_set,
            "model_name": model_name,
            "training_samples": int(
                len(X_train)
            ),
            "validation_samples": int(
                len(X_valid)
            ),
            "metrics": metrics,
            "features": list(
                X.columns
            ),
            "model": model,
        }

    # ============================================================
    # MODEL SELECTION
    # ============================================================

    def select_best_model(
        self,
        data: pd.DataFrame,
        parameter: str,
        feature_set: str,
    ) -> Dict[str, Any]:

        allowed_models = (
            self.model_manager
            .get_allowed_models()
        )

        if not allowed_models:
            raise ValueError(
                "No models are configured."
            )

        results = []

        for model_name in allowed_models:

            result = self.evaluate_model(
                data=data,
                parameter=parameter,
                feature_set=feature_set,
                model_name=model_name,
            )

            results.append(
                result
            )

        selection_metric = (
            self.model_manager
            .get_selection_metric()
        ).upper()

        if selection_metric not in {
            "MAE",
            "RMSE",
        }:
            selection_metric = "MAE"

        best = min(
            results,
            key=lambda item: (
                item["metrics"]
                .get(
                    selection_metric,
                    float("inf"),
                )
            ),
        )

        return {
            "best_model": best,
            "all_models": [
                {
                    "model_name": item[
                        "model_name"
                    ],
                    "metrics": item[
                        "metrics"
                    ],
                }
                for item in results
            ],
            "selection_metric": (
                selection_metric
            ),
        }

    # ============================================================
    # TRAIN AND SAVE
    # ============================================================

    def train_and_save(
        self,
        data: pd.DataFrame,
        parameter: str,
        feature_set: str,
    ) -> Dict[str, Any]:

        selection = (
            self.select_best_model(
                data=data,
                parameter=parameter,
                feature_set=feature_set,
            )
        )

        best = selection[
            "best_model"
        ]

        features = self._get_features(
            parameter,
            feature_set,
        )

        target = (
            self.model_manager
            .get_target_column(parameter)
        )

        X, y = self._prepare_training_data(
            data,
            parameter,
            feature_set,
        )

        final_model = self._create_model(
            best["model_name"]
        )

        final_model.fit(
            X,
            y,
        )

        metadata = {
            "predictor_version": (
                self.PREDICTOR_VERSION
            ),
            "model_type": (
                best["model_name"]
            ),
            "feature_set": feature_set,
            "features": features,
            "target_column": target,
            "training_samples": int(
                len(X)
            ),
            "validation_metrics": (
                best["metrics"]
            ),
            "all_model_metrics": (
                selection["all_models"]
            ),
            "selection_metric": (
                selection["selection_metric"]
            ),
            "target_horizon_hours": 168,
        }

        path = (
            self.model_manager.save_model(
                model=final_model,
                parameter=parameter,
                feature_set=feature_set,
                metadata=metadata,
            )
        )

        try:
            stored_path = str(
                path.relative_to(
                    self.model_manager.project_root
                )
            )
        except ValueError:
            stored_path = str(path)

        return {
            "parameter": parameter,
            "feature_set": feature_set,
            "model_name": best[
                "model_name"
            ],
            "model_path": stored_path,
            "features": features,
            "target": target,
            "metrics": best[
                "metrics"
            ],
            "all_models": selection[
                "all_models"
            ],
            "training_samples": int(
                len(X)
            ),
        }

    # ============================================================
    # TRAIN ALL PARAMETERS
    # ============================================================

    def train_all(
        self,
        data: pd.DataFrame,
        feature_sets: Optional[list[str]] = None,
    ) -> Dict[str, Any]:

        if feature_sets is None:
            feature_sets = [
                "early",
                "full",
            ]

        results = {}

        for parameter in (
            self.model_manager
            .get_enabled_parameters()
        ):

            results[parameter] = {}

            for feature_set in feature_sets:

                results[
                    parameter
                ][feature_set] = (
                    self.train_and_save(
                        data=data,
                        parameter=parameter,
                        feature_set=feature_set,
                    )
                )

        return results

    # ============================================================
    # FUTURE PREDICTION
    # ============================================================

    def predict_future(
        self,
        data: pd.DataFrame,
        parameter: str,
        feature_set: str = "early",
        require_saved_model: bool = True,
    ) -> Dict[str, Any]:

        features = self._get_features(
            parameter,
            feature_set,
        )

        prediction_column = (
            self.model_manager
            .get_prediction_column(
                parameter
            )
        )

        if not prediction_column:
            raise ValueError(
                f"No prediction column registered "
                f"for {parameter}"
            )

        X = self._prepare_prediction_data(
            data,
            parameter,
            feature_set,
        )

        if X.empty:
            raise ValueError(
                f"No valid prediction rows for {parameter}"
            )

        model_exists = (
            self.model_manager
            .model_exists(
                parameter,
                feature_set,
            )
        )

        if require_saved_model:

            if not model_exists:
                raise FileNotFoundError(
                    f"No trained model available for "
                    f"{parameter}/{feature_set}."
                )

            model = (
                self.model_manager
                .load_model(
                    parameter,
                    feature_set,
                )
            )

        else:

            model = None

        if model is None:
            raise ValueError(
                "A trained model is required "
                "for future prediction."
            )

        predictions = model.predict(
            X
        )

        output = data.copy()

        output[
            prediction_column
        ] = np.nan

        valid_index = X.index

        output.loc[
            valid_index,
            prediction_column,
        ] = predictions

        return {
            "data": output,
            "parameter": parameter,
            "feature_set": feature_set,
            "features": features,
            "prediction_column": (
                prediction_column
            ),
            "prediction_horizon_hours": 168,
            "prediction_count": int(
                len(predictions)
            ),
            "model_metadata": (
                self.model_manager
                .get_model_metadata(
                    parameter,
                    feature_set,
                )
            ),
        }

    # ============================================================
    # PREDICT ALL
    # ============================================================

    def predict_all(
        self,
        data: pd.DataFrame,
        feature_set: str = "early",
    ) -> Dict[str, Any]:

        output = data.copy()
        results = {}

        for parameter in (
            self.model_manager
            .get_enabled_parameters()
        ):

            result = self.predict_future(
                data=output,
                parameter=parameter,
                feature_set=feature_set,
            )

            output = result[
                "data"
            ]

            results[parameter] = result

        return {
            "data": output,
            "feature_set": feature_set,
            "parameters": results,
        }

    # ============================================================
    # PREDICTION QUALITY
    # ============================================================

    def get_prediction_quality(
        self,
        parameter: str,
        feature_set: str,
    ) -> Dict[str, Any]:

        metadata = (
            self.model_manager
            .get_model_metadata(
                parameter,
                feature_set,
            )
        )

        if not metadata:
            return {
                "available": False,
                "parameter": parameter,
                "feature_set": feature_set,
            }

        metrics = metadata.get(
            "validation_metrics",
            {}
        )

        return {
            "available": True,
            "parameter": parameter,
            "feature_set": feature_set,
            "model_name": metadata.get(
                "model_type"
            ),
            "MAE": metrics.get(
                "MAE"
            ),
            "RMSE": metrics.get(
                "RMSE"
            ),
            "R2": metrics.get(
                "R2"
            ),
            "training_samples": metadata.get(
                "training_samples"
            ),
            "prediction_horizon_hours": (
                metadata.get(
                    "target_horizon_hours",
                    168,
                )
            ),
        }

    # ============================================================
    # MODEL INFORMATION
    # ============================================================

    def get_model_info(
        self,
        parameter: Optional[str] = None,
    ) -> Dict[str, Any]:

        if parameter:

            return {
                "parameter": parameter,
                "early": (
                    self.model_manager
                    .get_model_status(
                        parameter
                    )["early"]
                ),
                "full": (
                    self.model_manager
                    .get_model_status(
                        parameter
                    )["full"]
                ),
            }

        return {
            parameter: self.get_model_info(
                parameter
            )
            for parameter in (
                self.model_manager
                .get_enabled_parameters()
            )
        }