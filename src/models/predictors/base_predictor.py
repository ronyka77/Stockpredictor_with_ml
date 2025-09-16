"""
Base Model Predictor

This module provides a base class for model predictors (e.g., LightGBM, XGBoost)
to handle common functionality like data loading, validation, and saving predictions.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple
import mlflow
from abc import ABC, abstractmethod

from src.data_utils.target_engineering import convert_percentage_predictions_to_prices
from src.feature_engineering.data_loader import StockDataLoader
from src.data_utils.ml_data_pipeline import prepare_ml_data_for_prediction_with_cleaning
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BasePredictor(ABC):
    """
    Abstract Base Class for model predictors.
    """

    def __init__(self, run_id: str, model_type: str):
        """
        Initialize the predictor.
        Args:
            run_id: MLflow run ID to load the model from.
            model_type: The type of the model (e.g., 'lightgbm', 'xgboost', 'autogluon').
        """
        self.run_id = run_id
        self.model_type = model_type
        self.model = None
        self.prediction_horizon = 10
        self.optimal_threshold = None

    @abstractmethod
    def load_model_from_mlflow(self) -> None:
        """
        Abstract method to load a model from MLflow.
        Must be implemented by subclasses.
        """
        pass

    def _load_metadata_from_mlflow(self):
        """
        Load model metadata from MLflow.
        """
        client = mlflow.tracking.MlflowClient()
        run_info = client.get_run(self.run_id)
        model_metadata = run_info.data.params

        if "prediction_horizon" in model_metadata:
            self.prediction_horizon = int(model_metadata["prediction_horizon"])
        elif "model_prediction_horizon" in model_metadata:
            self.prediction_horizon = int(model_metadata["model_prediction_horizon"])

        run_metrics = run_info.data.metrics
        if "final_optimal_threshold" in run_metrics:
            self.optimal_threshold = float(run_metrics["final_optimal_threshold"])

        logger.info(f"✅ Model metadata loaded for run: {self.run_id}")
        logger.info(f"   Prediction horizon: {self.prediction_horizon}")
        if self.optimal_threshold:
            logger.info(f"   Optimal threshold: {self.optimal_threshold:.3f}")

    def load_recent_data(
        self, days_back: int = 30
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the most recent data for making predictions.
        """
        logger.info(f"📊 Loading recent data (last {days_back} days)...")
        data_result = prepare_ml_data_for_prediction_with_cleaning(
            prediction_horizon=self.prediction_horizon,
            days_back=days_back,
        )
        logger.info(f"Data shape after cleaning: {data_result['X_test'].shape}")

        recent_features = data_result["X_test"].copy()
        recent_targets = data_result["y_test"].copy()

        if "ticker_id" in recent_features.columns:
            unique_tickers = recent_features["ticker_id"].nunique()
            logger.info(f"   Unique tickers: {unique_tickers}")

        if (
            self.model
            and hasattr(self.model, "feature_names")
            and self.model.feature_names
        ):
            model_features = set(self.model.feature_names)
            data_features = set(recent_features.columns)

            missing_in_data = model_features - data_features
            if missing_in_data:
                logger.warning(
                    f"   ⚠️  Missing features ({len(missing_in_data)}): Filling with 0.0"
                )
                for feature in missing_in_data:
                    recent_features[feature] = 0.0

            # Reorder columns to match model's expected feature order
            recent_features = recent_features[self.model.feature_names]

        self._validate_feature_diversity(recent_features)
        self._validate_model_compatibility(recent_features)

        metadata_df = pd.DataFrame({"target_values": recent_targets.values})
        return recent_features, metadata_df

    def _validate_feature_diversity(self, features_df: pd.DataFrame) -> None:
        """
        Validate that features show diversity across tickers.
        """
        logger.info("🔍 Validating feature diversity...")
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [
            col for col in numeric_cols if col not in ["ticker_id", "date_int"]
        ]

        if not numeric_cols:
            logger.warning("   ⚠️  No numeric features to validate.")
            return

        for feature in numeric_cols[:5]:
            stats = features_df[feature].describe()
            logger.info(
                f"   {feature}: unique={stats['count'] > 0}, mean={stats['mean']:.3f}, std={stats['std']:.3f}"
            )

    def _validate_model_compatibility(self, features_df: pd.DataFrame) -> None:
        """
        Validate that features are compatible with the loaded model.
        """
        if (
            not self.model
            or not hasattr(self.model, "feature_names")
            or not self.model.feature_names
        ):
            logger.warning("   ⚠️  No model feature names available for validation.")
            return

        logger.info("🔗 Validating model compatibility...")
        model_features = set(self.model.feature_names)
        data_features = set(features_df.columns)
        missing_in_data = model_features - data_features

        if missing_in_data:
            logger.warning(
                f"   🚨 WARNING: Missing {len(missing_in_data)} features. This may impact prediction quality."
            )

    def make_predictions(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the loaded model.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model_from_mlflow() first.")

        logger.info("🔮 Making predictions...")
        prediction_features = features_df.copy()
        if self.model.feature_names:
            prediction_features = prediction_features[self.model.feature_names]

        predictions = self.model.predict(prediction_features)
        logger.info(f"   Predictions generated: {len(predictions)}")
        return predictions

    def get_confidence_scores(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Get confidence scores for predictions.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model_from_mlflow() first.")

        if self.model.feature_names:
            features_df = features_df[self.model.feature_names]

        return self.model.get_prediction_confidence(features_df)

    def apply_threshold_filter(
        self, predictions: np.ndarray, confidence_scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply threshold filtering to predictions via centralized ThresholdPolicy.
        """
        try:
            from src.models.evaluation.threshold_policy import (
                ThresholdPolicy,
                ThresholdConfig,
            )

            policy = ThresholdPolicy()
            threshold_value = (
                self.optimal_threshold if self.optimal_threshold is not None else 0.5
            )
            cfg = ThresholdConfig(method="ge", value=threshold_value)
            # Note: BasePredictor does not keep X here; mask on confidence only
            result = policy.compute_mask(confidence_scores, None, cfg)
            return result.indices, result.mask
        except Exception:
            # Legacy fallback to preserve behavior on unexpected errors
            if self.optimal_threshold is None:
                return np.arange(len(predictions)), np.ones(
                    len(predictions), dtype=bool
                )
            threshold_mask = confidence_scores >= self.optimal_threshold
            filtered_indices = np.where(threshold_mask)[0]
            return filtered_indices, threshold_mask

    def save_predictions_to_excel(
        self,
        features_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        predictions: np.ndarray,
        model_name=None,
    ) -> str:
        """
        Save predictions to an Excel file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if model_name is not None:
            file_name = f"predictions_{model_name}_{timestamp}.xlsx"
        else:
            file_name = f"predictions_{self.run_id[:8]}_{timestamp}.xlsx"

        output_dir = os.path.join("predictions", self.model_type)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, file_name)

        logger.info(f"💾 Saving predictions to: {output_path}")

        results_df = self._build_results_dataframe_and_profit(
            features_df=features_df,
            metadata_df=metadata_df,
            predictions=predictions,
        )
        if not results_df.empty:
            # Drop threshold-related columns before saving (keep date_int for schema compatibility)
            results_df = results_df.drop(
                ["passes_threshold", "optimal_threshold"], axis=1, errors="ignore"
            )
            results_df.to_excel(output_path, index=False)
            logger.info(f"   Saved {len(results_df)} predictions.")
            return output_path
        else:
            logger.info("No predictions to save")
            return None

    def _merge_ticker_metadata(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get ticker metadata from the database.
        """
        try:
            results_df["date"] = pd.to_datetime(
                results_df["date_int"], unit="D", origin="2020-01-01"
            ).dt.strftime("%Y-%m-%d")
            data_loader = StockDataLoader()
            all_metadata_df = data_loader.get_ticker_metadata()
            if not all_metadata_df.empty:
                ticker_map = dict(zip(all_metadata_df["id"], all_metadata_df["ticker"]))
                name_map = dict(zip(all_metadata_df["id"], all_metadata_df["name"]))
                results_df["ticker"] = (
                    results_df["ticker_id"]
                    .map(ticker_map)
                    .fillna(results_df["ticker_id"])
                )
                results_df["company_name"] = (
                    results_df["ticker_id"].map(name_map).fillna("Unknown")
                )
        except Exception as e:
            logger.warning(f"   Could not fetch ticker metadata: {e}")
            results_df["ticker"] = results_df["ticker_id"]
            results_df["company_name"] = "Unknown"
        return results_df

    def _build_results_dataframe_and_profit(
        self,
        *,
        features_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        predictions: np.ndarray,
    ) -> Tuple[pd.DataFrame, float]:
        """
        Build the results DataFrame and compute the average profit per investment
        without writing to disk.
        Returns a tuple of (results_df, avg_profit_per_investment).
        """
        # 1) Minimal result skeleton from predictions and metadata
        results_df = metadata_df.copy()
        threshold_filtered_count = 0
        try:
            results_df["ticker_id"] = features_df["ticker_id"].values
        except KeyError:
            logger.warning(
                "   ⚠️  No ticker_id column found in features_df. Using date_int instead."
            )
            logger.warning(f"   ⚠️  Features_df columns: {features_df.columns.tolist()}")

        results_df["date_int"] = features_df["date_int"].values
        results_df = self._merge_ticker_metadata(results_df)

        current_prices = features_df["close"].values
        results_df["predicted_return"] = predictions
        results_df["predicted_price"] = convert_percentage_predictions_to_prices(
            predictions, current_prices, apply_bounds=True
        )
        results_df["current_price"] = current_prices
        results_df.rename(columns={"target_values": "actual_return"}, inplace=True)

        # 2) Apply threshold filtering and immediately top-10 per date on the filtered set
        if self.optimal_threshold is not None:
            confidence_scores = self.get_confidence_scores(features_df)
            _, threshold_mask = self.apply_threshold_filter(
                predictions, confidence_scores
            )
            results_df["confidence_score"] = confidence_scores
            results_df["passes_threshold"] = threshold_mask
            results_df["optimal_threshold"] = self.optimal_threshold

            original_count = len(results_df)
            results_df = results_df[results_df["passes_threshold"]].copy()
            threshold_filtered_count = len(results_df)
            logger.info(
                f"   📊 Threshold filtering: {original_count} → {threshold_filtered_count} predictions ({threshold_filtered_count / original_count:.1%} kept)"
            )

        if threshold_filtered_count == 0:
            logger.warning("   ⚠️  WARNING: No predictions passed the threshold!")
            return pd.DataFrame()
        else:
            results_df = (
                results_df[results_df["predicted_return"] > 0]
                .sort_values(["date", "confidence_score"], ascending=[True, False])
                .groupby("date")
                .head(10)
                .reset_index(drop=True)
            )
            results_df["day_of_week"] = pd.to_datetime(results_df["date"]).dt.day_name()
            top_10_count = len(results_df)
            logger.info(f"   📈 Top 10 final count: {top_10_count}")

        # 3) Compute derived evaluation fields AFTER filtering/top-10
        non_nan_mask = results_df["actual_return"].notna()
        if non_nan_mask.any():
            # Map indices from filtered results back to current_prices by positional alignment
            filtered_prices = results_df.loc[non_nan_mask, "current_price"].values
            filtered_returns = results_df.loc[non_nan_mask, "actual_return"]

            results_df.loc[non_nan_mask, "actual_price"] = (
                convert_percentage_predictions_to_prices(
                    filtered_returns, filtered_prices, apply_bounds=False
                )
            )

            ap = results_df.loc[non_nan_mask, "actual_price"]
            cp = results_df.loc[non_nan_mask, "current_price"]

            # profit per $100 = (100 / current_price) * (actual_price - current_price)
            results_df.loc[non_nan_mask, "profit_100_investment"] = (100.0 / cp) * (
                ap - cp
            )
            results_df.loc[non_nan_mask, "price_prediction_error"] = (
                results_df.loc[non_nan_mask, "predicted_price"]
                - results_df.loc[non_nan_mask, "actual_price"]
            )
            results_df.loc[non_nan_mask, "prediction_successful"] = (
                results_df.loc[non_nan_mask, "actual_price"]
                > results_df.loc[non_nan_mask, "predicted_price"]
            ).astype(int)

        # 4) Aggregate profit metric on the final DataFrame
        if "actual_price" in results_df.columns:
            valid_mask = results_df["actual_price"] > 0
        else:
            valid_mask = pd.Series([False] * len(results_df), index=results_df.index)
        valid_profit_df = results_df[valid_mask].copy()
        avg_profit_per_investment = (
            float(valid_profit_df["profit_100_investment"].mean())
            if not valid_profit_df.empty
            else 0
        )
        valid_profit_count = valid_profit_df.shape[0]
        logger.info(
            f"   💰 Average profit per $100 investment: ${avg_profit_per_investment:.2f} (based on {valid_profit_count} valid predictions)"
        )

        # 4a) Weekday-specific profit aggregates (Friday and Monday)
        if not valid_profit_df.empty:
            friday_mask = results_df["day_of_week"] == "Friday"
            # monday_mask = results_df["day_of_week"] == "Monday"
            # monday_df = valid_profit_df[monday_mask]
            # monday_avg_profit = (
            #     float(monday_df["profit_100_investment"].mean())
            #     if not monday_df.empty
            #     else 0
            # )

            friday_df = valid_profit_df[friday_mask]

            friday_avg_profit = (
                float(friday_df["profit_100_investment"].mean())
                if not friday_df.empty
                else 0
            )

            logger.info(
                f"   🗓️ Friday average profit per $100 investment: ${friday_avg_profit:.2f} (based on {len(friday_df)} predictions)"
            )
            # logger.info(
            #     f"   🗓️ Monday average profit per $100 investment: ${monday_avg_profit:.2f} (based on {len(monday_df)} predictions)"
            # )
            if friday_avg_profit > 10:
                return results_df
            else:
                logger.warning("   ⚠️  WARNING: No predictions should be exported!")
                return pd.DataFrame()

        logger.warning("   ⚠️  WARNING: No predictions should be exported!")
        return pd.DataFrame()

    def evaluate_on_recent_data(
        self, days_back: int = 30
    ) -> Tuple[pd.DataFrame, float, pd.DataFrame, pd.DataFrame, np.ndarray]:
        """
        Run the prediction pipeline without writing to disk and return:
        - results_df
        - avg_profit_per_investment
        - features_df
        - metadata_df
        - predictions
        """
        logger.info(
            f"🚀 Evaluating {self.model_type.upper()} predictions (no file output)..."
        )
        if self.model is None:
            self.load_model_from_mlflow()
            self._load_metadata_from_mlflow()
        features_df, metadata_df = self.load_recent_data(days_back)
        predictions = self.make_predictions(features_df)
        results_df, avg_profit = self._build_results_dataframe_and_profit(
            features_df=features_df, metadata_df=metadata_df, predictions=predictions
        )
        return results_df, avg_profit, features_df, metadata_df, predictions

    def run_prediction_pipeline(self, days_back: int = 30) -> str:
        """
        Run the complete prediction pipeline.
        """
        logger.info(f"🚀 Starting {self.model_type.upper()} prediction pipeline...")
        self.load_model_from_mlflow()
        self._load_metadata_from_mlflow()
        features_df, metadata_df = self.load_recent_data(days_back)
        predictions = self.make_predictions(features_df)
        output_file = self.save_predictions_to_excel(
            features_df, metadata_df, predictions
        )
        return output_file
