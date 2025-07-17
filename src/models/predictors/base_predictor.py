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


class BasePredictor(ABC):
    """
    Abstract Base Class for model predictors.
    """

    def __init__(self, run_id: str, model_type: str):
        """
        Initialize the predictor.

        Args:
            run_id: MLflow run ID to load the model from.
            model_type: The type of the model (e.g., 'lightgbm', 'xgboost').
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

        if 'prediction_horizon' in model_metadata:
            self.prediction_horizon = int(model_metadata['prediction_horizon'])
        elif 'model_prediction_horizon' in model_metadata:
            self.prediction_horizon = int(model_metadata['model_prediction_horizon'])

        run_metrics = run_info.data.metrics
        if 'final_optimal_threshold' in run_metrics:
            self.optimal_threshold = float(run_metrics['final_optimal_threshold'])

        print(f"✅ Model metadata loaded for run: {self.run_id}")
        print(f"   Prediction horizon: {self.prediction_horizon}")
        if self.optimal_threshold:
            print(f"   Optimal threshold: {self.optimal_threshold:.3f}")

    def load_recent_data(self, days_back: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the most recent data for making predictions.
        """
        print(f"📊 Loading recent data (last {days_back} days)...")
        data_result = prepare_ml_data_for_prediction_with_cleaning(
            prediction_horizon=self.prediction_horizon,
            days_back=days_back
        )
        print(f"   Data shape after cleaning: {data_result['X_test'].shape}")

        recent_features = data_result['X_test'].copy()
        recent_targets = data_result['y_test'].copy()

        if 'ticker_id' in recent_features.columns:
            unique_tickers = recent_features['ticker_id'].nunique()
            print(f"   Unique tickers: {unique_tickers}")

        if self.model and hasattr(self.model, 'feature_names') and self.model.feature_names:
            model_features = set(self.model.feature_names)
            data_features = set(recent_features.columns)

            missing_in_data = model_features - data_features
            if missing_in_data:
                print(f"   ⚠️  Missing features ({len(missing_in_data)}): Filling with 0.0")
                for feature in missing_in_data:
                    recent_features[feature] = 0.0

            # Reorder columns to match model's expected feature order
            recent_features = recent_features[self.model.feature_names]

        self._validate_feature_diversity(recent_features)
        self._validate_model_compatibility(recent_features)

        metadata_df = pd.DataFrame({'target_values': recent_targets.values})
        return recent_features, metadata_df

    def _validate_feature_diversity(self, features_df: pd.DataFrame) -> None:
        """
        Validate that features show diversity across tickers.
        """
        print("🔍 Validating feature diversity...")
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['ticker_id', 'date_int']]

        if not numeric_cols:
            print("   ⚠️  No numeric features to validate.")
            return

        for feature in numeric_cols[:5]:
            stats = features_df[feature].describe()
            print(f"   {feature}: unique={stats['count']>0}, mean={stats['mean']:.3f}, std={stats['std']:.3f}")

    def _validate_model_compatibility(self, features_df: pd.DataFrame) -> None:
        """
        Validate that features are compatible with the loaded model.
        """
        if not self.model or not hasattr(self.model, 'feature_names') or not self.model.feature_names:
            print("   ⚠️  No model feature names available for validation.")
            return

        print("🔗 Validating model compatibility...")
        model_features = set(self.model.feature_names)
        data_features = set(features_df.columns)
        missing_in_data = model_features - data_features

        if missing_in_data:
            print(f"   🚨 WARNING: Missing {len(missing_in_data)} features. This may impact prediction quality.")

    def make_predictions(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the loaded model.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model_from_mlflow() first.")

        print("🔮 Making predictions...")
        prediction_features = features_df.copy()
        if self.model.feature_names:
            prediction_features = prediction_features[self.model.feature_names]

        predictions = self.model.predict(prediction_features)
        print(f"   Predictions generated: {len(predictions)}")
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

    def apply_threshold_filter(self, predictions: np.ndarray, confidence_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply threshold filtering to predictions.
        """
        if self.optimal_threshold is None:
            return np.arange(len(predictions)), np.ones(len(predictions), dtype=bool)
        
        threshold_mask = confidence_scores >= self.optimal_threshold
        filtered_indices = np.where(threshold_mask)[0]
        return filtered_indices, threshold_mask

    def save_predictions_to_excel(self, features_df: pd.DataFrame, metadata_df: pd.DataFrame, predictions: np.ndarray) -> str:
        """
        Save predictions to an Excel file.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f"predictions_{self.run_id[:8]}_{timestamp}.xlsx"
        output_dir = os.path.join("predictions", self.model_type)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, file_name)
        
        print(f"💾 Saving predictions to: {output_path}")

        results_df = metadata_df.copy()
        results_df['ticker_id'] = features_df['ticker_id'].values
        results_df['date_int'] = features_df['date_int'].values
        results_df['date'] = pd.to_datetime(results_df['date_int'], unit='D', origin='2020-01-01').dt.strftime('%Y-%m-%d')

        try:
            data_loader = StockDataLoader()
            all_metadata_df = data_loader.get_ticker_metadata()
            if not all_metadata_df.empty:
                ticker_map = dict(zip(all_metadata_df['id'], all_metadata_df['ticker']))
                name_map = dict(zip(all_metadata_df['id'], all_metadata_df['name']))
                results_df['ticker'] = results_df['ticker_id'].map(ticker_map).fillna(results_df['ticker_id'])
                results_df['company_name'] = results_df['ticker_id'].map(name_map).fillna('Unknown')
            data_loader.close()
        except Exception as e:
            print(f"   Could not fetch ticker metadata: {e}")
            results_df['ticker'] = results_df['ticker_id']
            results_df['company_name'] = 'Unknown'

        current_prices = features_df['close'].values
        results_df['predicted_return'] = predictions
        results_df['predicted_price'] = convert_percentage_predictions_to_prices(predictions, current_prices, apply_bounds=True)
        results_df['current_price'] = current_prices

        results_df.rename(columns={'target_values': 'actual_return'}, inplace=True)
        non_nan_mask = ~np.isnan(results_df['actual_return'])
        
        if non_nan_mask.any():
            valid_returns = results_df.loc[non_nan_mask, 'actual_return']
            valid_prices = current_prices[non_nan_mask]
            results_df.loc[non_nan_mask, 'actual_price'] = convert_percentage_predictions_to_prices(valid_returns, valid_prices, apply_bounds=True)
            results_df.loc[non_nan_mask, 'profit_100_investment'] = 100 * valid_returns
            results_df.loc[non_nan_mask, 'price_prediction_error'] = results_df.loc[non_nan_mask, 'predicted_price'] - results_df.loc[non_nan_mask, 'actual_price']
            results_df.loc[non_nan_mask, 'prediction_successful'] = (results_df.loc[non_nan_mask, 'predicted_price'] > results_df.loc[non_nan_mask, 'actual_price']).astype(int)

        if self.optimal_threshold is not None:
            confidence_scores = self.get_confidence_scores(features_df)
            _, threshold_mask = self.apply_threshold_filter(predictions, confidence_scores)
            results_df['confidence_score'] = confidence_scores
            results_df['passes_threshold'] = threshold_mask
            results_df['optimal_threshold'] = self.optimal_threshold
            
            # 🔥 NEW: Filter to only include rows that pass the threshold
            original_count = len(results_df)
            results_df = results_df[results_df['passes_threshold'] == True].copy()
            filtered_count = len(results_df)
            
            print(f"   📊 Threshold filtering: {original_count} → {filtered_count} predictions ({filtered_count/original_count:.1%} kept)")
            
            if filtered_count == 0:
                print("   ⚠️  WARNING: No predictions passed the threshold! Exporting empty file.")
        else:
            print("   ℹ️  No optimal threshold available - exporting all predictions")

        results_df.to_excel(output_path, index=False)
        print(f"   Saved {len(results_df)} predictions.")
        return output_path

    def run_prediction_pipeline(self, days_back: int = 30) -> str:
        """
        Run the complete prediction pipeline.
        """
        print(f"🚀 Starting {self.model_type.upper()} prediction pipeline...")
        self.load_model_from_mlflow()
        self._load_metadata_from_mlflow()
        features_df, metadata_df = self.load_recent_data(days_back)
        predictions = self.make_predictions(features_df)
        output_file = self.save_predictions_to_excel(features_df, metadata_df, predictions)
        print("✅ Prediction pipeline completed!")
        return output_file 