"""
MLP Evaluation Module

This module contains feature selection methods, evaluation utilities, and data validation functions.
Includes threshold optimization and data cleaning utilities.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List

from src.models.time_series.mlp.mlp_predictor import MLPPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MLPEvaluationMixin:
    """
    Mixin class providing evaluation and feature selection functionality for MLPPredictor.
    """

    def select_features(
        self, X: pd.DataFrame, y: pd.Series, n_features_to_select: int = 50
    ) -> List[str]:
        """
        Simple and reliable feature selection using correlation.

        Args:
            X: Feature matrix
            y: Target values
            n_features_to_select: Number of features to select

        Returns:
            List of selected feature names
        """
        logger.info(
            f"🔍 Selecting {n_features_to_select} features from {len(X.columns)} total features..."
        )

        # Clean data
        X_clean, y_clean = self._clean_data_simple(X, y)

        # Calculate correlations with target
        correlations = X_clean.corrwith(y_clean).abs()

        # Select top features
        selected_features = correlations.nlargest(n_features_to_select).index.tolist()

        # Add important columns if they exist and are not already selected
        important_columns = ["close", "date_int", "ticker_id"]
        missing_important_columns = []

        # First, collect all missing important columns
        for column in important_columns:
            if column in X.columns and column not in selected_features:
                missing_important_columns.append(column)

        # If we have missing important columns, add them all at once
        if missing_important_columns:
            # Calculate how many features we need to remove to make room
            total_needed = len(selected_features) + len(missing_important_columns)
            features_to_remove = max(0, total_needed - n_features_to_select)

            # Remove the lowest correlation features if needed
            if features_to_remove > 0:
                selected_features = selected_features[:-features_to_remove]
                logger.info(
                    f"   Removed {features_to_remove} lowest correlation features to make room for important columns"
                )

            # Add all missing important columns
            selected_features.extend(missing_important_columns)
            logger.info(
                f"   Added {len(missing_important_columns)} important columns: {missing_important_columns}"
            )

        logger.info(f"✅ Selected {len(selected_features)} features using correlation")
        logger.info(f"   Top 10 features: {selected_features[:10]}")

        return selected_features

    def _clean_data_simple(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Simple data cleaning for feature selection.

        Args:
            X: Feature matrix
            y: Target values

        Returns:
            Tuple of cleaned (X, y)
        """
        # Handle NaN/Inf in features
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

        # Handle NaN/Inf in target
        valid_mask = y.notna() & ~np.isinf(y)
        X_clean = X_clean[valid_mask]
        y_clean = y[valid_mask]

        logger.info(f"   Cleaned data: {len(y_clean)} samples, {len(X_clean.columns)} features")

        return X_clean, y_clean


# Extend MLPPredictor with evaluation mixin
class MLPPredictorWithEvaluation(MLPPredictor, MLPEvaluationMixin):
    """
    MLPPredictor with evaluation and feature selection capabilities.
    """

    pass
