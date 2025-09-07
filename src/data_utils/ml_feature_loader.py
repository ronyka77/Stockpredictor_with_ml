"""
ML Feature Loader Module

This module provides the MLFeatureLoader class and convenience functions for loading
and preparing features for ML models from consolidated yearly parquet files.
"""

import pandas as pd
from typing import List, Tuple, Optional
from datetime import date

from src.data_collector.indicator_pipeline.feature_storage import FeatureStorage
from src.data_collector.indicator_pipeline.consolidated_storage import (
    ConsolidatedFeatureStorage,
)
from src.feature_engineering.data_loader import StockDataLoader
from src.utils.logger import get_logger

logger = get_logger(__name__, utility="feature_engineering")


class MLFeatureLoader:
    """
    Utility class for loading and preparing features for ML models
    """

    def __init__(
        self, storage: Optional[FeatureStorage] = None, use_consolidated: bool = True
    ):
        """
        Initialize ML feature loader
        Args:
            storage: FeatureStorage instance (creates new if None)
            use_consolidated: Whether to use consolidated yearly files (recommended)
        """
        # Preserve backward compatibility: use provided storage if available
        self.storage = storage if storage is not None else FeatureStorage()
        self.use_consolidated = use_consolidated

        if self.use_consolidated:
            # Lazily initialize consolidated storage only when requested
            self.consolidated_storage = ConsolidatedFeatureStorage()
            logger.info("MLFeatureLoader initialized with consolidated yearly files")
        else:
            logger.info("MLFeatureLoader initialized with individual ticker files")

    def _load_from_consolidated(
        self,
        ticker: Optional[str] = None,
        start_date: Optional[date] = "2024-01-01",
        end_date: Optional[date] = "2025-09-01",
        categories: Optional[List[str]] = None,
        prediction_horizon: int = 10,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Load data from consolidated yearly files with pre-calculated targets"""
        try:
            # Load consolidated features with filters
            features = self.consolidated_storage.load_consolidated_features(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                categories=categories,
            )

            if features.empty:
                raise ValueError("No features loaded from consolidated storage")

            logger.info(
                f"Loaded {len(features)} records with {len(features.columns)} features from consolidated storage"
            )

            # Extract pre-calculated future price targets
            target_column = f"Future_Close_{prediction_horizon}D"

            if target_column in features.columns:
                # Use pre-calculated future high prices as targets
                targets = features[target_column].copy()
                targets.name = f"Future_Close_{prediction_horizon}d"

                # Remove target column from features to avoid data leakage
                features_clean = features.drop(
                    columns=[
                        col for col in features.columns if col.startswith("Future_")
                    ]
                )

                # Remove rows with NaN targets
                valid_mask = targets.notna()
                features_final = features_clean[valid_mask].copy()
                targets_final = targets[valid_mask].copy()

                logger.info(
                    f"Using pre-calculated target '{target_column}': {len(targets_final)} valid samples"
                )
                logger.info(
                    f"Target range: [{targets_final.min():.2f}, {targets_final.max():.2f}]"
                )
                logger.info(
                    f"Final dataset: {len(features_final)} samples, {len(features_final.columns)} features"
                )

                return features_final, targets_final
            else:
                # Fallback to any available Future_High_XD column
                future_cols = [
                    col for col in features.columns if col.startswith("Future_High_")
                ]
                if future_cols:
                    fallback_col = future_cols[0]
                    logger.warning(
                        f"Target column '{target_column}' not found. Using '{fallback_col}' instead."
                    )

                    targets = features[fallback_col].copy()
                    targets.name = fallback_col.lower()

                    # Remove target columns from features
                    features_clean = features.drop(
                        columns=[
                            col for col in features.columns if col.startswith("Future_")
                        ]
                    )

                    # Remove rows with NaN targets
                    valid_mask = targets.notna()
                    features_final = features_clean[valid_mask].copy()
                    targets_final = targets[valid_mask].copy()

                    logger.info(
                        f"Using fallback target '{fallback_col}': {len(targets_final)} valid samples"
                    )
                    return features_final, targets_final
                else:
                    raise ValueError(
                        f"No future price target columns found. Expected '{target_column}' or similar Future_High_XD columns."
                    )

        except Exception as e:
            logger.error(f"Error loading from consolidated storage: {str(e)}")
            raise


def load_yearly_data(
    year: int, ticker: Optional[str] = None, categories: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Convenience function to load data for a specific year

    Args:
        year: Year to load (e.g., 2024, 2025)
        ticker: Single ticker symbol (None for ALL available)
        categories: Feature categories to include

    Returns:
        DataFrame with features for the specified year
    """
    start_date = date(year, 1, 1)
    end_date = date(year, 12, 31)
    loader = MLFeatureLoader(use_consolidated=True)
    features = loader.consolidated_storage.load_consolidated_features(
        ticker=ticker, start_date=start_date, end_date=end_date, categories=categories
    )
    return features


def load_all_data(ticker: Optional[str] = None) -> pd.DataFrame:
    """
    Load all data from yearly parquet files with comprehensive feature analysis

    Args:
        ticker: Single ticker symbol (None for ALL available tickers)

    Returns:
        Combined DataFrame with all loaded data
    """
    years_to_load = [2023, 2024, 2025]
    try:
        data_loader = StockDataLoader()

        # Load data for multiple years
        all_features_data = []

        for year in years_to_load:
            logger.info(f"Loading data for year {year}...")

            try:
                features = load_yearly_data(year=year, ticker=ticker)

                if not features.empty:
                    logger.info(
                        f"✓ {year} data loaded: {features.shape[0]:,} records, {features.shape[1]} columns"
                    )

                    # Add year column for tracking
                    features["data_year"] = year
                    all_features_data.append(features)

                    # Log unique tickers for this year
                    if "ticker" in features.columns:
                        year_tickers = features["ticker"].unique()
                        logger.info(
                            f"📊 Tickers in {year}: {len(year_tickers)} ({year_tickers[:5]}...)"
                        )

                    # Log date range for this year
                    if "date" in features.columns:
                        features["date"] = pd.to_datetime(features["date"])
                        date_range = f"{features['date'].min().date()} to {features['date'].max().date()}"
                        logger.info(f"📅 Date range: {date_range}")
                else:
                    logger.warning(f"No data found for {year}")

            except Exception as e:
                logger.error(f"Error loading {year} data: {str(e)}")

        # Combine all yearly data
        if all_features_data:
            logger.info("Combining and analyzing ALL yearly data...")
            combined_features = pd.concat(all_features_data, ignore_index=True)

            logger.info(
                f"✓ Combined dataset: {combined_features.shape[0]:,} total records"
            )
            logger.info(f"✓ Total features: {combined_features.shape[1]} columns")

            # Add ticker_id column from database metadata
            if "ticker" in combined_features.columns:
                logger.info("Adding ticker_id column from database metadata...")

                # Get unique tickers from the combined data
                unique_tickers = combined_features["ticker"].unique()
                logger.info("📊 Fetching metadata for all tickers in one query...")

                try:
                    # Get all ticker metadata in one efficient query
                    all_metadata_df = data_loader.get_ticker_metadata(ticker=None)

                    if not all_metadata_df.empty:
                        # Create ticker_id mapping from the DataFrame
                        ticker_id_mapping = dict(
                            zip(all_metadata_df["ticker"], all_metadata_df["id"])
                        )

                        # Map ticker_id to the combined features
                        combined_features["ticker_id"] = combined_features[
                            "ticker"
                        ].map(ticker_id_mapping)

                        # Log statistics about ticker_id mapping
                        null_ticker_ids = combined_features["ticker_id"].isnull().sum()
                        if null_ticker_ids > 0:
                            logger.warning(
                                f"⚠ {null_ticker_ids:,} records have null ticker_id"
                            )
                            missing_tickers = [
                                t for t in unique_tickers if t not in ticker_id_mapping
                            ]
                            if missing_tickers:
                                logger.warning(
                                    f"⚠ Tickers not found in database: {missing_tickers[:10]}{'...' if len(missing_tickers) > 10 else ''}"
                                )
                    else:
                        logger.error("❌ No ticker metadata retrieved from database")
                        combined_features["ticker_id"] = None

                except Exception as e:
                    logger.error(f"❌ Error fetching all ticker metadata: {str(e)}")
                    combined_features["ticker_id"] = None

            return combined_features
        else:
            logger.warning("No yearly data could be loaded")
            try:
                if data_loader is not None:
                    data_loader.close()
            except Exception as e:
                logger.warning(f"Failed to close data_loader before early return: {e}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"CRITICAL ERROR in load_all_data: {str(e)}")
        raise
    finally:
        try:
            if data_loader is not None:
                data_loader.close()
        except Exception as e:
            logger.warning(f"Failed to close data_loader in finally: {e}")
