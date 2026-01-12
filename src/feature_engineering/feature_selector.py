"""
Feature Selection and Engineering Module for Financial Time Series

This module provides comprehensive feature selection and engineering capabilities
to reduce high-dimensional financial data (~190 features) to an optimal subset
(50-80 features) while preserving predictive power.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from src.utils.core.logger import get_logger, timeit

logger = get_logger(__name__)

CATEGORIES = {
    "price_volume": [
        "close",
        "open",
        "high",
        "low",
        "volume",
        "vwap",
        "returns",
        "log_returns",
        "price_change",
        # Candlestick / price-shape features (often engineered from OHLC)
        "Gap_Up",
        "Gap_Down",
        "Body_Size",
        "Lower_Shadow",
        "Upper_Shadow",
        "Body_Shadow_Ratio",
        "Price_Range",
        "Price_Range_Pct",
        "Price_High_Low_Ratio",
        "Price_Close_Open_Ratio",
        # Future price columns sometimes appear in datasets (ensure categorization)
        "Future_Close",
        "Future_High",
    ],
    "technical_indicators": [
        "sma_5",
        "sma_10",
        "sma_20",
        "sma_50",
        "sma_200",
        "ema_5",
        "ema_10",
        "ema_20",
        "ema_50",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "bb_width",
        "stoch_k",
        "stoch_d",
        "williams_r",
        "cci_14",
        "mfi_14",
        "roc_10",
        "mom_10",
        # Token-style matches for alternate naming conventions
        "BB_",
        "BB_Squeeze",
        "BB_Above_Upper",
        "BB_Below_Lower",
        "MACD",
        "MACD_Above_Signal",
        "RSI_",
        "Stoch_",
        "Stoch_K_Above_D",
        "Williams_R_",
        "MFI_",
        # Trend / cloud indicators
        "Ichimoku_",
    ],
    "volatility": [
        "volatility_10",
        "volatility_20",
        "volatility_30",
        "atr_14",
        "bollinger_volatility",
        "parkinson_volatility",
        # Alternate ATR feature naming
        "ATR",
        "ATR_Percent",
        "ATR_Lower_Band",
        "ATR_Upper_Band",
        "ATR_Low_Volatility",
        "ATR_High_Volatility",
        # Volatility regimes / clustering
        "Vol_Regime_",
        "Vol_Clustering_",
    ],
    "momentum": [
        "momentum_1d",
        "momentum_5d",
        "momentum_10d",
        "momentum_20d",
        "trend_strength",
        "trend_direction",
        # Alternate ROC naming (e.g., ROC_10_Positive, ROC_20_Strong_Negative)
        "ROC_",
    ],
    "fundamental": [
        "pe_ratio",
        "pb_ratio",
        "ps_ratio",
        "ev_ebitda",
        "roe",
        "roa",
        "roi",
        "profit_margin",
        "debt_to_equity",
        "current_ratio",
        "quick_ratio",
        "book_value",
    ],
    "market": [
        "market_cap",
        "beta",
        "alpha",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "calmar_ratio",
    ],
    "liquidity": [
        "dollar_volume",
        "turnover_ratio",
        "bid_ask_spread",
        "market_impact",
        "liquidity_ratio",
        # Volume/flow derived indicators (treated as liquidity/participation signals)
        "OBV",
        "VPT",
        "AD_Line",
        "AD_Oscillator",
        "Volume_",
    ],
}


class FinancialFeatureSelector:
    """
    Advanced feature selection for financial time series data.

    Combines multiple selection methods:
    - Statistical tests (correlation, mutual information)
    - Model-based selection (Lasso, Random Forest)
    - Recursive feature elimination
    - Domain knowledge rules
    """

    @staticmethod
    def _matches_any_token(feature_name: str, tokens: List[str]) -> bool:
        """Return True if any token is contained within the feature name (case-insensitive)."""
        feat = str(feature_name).lower()
        return any(str(tok).lower() in feat for tok in tokens)

    def __init__(
        self,
        target_features: int = 100,
        correlation_threshold: float = 0.95,
        importance_threshold: float = 0.01,
        variance_threshold: float = 1e-4,
        random_state: int = 42,
    ):
        """Initialize feature selector."""
        (
            self.target_features,
            self.correlation_threshold,
            self.importance_threshold,
            self.variance_threshold,
            self.random_state,
        ) = (
            target_features,
            correlation_threshold,
            importance_threshold,
            variance_threshold,
            random_state,
        )
        self.selected_features, self.feature_importance, self.selection_report = [], {}, {}
        logger.info(f"Initialized feature selector targeting {target_features} features")

    @timeit
    def select_features(
        self, X: pd.DataFrame, y: pd.Series, method: str = "lasso", return_scores: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]] | pd.DataFrame:
        """Select optimal features using specified method."""
        logger.info(f"Selecting features using {method}, Input: {X.shape}")

        supported_methods = {
            "correlation",
            "mutual_info",
            "lasso",
            "rf",
            "random_forest",
            "rfe",
            "ensemble",
        }
        if method not in supported_methods:
            raise ValueError(f"Unknown selection method: {method}")

        selected_features, scores = getattr(self, f"_select_by_{method}")(X, y)
        selected_features = self._apply_domain_filtering(selected_features, scores)[
            : self.target_features
        ]

        x_selected = X[selected_features]
        self.selected_features, self.feature_importance = selected_features, scores
        self.selection_report = {
            "method": method,
            "original_features": X.shape[1],
            "selected_features": len(selected_features),
            "reduction_ratio": len(selected_features) / X.shape[1],
            "top_features": sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10],
            "feature_categories": {
                cat: sum(1 for f in selected_features if self._matches_any_token(f, cat_feats))
                for cat, cat_feats in self.CATEGORIES.items()
                if any(self._matches_any_token(f, cat_feats) for f in selected_features)
            },
            "scores_summary": {
                k: getattr(np, k.split("_")[0])(list(scores.values()))
                for k in ["mean_score", "std_score", "min_score", "max_score"]
            },
        }

        logger.info(f"Selected {len(selected_features)} features")
        return (x_selected, self.selection_report) if return_scores else x_selected

    def _take_last_rows_ordered_by_date_int(
        self, X: pd.DataFrame, y: pd.Series, n_rows: int = 300_000, date_col: str = "date_int"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Take the last n_rows ordered by date_int.

        Falls back to DataFrame tail if date_int is missing.
        Uses O(n) selection (argpartition) to avoid sorting the full dataset.
        """
        if not hasattr(X, "shape") or X.shape[0] <= n_rows:
            return X, y

        if hasattr(X, "columns") and date_col in X.columns:
            date_vals = X[date_col].to_numpy(copy=False)
            date_vals = np.nan_to_num(date_vals, nan=-np.inf)
            idx = np.argpartition(date_vals, -n_rows)[-n_rows:]
            idx = idx[np.argsort(date_vals[idx], kind="mergesort")]
            x_sample = X.iloc[idx]
            y_sample = y.iloc[idx] if hasattr(y, "iloc") else y[idx]
            return x_sample, y_sample

        x_sample = X.tail(n_rows)
        y_sample = y.loc[x_sample.index] if hasattr(y, "loc") else y[-n_rows:]
        return x_sample, y_sample

    def _get_numeric(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get numeric DataFrame."""
        return (
            X.select_dtypes(include=[np.number]).astype(np.float32)
            if hasattr(X, "select_dtypes")
            else X
        )

    @timeit
    def _select_by_correlation(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[List[str], Dict[str, float]]:
        """Select features based on correlation with target."""
        logger.info(
            f"Starting correlation-based selection on {X.shape[0]} samples, {X.shape[1]} features"
        )
        X_num = self._get_numeric(X)
        correlations = (
            X_num.corrwith(y, method="spearman").abs().to_dict()
            if hasattr(X_num, "corrwith")
            else {col: abs(spearmanr(X_num[col], y)[0]) for col in X_num.columns}
        )

        selected = []
        for feat, _ in sorted(correlations.items(), key=lambda x: x[1], reverse=True):
            if len(selected) >= self.target_features:
                break
            if not any(
                abs(X_num[feat].corr(X_num[s])) > self.correlation_threshold for s in selected
            ):
                selected.append(feat)

        scores_selected = {f: correlations[f] for f in selected}
        logger.info(
            f"Correlation method selected {len(selected)} features, score range: [{min(scores_selected.values()):.4f}, {max(scores_selected.values()):.4f}]"
        )
        return selected, scores_selected

    @timeit
    def _select_by_mutual_info(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[List[str], Dict[str, float]]:
        """Select features using tree-based importance on a sample.

        This replaces mutual information for large datasets (significantly faster).
        """
        logger.info(
            f"Starting mutual info selection on {X.shape[0]} samples, {X.shape[1]} features"
        )
        x_sample, y_sample = self._take_last_rows_ordered_by_date_int(X, y)
        logger.info(f"Sampled {x_sample.shape[0]} rows for ExtraTreesRegressor")
        x_num = self._get_numeric(x_sample)

        logger.info(
            f"Training ExtraTreesRegressor with 100 estimators on {x_num.shape[1]} features"
        )
        model = ExtraTreesRegressor(
            n_estimators=100, random_state=self.random_state, n_jobs=-1
        ).fit(x_num, y_sample)

        scores = dict(zip(x_num.columns, model.feature_importances_))
        selected = [
            f
            for f, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[
                : self.target_features
            ]
        ]
        scores_selected = {f: scores[f] for f in selected}
        logger.info(
            f"Mutual info method selected {len(selected)} features, score range: [{min(scores_selected.values()):.4f}, {max(scores_selected.values()):.4f}]"
        )
        return selected, scores_selected

    @timeit
    def _select_by_lasso(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], Dict[str, float]]:
        """Select features using Lasso regression."""
        logger.info(f"Starting Lasso selection on {X.shape[0]} samples, {X.shape[1]} features")
        x_sample, y_sample = self._take_last_rows_ordered_by_date_int(X, y)
        logger.info(f"Sampled {x_sample.shape[0]} rows for Lasso regression")
        x_num = self._get_numeric(x_sample)
        logger.info(f"Scaling {x_num.shape[1]} features for Lasso")
        x_scaled = pd.DataFrame(StandardScaler().fit_transform(x_num), columns=x_num.columns)
        logger.info("Training LassoCV with 3-fold CV, max_iter=10000")
        lasso = LassoCV(cv=3, random_state=self.random_state, max_iter=10000).fit(
            x_scaled, y_sample
        )
        logger.info(f"Lasso converged with alpha={lasso.alpha_:.6f}")
        scores = dict(zip(x_num.columns, np.abs(lasso.coef_)))
        selected = [f for f, c in scores.items() if c > self.importance_threshold] or [
            f
            for f, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[
                : self.target_features
            ]
        ]
        final_selected = selected[: self.target_features]
        scores_selected = {f: scores[f] for f in final_selected}
        logger.info(
            f"Lasso method selected {len(final_selected)} features, score range: [{min(scores_selected.values()):.4f}, {max(scores_selected.values()):.4f}]"
        )
        return final_selected, scores_selected

    @timeit
    def _select_by_random_forest(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[List[str], Dict[str, float]]:
        """Select features using Random Forest importance."""
        logger.info(
            f"Starting Random Forest selection on {X.shape[0]} samples, {X.shape[1]} features"
        )
        x_sample, y_sample = self._take_last_rows_ordered_by_date_int(X, y)
        logger.info(f"Sampled {x_sample.shape[0]} rows for Random Forest")
        x_num = self._get_numeric(x_sample)
        logger.info(
            f"Training RandomForestRegressor with 100 estimators on {x_num.shape[1]} features"
        )
        rf = RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1).fit(
            x_num, y_sample
        )
        scores = dict(zip(x_num.columns, rf.feature_importances_))
        selected = [
            f
            for f, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[
                : self.target_features
            ]
        ]
        scores_selected = {f: scores[f] for f in selected}
        logger.info(
            f"Random Forest method selected {len(selected)} features, score range: [{min(scores_selected.values()):.4f}, {max(scores_selected.values()):.4f}]"
        )
        return selected, scores_selected

    def _select_by_rf(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], Dict[str, float]]:
        """Alias for Random Forest selector (kept for backward compatibility)."""
        return self._select_by_random_forest(X, y)

    @timeit
    def _select_by_rfe(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], Dict[str, float]]:
        """Select features using Recursive Feature Elimination."""
        logger.info(f"Starting RFE selection on {X.shape[0]} samples, {X.shape[1]} features")
        x_sample, y_sample = self._take_last_rows_ordered_by_date_int(X, y)
        logger.info(f"Sampled {x_sample.shape[0]} rows for RFE")
        X_num = self._get_numeric(x_sample)
        logger.info(
            f"Running RFE with RandomForest (50 estimators), target={self.target_features} features, step=10"
        )
        rfe = RFE(
            RandomForestRegressor(n_estimators=50, random_state=self.random_state, n_jobs=-1),
            n_features_to_select=self.target_features,
            step=10,
        ).fit(X_num, y_sample)
        selected = X_num.columns[rfe.support_].tolist()
        scores_selected = dict(zip(selected, rfe.estimator_.feature_importances_))
        logger.info(
            f"RFE method selected {len(selected)} features, score range: [{min(scores_selected.values()):.4f}, {max(scores_selected.values()):.4f}]"
        )
        return selected, scores_selected

    @timeit
    def _select_by_ensemble(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[List[str], Dict[str, float]]:
        """Select features using ensemble of methods."""
        logger.info(
            "Starting ensemble selection with 4 methods: correlation, mutual_info, lasso, rf"
        )
        all_scores = {}
        for method in ["correlation", "mutual_info", "lasso", "rf"]:
            logger.info(f"Running ensemble component: {method}")
            method_selected, method_scores = getattr(self, f"_select_by_{method}")(X, y)
            logger.info(f"Method {method} contributed {len(method_scores)} features to ensemble")
            for feat, score in method_scores.items():
                all_scores.setdefault(feat, []).append(score)

        logger.info(f"Aggregating scores from {len(all_scores)} unique features across all methods")
        ensemble_scores = {f: np.mean(s) for f, s in all_scores.items()}
        selected = [
            f
            for f, _ in sorted(ensemble_scores.items(), key=lambda x: x[1], reverse=True)[
                : self.target_features
            ]
        ]
        scores_selected = {f: ensemble_scores[f] for f in selected}
        logger.info(
            f"Ensemble method selected {len(selected)} features, score range: [{min(scores_selected.values()):.4f}, {max(scores_selected.values()):.4f}]"
        )
        return selected, scores_selected

    def _apply_domain_filtering(self, features: List[str], scores: Dict[str, float]) -> List[str]:
        """Apply domain knowledge filtering to ensure balanced feature selection."""
        category_best = {
            cat: max(
                [f for f in features if self._matches_any_token(f, cat_feats)],
                key=lambda x: scores.get(x, 0),
                default=None,
            )
            for cat, cat_feats in self.CATEGORIES.items()
        }
        # De-duplicate while preserving order (a feature may match multiple categories).
        selected_by_cat = []
        seen = set()
        for f in category_best.values():
            if f and f not in seen:
                selected_by_cat.append(f)
                seen.add(f)
        remaining = [f for f in features if f not in selected_by_cat]
        logger.info(f"Domain filtering: {len(selected_by_cat)} categories represented")
        return selected_by_cat + remaining

    def get_feature_importance_report(self) -> Dict[str, Any]:
        """Get detailed feature importance report."""
        return (
            {
                "selected_features": self.selected_features,
                "importance_scores": self.feature_importance,
                "top_10_features": sorted(
                    self.feature_importance.items(), key=lambda x: x[1], reverse=True
                )[:10],
                "selection_report": self.selection_report,
            }
            if self.feature_importance
            else {}
        )


def main():
    """Run feature selection on training data from Polars ML pipeline."""
    from src.data_utils.polars_ml_pipeline import (
        prepare_ml_data_for_training_with_cleaning_polars,
        to_pandas,
    )

    logger.info("Loading training data with Polars ML pipeline...")
    data = prepare_ml_data_for_training_with_cleaning_polars(
        prediction_horizon=20, split_date="2025-05-01", clean_features=False
    )

    logger.info("Converting Polars to pandas for feature selection...")
    x_train_pd = to_pandas(data["x_train"])
    y_train_pd = to_pandas(data["y_train"])
    x_test_pd = to_pandas(data["x_test"])

    logger.info(f"Training data shape: {x_train_pd.shape}, Test data shape: {x_test_pd.shape}")

    selector = FinancialFeatureSelector(target_features=100, correlation_threshold=0.95)

    logger.info("Running ensemble feature selection...")
    x_train_selected, report = selector.select_features(
        x_train_pd, y_train_pd, method="ensemble", return_scores=True
    )

    x_test_selected = x_test_pd[selector.selected_features]

    logger.info(f"Selected {len(selector.selected_features)} features from {x_train_pd.shape[1]}")
    logger.info(f"Training set: {x_train_selected.shape}")
    logger.info(f"Test set: {x_test_selected.shape}")

    logger.info("Top 20 features by importance:")
    for i, (feat, score) in enumerate(report["top_features"][:20], 1):
        logger.info(f"  {i}. {feat}: {score:.4f}")

    logger.info("Feature distribution by category:")
    for cat, count in report["feature_categories"].items():
        logger.info(f"  {cat}: {count} features")

    logger.info(
        "Feature selection complete, features selected: " + ", ".join(selector.selected_features)
    )


if __name__ == "__main__":
    main()
