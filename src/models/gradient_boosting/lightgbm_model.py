import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import optuna
import pandas as pd

from src.models.base_model import (
    BaseModel,
    load_and_prepare_data_polars_as_pandas,
)
from src.models.evaluation import CustomMetrics, ThresholdEvaluator
from src.feature_engineering.feature_selector import CATEGORIES
from src.utils.core.logger import get_logger, timeit
from src.utils.data.cleaned_data_cache import collect_garbage
from src.utils.mlops.mlflow_utils import MLFlowManager

logger = get_logger(__name__)
experiment_name = "lightgbm_stock_predictor"
threshold_evaluator = ThresholdEvaluator()


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _extract_feature_names_from_signature(model_uri: str) -> Optional[List[str]]:
    try:
        from mlflow.models import get_model_info

        info = get_model_info(model_uri)
        sig = getattr(info, "signature", None)
        inputs = getattr(sig, "inputs", None) if sig else None
        if not inputs:
            return None
        if hasattr(inputs, "inputs"):
            names = [i.name for i in inputs.inputs if getattr(i, "name", None)]
            return names or None
        schema = getattr(inputs, "schema", None)
        return list(schema.input_names) if schema and hasattr(schema, "input_names") else None
    except Exception:
        return None


class LightGBMModel(BaseModel):
    def __init__(
        self,
        model_name: str = "lightgbm_stock_predictor",
        config: Optional[Dict[str, Any]] = None,
        prediction_horizon: int = 10,
        threshold_evaluator: Optional[ThresholdEvaluator] = None,
    ):
        cfg = dict(config or {})
        cfg["prediction_horizon"] = prediction_horizon
        super().__init__(model_name, cfg, threshold_evaluator=threshold_evaluator)

        self.prediction_horizon = self.config.get("prediction_horizon", 5)
        self.model = None
        self.feature_names = None
        self.categorical_features = None
        self.early_stopping_rounds = self.config.get("early_stopping_rounds", 50)
        self.eval_metric = self.config.get("eval_metric", "rmse")
        self.base_threshold = 0.5
        self.default_confidence_method = "leaf_depth"

        total_cores = os.cpu_count() or 4
        self.max_cpu_cores = max(1, int(total_cores * 0.85))
        logger.info(f"CPU usage limit: {self.max_cpu_cores}/{total_cores} cores (85%)")
        self.custom_metrics = CustomMetrics()

    def _create_model(
        self, params: Optional[Dict[str, Any]] = None, model_name: Optional[str] = None, **kwargs
    ) -> "LightGBMModel":
        params = params or {}
        model_name = model_name or f"{self.model_name}_configured"
        new_model = LightGBMModel(
            model_name=model_name,
            config={**params, "investment_amount": self.threshold_evaluator.investment_amount},
            prediction_horizon=self.prediction_horizon,
            threshold_evaluator=self.threshold_evaluator,
        )
        new_model.early_stopping_rounds, new_model.eval_metric = (
            self.early_stopping_rounds,
            self.eval_metric,
        )
        logger.info(f"Created new LightGBM model '{model_name}' with parameters: {params}")
        return new_model

    def _identify_categorical_features(self, X: pd.DataFrame) -> List[str]:
        cats: List[str] = ["ticker_id"] if "ticker_id" in X.columns else []
        for col in X.columns:
            if col == "ticker_id":
                continue
            s, dt = X[col], X[col].dtype
            if (
                dt is object
                or getattr(dt, "name", "") == "category"
                or (getattr(dt, "kind", "") == "i" and s.nunique() < 50)
            ):
                cats.append(col)
        logger.info(f"Identified {len(cats)} categorical features: {cats}")
        return cats

    def _validate_training_data(self, X: pd.DataFrame) -> None:
        constant_cols = [c for c in X.columns if X[c].nunique(dropna=False) == 1]
        if not constant_cols:
            return
        logger.error("Data validation failed: Constant features detected!")
        logger.error(f"Number of constant features: {len(constant_cols)} out of {len(X.columns)}")
        logger.error(f"Constant feature names (sample): {constant_cols[:10]}")
        ratio = len(constant_cols) / max(1, len(X.columns))
        if ratio > 0.5:
            raise ValueError(
                f"CRITICAL: {ratio:.1%} of features are constant. This will lead to identical predictions. Please check the data preparation pipeline (Polars: prepare_ml_data_for_training_with_cleaning_polars). Aborting training."
            )
        logger.warning(
            "A significant number of features are constant. Model performance will likely be poor."
        )

    def _prepare_data(
        self, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series
    ) -> Tuple[lgb.Dataset, lgb.Dataset]:
        self.feature_names = list(x_train.columns)
        self.categorical_features = self._identify_categorical_features(x_train)
        train = lgb.Dataset(
            x_train,
            label=y_train,
            feature_name=self.feature_names,
            categorical_feature=self.categorical_features,
        )
        test = lgb.Dataset(
            x_test,
            label=y_test,
            feature_name=self.feature_names,
            categorical_feature=self.categorical_features,
            reference=train,
        )
        return train, test

    @timeit
    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        params: Optional[Dict[str, Any]] = None,
    ) -> "LightGBMModel":
        self._validate_training_data(x_train)
        train_data, test_data = self._prepare_data(x_train, y_train, x_test, y_test)
        params = dict((params or self.hyperparameter_config.get_default_params()) or {})
        n_estimators = params.pop("n_estimators", 1000)
        params.setdefault("random_state", 42)
        params.update(
            {
                "objective": "regression",
                "metric": self.eval_metric,
                "verbose": -1,
                "num_threads": self.max_cpu_cores,
            }
        )
        callbacks = [
            lgb.early_stopping(stopping_rounds=self.early_stopping_rounds),
            lgb.log_evaluation(period=0),
        ]
        self.model = lgb.train(
            params=params,
            train_set=train_data,
            valid_sets=[train_data, test_data],
            valid_names=["train", "test"],
            num_boost_round=n_estimators,
            callbacks=callbacks,
        )
        logger.info(
            f"Training completed. Best iteration: {self.model.best_iteration}, Best score: {self.model.best_score['test'][self.eval_metric]:.4f}"
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model must be trained before making predictions")
        if self.feature_names and list(X.columns) != self.feature_names:
            logger.warning("Feature names don't match training data")
        preds = self.model.predict(X, num_iteration=self.model.best_iteration)
        logger.info(f"Generated {len(preds)} predictions")
        return preds

    def get_feature_importance(self, importance_type: str = "split") -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model must be trained before getting feature importance")
        scores = self.model.feature_importance(importance_type=importance_type)
        feats = self.feature_names or [f"feature_{i}" for i in range(len(scores))]
        return pd.DataFrame({"feature": feats, "importance": scores}).sort_values(
            "importance", ascending=False
        )

    def objective(
        self, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series
    ) -> Callable[[optuna.trial.Trial], float]:
        (
            self.best_investment_success_rate,
            self.best_trial_model,
            self.best_trial_params,
            self.best_threshold_info,
        ) = -np.inf, None, None, None

        def _trial_objective(trial: optuna.trial.Trial) -> float:
            # constraint_strategy = trial.suggest_categorical(
            #     "interaction_constraint_strategy", ["none", "strict", "moderate", "loose"]
            # )
            constraint_strategy = "none"
            params = {
                "boosting_type": trial.suggest_categorical(
                    "boosting_type", ["gbdt", "goss"]
                ),  # dart
                "max_depth": trial.suggest_int("max_depth", 9, 18),
                "learning_rate": trial.suggest_float("learning_rate", 0.00001, 0.005, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 500, 15000),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0, step=0.1),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0, step=0.1),
                "min_child_weight": trial.suggest_float("min_child_weight", 1, 30, step=0.1),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
                "path_smooth": trial.suggest_float("path_smooth", 0.0, 1.0),
                "num_leaves": trial.suggest_int("num_leaves", 10, 500),
            }

            # Apply interaction constraints based on strategy
            if constraint_strategy != "none":
                feature_names = list(x_train.columns)
                interaction_groups = self._create_financial_interaction_groups(
                    feature_names, strategy=constraint_strategy
                )
                if interaction_groups:
                    params["interaction_constraints"] = interaction_groups
                    logger.info(
                        f"Trial {trial.number}: Applying {constraint_strategy} interaction constraints "
                        f"with {len(interaction_groups)} groups"
                    )
                    trial.set_user_attr("used_interaction_constraints", True)
                    trial.set_user_attr("num_constraint_groups", len(interaction_groups))
                    trial.set_user_attr("constraint_strategy", constraint_strategy)
                else:
                    logger.warning(f"Trial {trial.number}: No interaction groups created")
                    trial.set_user_attr("used_interaction_constraints", False)
            else:
                trial.set_user_attr("used_interaction_constraints", False)
                trial.set_user_attr("constraint_strategy", "none")

            # Helpful metadata for later dashboard analysis.
            self._optuna_log_dataset_sizes(trial, x_train=x_train, x_test=x_test)
            try:
                trial_model = self._create_model(
                    params=params, model_name=f"lightgbm_trial_{trial.number}"
                )
                trial_model.disable_mlflow = True
                trial_model.fit(x_train, y_train, x_test, y_test, params=params)

                logger.info(f"Running threshold optimization for trial {trial.number}")
                res = trial_model.optimize_prediction_threshold(
                    x_test=x_test,
                    y_test=y_test,
                    current_prices_test=x_test["close"].values,
                    confidence_method="leaf_depth",
                    threshold_range=(0.01, 0.99),
                    n_thresholds=150,
                )
                best = res["best_result"]
                score = float(best["test_profit_per_investment"])
                info = {
                    "optimal_threshold": res.get("optimal_threshold"),
                    "samples_kept_ratio": best.get("test_samples_ratio"),
                    "investment_success_rate": best.get("investment_success_rate"),
                    "test_profit_per_investment": best.get("test_profit_per_investment"),
                    "custom_accuracy": best.get("test_custom_accuracy"),
                    "total_threshold_profit": best.get("test_profit"),
                    "profitable_investments": best.get("profitable_investments"),
                }

                # Persist key metrics to Optuna trial information (user attrs).
                best_iteration = getattr(
                    getattr(trial_model, "model", None), "best_iteration", None
                )
                best_score = None
                bs = getattr(getattr(trial_model, "model", None), "best_score", None)
                if isinstance(bs, dict) and "test" in bs and self.eval_metric in bs["test"]:
                    best_score = bs["test"][self.eval_metric]
                self._optuna_log_trial_core_metrics(
                    trial,
                    profit_per_investment=info.get("test_profit_per_investment") or score,
                    best_iteration=best_iteration,
                    best_score=best_score,
                )
                self._optuna_log_threshold_metrics(
                    trial,
                    optimal_threshold=info.get("optimal_threshold"),
                    samples_kept_ratio=info.get("samples_kept_ratio"),
                    investment_success_rate=info.get("investment_success_rate"),
                    custom_accuracy=info.get("custom_accuracy"),
                    total_profit=info.get("total_threshold_profit"),
                    profitable_investments=info.get("profitable_investments"),
                )

                if score > self.best_investment_success_rate:
                    (
                        self.best_investment_success_rate,
                        self.best_trial_model,
                        self.best_trial_params,
                        self.best_threshold_info,
                    ) = score, trial_model, params.copy(), info.copy()
                    self.model, self.feature_names = trial_model.model, trial_model.feature_names
                    if info.get("optimal_threshold") is not None:
                        self.optimal_threshold, self.confidence_method = (
                            info["optimal_threshold"],
                            "leaf_depth",
                        )
                    logger.info(
                        f"NEW BEST TRIAL {trial.number}: Profit Per Investment = {score:.3f}"
                    )
                else:
                    logger.info(
                        f"Trial {trial.number}: Profit Per Investment = {score:.3f} (Best: {self.best_investment_success_rate:.3f})"
                    )
                return score
            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {e}")
                return -1e6

        return _trial_objective

    def get_best_trial_info(self) -> Dict[str, Any]:
        if not hasattr(self, "best_investment_success_rate"):
            return {"message": "No hyperparameter optimization has been run yet"}
        base: Dict[str, Any] = {
            "best_investment_success_rate": self.best_investment_success_rate,
            "best_trial_params": self.best_trial_params,
            "has_best_model": self.best_trial_model is not None,
            "model_updated": self.model is not None,
        }
        t = getattr(self, "best_threshold_info", None)
        base["threshold_optimization"] = (
            {
                "optimal_threshold": t.get("optimal_threshold"),
                "samples_kept_ratio": t.get("samples_kept_ratio"),
                "investment_success_rate": t.get("investment_success_rate"),
                "custom_accuracy": t.get("custom_accuracy"),
                "total_threshold_profit": t.get("total_threshold_profit"),
                "profitable_investments": t.get("profitable_investments"),
                "confidence_method": getattr(self, "confidence_method", "leaf_depth"),
            }
            if t
            else None
        )
        return base

    def finalize_best_model(self) -> None:
        if not getattr(self, "best_trial_model", None):
            logger.warning("No best model found to finalize")
            return
        self.model, self.feature_names = (
            self.best_trial_model.model,
            self.best_trial_model.feature_names,
        )
        t = getattr(self, "best_threshold_info", None) or {}
        opt = t.get("optimal_threshold")
        if opt is not None:
            self.optimal_threshold, self.confidence_method = (
                opt,
                getattr(self, "confidence_method", "leaf_depth"),
            )
        logger.info(
            f"Best model finalized with investment success rate: {self.best_investment_success_rate:.3f}"
        )
        logger.info(f"Best parameters: {self.best_trial_params}")
        if getattr(self, "disable_mlflow", False):
            return
        self.log_params({f"best_{k}": v for k, v in (self.best_trial_params or {}).items()})
        metrics: Dict[str, Any] = {
            "best_investment_success_rate": self.best_investment_success_rate,
            "hypertuning_completed": 1,
        }
        if opt is not None:
            metrics.update(
                {
                    "best_optimal_threshold": opt,
                    "best_samples_kept_ratio": t.get("samples_kept_ratio"),
                    "best_investment_success_rate": t.get("investment_success_rate"),
                    "best_custom_accuracy": t.get("custom_accuracy"),
                    "best_total_threshold_profit": t.get("total_threshold_profit"),
                    "best_profitable_investments": t.get("profitable_investments"),
                }
            )
        self.log_metrics(metrics)

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        if self.model is None:
            return info
        info.update(
            {
                "n_features": self.model.num_feature(),
                "best_iteration": self.model.best_iteration,
                "best_score": self.model.best_score,
                "categorical_features": self.categorical_features,
                "early_stopping_rounds": self.early_stopping_rounds,
                "eval_metric": self.eval_metric,
            }
        )
        return info

    def save_model(
        self,
        metrics: Dict[str, float],
        params: Dict[str, Any],
        x_eval: pd.DataFrame,
        experiment_name: str = None,
    ) -> str:
        if self.model is None:
            raise RuntimeError("No trained model to save")
        run_id = self.log_model_to_mlflow(
            metrics=metrics, params=params, x_eval=x_eval, experiment_name=experiment_name
        )
        logger.info(f"LightGBM model saved using universal logging function. Run ID: {run_id}")
        return run_id

    def load_model(self, run_id: str) -> None:
        logger.info(f"Loading LightGBM model from MLflow run: {run_id}")
        client = mlflow.tracking.MlflowClient()
        run_info = client.get_run(run_id)
        artifacts = client.list_artifacts(run_id)
        model_dir = next(
            (a.path for a in artifacts if a.is_dir and a.path == "model"), None
        ) or next((a.path for a in artifacts if a.is_dir and "model" in a.path.lower()), "model")
        if model_dir == "model":
            logger.warning("No model artifact found, defaulting to 'model' path")
        model_uri = f"runs:/{run_id}/{model_dir}"
        logger.info(f"Loading model from URI: {model_uri}")
        self.model = mlflow.lightgbm.load_model(model_uri)
        self._load_metadata_from_run(run_info)
        logger.info(f"LightGBM model loaded successfully from MLflow run: {run_id}")

    def _load_metadata_from_run(self, run_info) -> None:
        try:
            p = run_info.data.params
            self.model_name = p.get("model_model_name", self.model_name)
            self.prediction_horizon = _safe_int(
                p.get("model_prediction_horizon"), self.prediction_horizon
            )
            self.early_stopping_rounds = _safe_int(
                p.get("model_early_stopping_rounds"), self.early_stopping_rounds
            )
            self.eval_metric = p.get("model_eval_metric", self.eval_metric)
            names = _extract_feature_names_from_signature(f"runs:/{run_info.info.run_id}/model")
            if names:
                self.feature_names = names
                logger.info(f"Loaded {len(names)} feature names from MLflow signature")
            logger.info("Model metadata loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load all metadata: {str(e)}")

    @classmethod
    def load_from_mlflow(cls, run_id: str) -> "LightGBMModel":
        m = cls()
        m.load_model(run_id=run_id)
        return m

    def log_model_to_mlflow(
        self,
        metrics: Dict[str, float],
        params: Dict[str, Any],
        x_eval: pd.DataFrame,
        experiment_name: str = None,
    ) -> str:
        if self.model is None:
            raise RuntimeError("No trained model to log")
        run_id = log_to_mlflow_lightgbm(
            model=self.model,
            metrics=metrics,
            params=params,
            experiment_name=experiment_name or f"{self.model_name}_experiment",
            x_eval=x_eval,
        )
        logger.info(f"Model logged to MLflow using universal function. Run ID: {run_id}")
        return run_id

    def get_prediction_confidence(self, X: pd.DataFrame, method: str = "leaf_depth") -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model must be trained before calculating confidence")
        if method == "leaf_depth":
            leaves = self.model.predict(X, pred_leaf=True, num_iteration=self.model.best_iteration)
            scores = np.power(np.mean(leaves, axis=1), 2)
        elif method == "margin":
            scores = np.abs(self.model.predict(X, num_iteration=self.model.best_iteration))
        elif method == "variance":
            n_trees = self.model.best_iteration or 100
            preds = [
                self.model.predict(X, num_iteration=i) for i in range(10, min(n_trees, 100), 10)
            ]
            scores = 1.0 / (1.0 + np.var(np.array(preds), axis=0))
        elif method == "simple":
            pred = self.model.predict(X, num_iteration=self.model.best_iteration)
            cur = X["close"].values if "close" in X.columns else np.ones_like(pred)
            scores = np.clip(np.abs(pred - cur) / cur, 0, 1)
        else:
            raise ValueError(f"Unknown confidence method: {method}")
        mn, mx = float(np.min(scores)), float(np.max(scores))
        scores = (scores - mn) / (mx - mn) if mx > mn else np.full_like(scores, 0.5)
        logger.info(f"Final confidence - Range: [{scores.min():.4f}, {scores.max():.4f}]")
        logger.info(f"Final confidence - Mean: {scores.mean():.4f}, std: {scores.std():.4f}")
        return scores

    def select_features(
        self, X: pd.DataFrame, y: pd.Series, n_features_to_select: int = 50
    ) -> List[str]:
        logger.info(
            f"Starting feature selection to find the best {n_features_to_select} features..."
        )
        prelim_params = {
            "objective": "regression",
            "metric": "rmse",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "verbose": -1,
            "num_threads": self.max_cpu_cores,
            "seed": 42,
        }
        train = lgb.Dataset(
            X,
            label=y,
            feature_name=list(X.columns),
            categorical_feature=self._identify_categorical_features(X),
        )
        prelim = lgb.train(params=prelim_params, train_set=train)
        imp = pd.DataFrame(
            {"feature": X.columns, "importance": prelim.feature_importance(importance_type="gain")}
        ).sort_values("importance", ascending=False)
        logger.info("Top 10 features from preliminary model (by gain):")
        for _, row in imp.head(10).iterrows():
            logger.info(f"- {row['feature']}: {row['importance']:.2f}")
        selected = list(imp["feature"].head(n_features_to_select))
        for col in ("close", "ticker_id"):
            if col in X.columns and col not in selected:
                removed = selected.pop()
                selected.append(col)
                logger.info(f"Added required column '{col}', removed '{removed}'")
        logger.info(f"Feature selection complete. Selected {len(selected)} features.")
        return selected

    @staticmethod
    def _matches_any_token(feature_name: str, tokens: List[str]) -> bool:
        """Check if feature name contains any of the given tokens (case-insensitive)."""
        feat_lower = str(feature_name).lower()
        return any(str(tok).lower() in feat_lower for tok in tokens)

    def _create_financial_interaction_groups(
        self, feature_names: List[str], strategy: str = "moderate"
    ) -> List[List[int]]:
        """
        Create financially-motivated interaction constraint groups using CATEGORIES.

        Args:
            feature_names: List of feature names in the dataset
            strategy: Constraint strategy - "strict", "moderate", or "loose"
                - strict: Separate groups with minimal cross-interaction
                - moderate: Some cross-group interactions for related categories
                - loose: Flexible grouping with broad interactions

        Returns:
            List of feature index groups that can interact within trees
        """
        logger.info(
            f"Creating {strategy} interaction constraints for {len(feature_names)} features"
        )

        # Map features to categories using CATEGORIES from feature_selector
        feature_to_category = {}
        for idx, feat in enumerate(feature_names):
            for category, tokens in CATEGORIES.items():
                if self._matches_any_token(feat, tokens):
                    feature_to_category[idx] = category
                    break

        # Group feature indices by category
        category_indices = {}
        for idx, category in feature_to_category.items():
            category_indices.setdefault(category, []).append(idx)

        logger.info(f"Categorized {len(feature_to_category)}/{len(feature_names)} features")
        for cat, indices in category_indices.items():
            logger.info(f"  {cat}: {len(indices)} features")

        # Uncategorized features (date_int, ticker_id, etc.)
        uncategorized = [i for i in range(len(feature_names)) if i not in feature_to_category]
        if uncategorized:
            logger.info(f"  uncategorized: {len(uncategorized)} features")

        # Create interaction groups based on strategy
        if strategy == "strict":
            # Each category is its own group, no cross-category interaction
            groups = list(category_indices.values())
            # Add uncategorized features to their own group
            if uncategorized:
                groups.append(uncategorized)

        elif strategy == "moderate":
            # Merge financially related categories
            groups = []

            # Core market microstructure: price_volume + volatility + liquidity
            core_group = (
                category_indices.get("price_volume", [])
                + category_indices.get("volatility", [])
                + category_indices.get("liquidity", [])
            )
            if core_group:
                groups.append(core_group)

            # Momentum + technical indicators (derived signals)
            signals_group = category_indices.get("momentum", []) + category_indices.get(
                "technical_indicators", []
            )
            if signals_group:
                groups.append(signals_group)

            # Fundamentals + market metrics
            fundamentals_group = category_indices.get("fundamental", []) + category_indices.get(
                "market", []
            )
            if fundamentals_group:
                groups.append(fundamentals_group)

            # Uncategorized features can interact with core group
            if uncategorized and groups:
                groups[0].extend(uncategorized)
            elif uncategorized:
                groups.append(uncategorized)

        elif strategy == "loose":
            # Most features can interact, only separate fundamentals from technicals
            groups = []

            # Technical group: price, volume, technical indicators, momentum, volatility
            technical_group = (
                category_indices.get("price_volume", [])
                + category_indices.get("technical_indicators", [])
                + category_indices.get("momentum", [])
                + category_indices.get("volatility", [])
                + category_indices.get("liquidity", [])
                + uncategorized
            )
            if technical_group:
                groups.append(technical_group)

            # Fundamental group (separate to avoid mixing with technical signals)
            fundamental_group = category_indices.get("fundamental", []) + category_indices.get(
                "market", []
            )
            if fundamental_group:
                groups.append(fundamental_group)
                # Allow fundamentals to also see some key technical features
                if technical_group:
                    # Add first 20 technical features to fundamental group
                    fundamental_group.extend(technical_group[:20])

        else:
            logger.warning(f"Unknown strategy '{strategy}', using moderate as default")
            return self._create_financial_interaction_groups(feature_names, "moderate")

        # Filter out empty groups
        groups = [g for g in groups if g]

        logger.info(
            f"Created {len(groups)} interaction constraint groups using {strategy} strategy"
        )
        for i, group in enumerate(groups, 1):
            logger.info(f"  Group {i}: {len(group)} features")

        return groups if groups else None


def log_to_mlflow_lightgbm(model, metrics, params, experiment_name, x_eval):
    try:
        pip_requirements = [
            "lightgbm>=3.3.0",
            "pandas>=1.5.0",
            "numpy>=1.21.0",
            "scikit-learn>=1.0.0",
            "mlflow>=2.0.0",
        ]
        try:
            mlflow.end_run()
            logger.info("Ended existing MLflow run before starting new one")
        except Exception as e:
            logger.info(f"No existing run to end: {e}")
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(
            run_name=f"lightgbm_final_{datetime.now().strftime('%Y%m%d_%H%M')}"
        ) as run:
            mlflow.log_params(
                {
                    k: v
                    for k, v in params.items()
                    if k not in {"device", "objective", "verbosity", "seed", "nthread", "verbose"}
                }
            )
            mlflow.log_metrics(metrics)
            ex = x_eval.iloc[:5].copy()
            if hasattr(ex, "select_dtypes"):
                cols = ex.select_dtypes(
                    include=["int", "int32", "int64", "uint32", "uint64"]
                ).columns
                if len(cols):
                    logger.info(f"Converting {len(cols)} integer columns to float64 for signature")
                    ex[cols] = ex[cols].astype("float64")
            signature = mlflow.models.infer_signature(
                ex, model.predict(ex, num_iteration=model.best_iteration)
            )
            model_info = mlflow.lightgbm.log_model(
                model, "model", pip_requirements=pip_requirements, signature=signature
            )
            try:
                mlflow.register_model(
                    f"runs:/{run.info.run_id}/model",
                    f"lightgbm_{datetime.now().strftime('%Y%m%d_%H%M')}",
                )
            except Exception as reg_error:
                logger.warning(f"Model registration failed (this is optional): {reg_error}")
                logger.info("Model logged successfully without registration")
            logger.info(f"Model logged to MLflow: {model_info.model_uri}")
            logger.info(f"Run ID: {run.info.run_id}")
            return run.info.run_id
    except Exception as e:
        logger.error(f"Error logging to MLflow: {str(e)}")
        return None


def main() -> None:
    logger.info("=" * 80)
    logger.info("STANDALONE LIGHTGBM HYPERTUNING & EVALUATION")
    logger.info("=" * 80)

    try:
        logger.info("0. Setting up MLflow experiment tracking...")
        exp_id = MLFlowManager().setup_experiment(experiment_name)
        logger.info(f"MLflow experiment setup completed: {exp_id}")
        logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

        prediction_horizon, number_of_trials = 10, 100
        ml_data = load_and_prepare_data_polars_as_pandas(
            prediction_horizon=prediction_horizon,
            split_date="2025-06-01",
            clean_features=True,
            filter_features=True,
        )
        x_train, x_test = ml_data["x_train"], ml_data["x_test"]
        y_train, y_test = ml_data["y_train"], ml_data["y_test"]
        target_column, train_date_range, test_date_range = (
            ml_data["target_column"],
            ml_data["train_date_range"],
            ml_data["test_date_range"],
        )
        logger.info(f"Data loaded with {x_train.shape[1]} features after selection")
        collect_garbage()

        if "date_int" in x_test.columns:
            threshold = x_test["date_int"].drop_duplicates().max() - 15
            mask = x_test["date_int"] < threshold
            x_test, y_test = x_test[mask], y_test[mask]
            logger.info(f"Removed rows with date_int >= {threshold} (kept {len(x_test)} samples)")
        else:
            logger.warning("'date_int' column not found - skipping date filtering")

        lgb_model = LightGBMModel(
            model_name="lightgbm_standalone_hypertuned", prediction_horizon=prediction_horizon
        )

        # Use the shared persistent Optuna study hook (sqlite-backed).
        study_name = f"{lgb_model.model_name}_ph{prediction_horizon}_profit"
        study = lgb_model.create_optuna_study(
            prediction_horizon=prediction_horizon,
            direction="maximize",
            objective="profit_per_investment",
            objective_slug="profit",
            split_date=str(ml_data.get("split_date")),
            target_column=str(target_column),
            study_name=study_name,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(
                seed=42, multivariate=True, warn_independent_sampling=False
            ),
            n_features=int(x_train.shape[1]),
            n_train=int(len(x_train)),
            n_test=int(len(x_test)),
        )
        study.optimize(
            lgb_model.objective(x_train, y_train, x_test, y_test),
            n_trials=number_of_trials,
            n_jobs=1,
        )
        best_params, best_profit = study.best_params, study.best_value
        logger.info(f"Best Threshold-Optimized Profit per Investment: ${best_profit:.2f}")
        logger.info(f"Best parameters: {best_params}")

        lgb_model.finalize_best_model()
        final_model = lgb_model
        tinfo = getattr(final_model, "best_threshold_info", None)
        has_thresh = bool(tinfo and tinfo.get("optimal_threshold") is not None)

        if has_thresh:
            cur = x_test["close"].values if "close" in x_test.columns else y_test.values * 0.95
            perf = final_model.threshold_evaluator.evaluate_threshold_performance(
                model=final_model,
                x_test=x_test,
                y_test=y_test,
                current_prices_test=cur,
                threshold=getattr(final_model, "optimal_threshold", 0.5),
                confidence_method=getattr(final_model, "confidence_method", "leaf_depth"),
            )
            base_preds = final_model.predict(x_test)
            base_profit = final_model.threshold_evaluator.calculate_profit_score(
                y_test.values, base_preds, cur
            )
            base_ppi = base_profit / len(y_test) if len(y_test) else 0.0
            ratio = (perf["profit_per_investment"] / base_ppi) if base_ppi else 0.0
            final_total_profit, final_profit_per_investment, _final_samples_kept = (
                perf["total_profit"],
                perf["profit_per_investment"],
                perf["samples_evaluated"],
            )
            final_mse, final_mae, final_r2 = perf["mse"], perf["mae"], perf["r2_score"]
            threshold_metrics: Dict[str, Any] = {
                "final_optimal_threshold": tinfo["optimal_threshold"],
                "final_samples_kept_ratio": perf["samples_kept_ratio"],
                "final_investment_success_rate": perf["investment_success_rate"],
                "final_baseline_profit_per_investment": base_ppi,
                "final_improvement_ratio": ratio,
            }
        else:
            cur = (
                x_test["close"].values
                if "close" in x_test.columns
                else np.ones(len(y_test), dtype=float)
            )
            preds = final_model.predict(x_test)
            final_total_profit = final_model.threshold_evaluator.calculate_profit_score(
                y_test.values, preds, cur
            )
            final_profit_per_investment = final_total_profit / len(y_test) if len(y_test) else 0.0
            len(y_test)
            final_mse, final_mae, final_r2 = (
                final_model.custom_metrics.calculate_mse(y_test.values, preds),
                final_model.custom_metrics.calculate_mae(y_test.values, preds),
                final_model.custom_metrics.calculate_r2(y_test.values, preds),
            )
            threshold_metrics = {}

        logger.info(f"Total Profit: ${final_total_profit:.2f}")
        logger.info(f"Profit per Investment: ${final_profit_per_investment:.2f}")
        logger.info(f"Traditional MSE: {final_mse:.4f}")

        feature_importance = final_model.get_feature_importance("gain")
        final_metrics: Dict[str, Any] = {
            "final_total_profit": final_total_profit,
            "final_profit_per_investment": final_profit_per_investment,
            "final_mse": final_mse,
            "final_mae": final_mae,
            "final_r2": final_r2,
            "final_test_samples": len(y_test),
            "hypertuning_best_profit": best_profit,
            "hypertuning_trials_completed": number_of_trials,
            "features_selected": x_train.shape[1],
            **threshold_metrics,
        }
        final_metrics.update(
            {
                f"feature_importance_{i + 1}_{row['feature']}": row["importance"]
                for i, (_, row) in enumerate(feature_importance.head(10).iterrows())
            }
        )

        final_params: Dict[str, Any] = {
            **best_params,
            "prediction_horizon": prediction_horizon,
            "early_stopping_rounds": final_model.early_stopping_rounds,
            "eval_metric": final_model.eval_metric,
            "hypertuning_trials": number_of_trials,
            "hypertuning_direction": "maximize",
            "hypertuning_metric": "profit_score",
            "target_column": target_column,
            "split_date": ml_data["split_date"],
            "feature_count": ml_data["feature_count"],
            "train_samples": ml_data["train_samples"],
            "test_samples": ml_data["test_samples"],
            "threshold_optimization_enabled": has_thresh,
        }
        if has_thresh:
            final_params.update(
                {
                    "threshold_method": final_model.confidence_method,
                    "threshold_optimization_during_hypertuning": True,
                    "optimal_threshold_from_hypertuning": tinfo["optimal_threshold"],
                }
            )

        saved_run_id = final_model.save_model(
            metrics=final_metrics,
            params=final_params,
            x_eval=x_test,
            experiment_name=experiment_name,
        )
        logger.info(f"Model saved to MLflow run: {saved_run_id}")
        logger.info(
            f"Dataset: {ml_data['train_samples'] + ml_data['test_samples']:,} samples, {ml_data['feature_count']} features"
        )
        logger.info(f"Target: {target_column} ({prediction_horizon}-day horizon)")
        logger.info(f"Train period: {train_date_range}")
        logger.info(f"Test period: {test_date_range}")

    except Exception as e:
        logger.error(f"CRITICAL ERROR in standalone LightGBM hypertuning: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
