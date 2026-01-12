"""XGBoost Model Implementation for stock prediction with hyperparameter optimization."""

import traceback
from typing import Dict, Any, Optional, Tuple, Callable
import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from datetime import datetime
from src.models.base_model import BaseModel, load_and_prepare_data_polars_as_pandas
from src.models.evaluation import ThresholdEvaluator, CustomMetrics
from src.utils.core.logger import get_logger
from src.utils.mlops.mlflow_utils import MLFlowManager

logger = get_logger(__name__)


class _OptunaPruningCallback(xgb.callback.TrainingCallback):
    """Report intermediate validation metrics to Optuna and prune unpromising trials."""

    def __init__(self, trial: "optuna.Trial", metric_name: str, dataset_name: str = "test"):
        self._trial = trial
        self._metric_name = metric_name
        self._dataset_name = dataset_name

    def after_iteration(self, model: xgb.Booster, epoch: int, evals_log: dict) -> bool:
        # evals_log example: {"train": {"rmse": [..]}, "test": {"rmse": [..]}}
        try:
            dataset_log = (
                evals_log.get(self._dataset_name, {}) if isinstance(evals_log, dict) else {}
            )
            metric_vals = dataset_log.get(self._metric_name)
            if metric_vals:
                value = float(metric_vals[-1])
                self._trial.report(value, step=int(epoch))
                if self._trial.should_prune():
                    raise optuna.TrialPruned()
        except optuna.TrialPruned:
            raise
        except Exception:
            # Best-effort pruning callback; never fail training due to logging issues.
            pass
        return False


class XGBoostModel(BaseModel):
    @staticmethod
    def _normalize_xgb_eval_metric(metric: Any) -> str:
        """Normalize common metric aliases to XGBoost-supported names.

        XGBoost does not support 'mse' as an eval_metric (it supports 'rmse').
        Keep behavior predictable by mapping common aliases.
        """
        if metric is None:
            return "rmse"
        try:
            m = str(metric).strip().lower()
        except Exception:
            return "rmse"

        aliases = {"mse": "rmse", "l2": "rmse", "rmse": "rmse", "mae": "mae", "l1": "mae"}
        return aliases.get(m, m)

    def __init__(
        self,
        model_name: str = "xgboost_stock_predictor",
        config: Optional[Dict[str, Any]] = None,
        prediction_horizon: int = 10,
        threshold_evaluator: Optional[ThresholdEvaluator] = None,
    ):
        config = {**(config or {}), "prediction_horizon": prediction_horizon}
        super().__init__(model_name, config, threshold_evaluator=threshold_evaluator)
        self.prediction_horizon, self.model, self.feature_names = prediction_horizon, None, None
        self.early_stopping_rounds = self.config.get("early_stopping_rounds", 50)
        raw_metric = self.config.get("eval_metric", "rmse")
        self.eval_metric = self._normalize_xgb_eval_metric(raw_metric)
        if str(raw_metric).strip().lower() != self.eval_metric:
            logger.warning(
                f"Normalized eval_metric '{raw_metric}' to '{self.eval_metric}' for XGBoost compatibility"
            )
        # Default confidence method: use a simple margin-based baseline.
        self.base_threshold, self.default_confidence_method = 0.5, "margin"
        self.custom_metrics = CustomMetrics()

    def _create_model(
        self, params: Optional[Dict[str, Any]] = None, model_name: Optional[str] = None, **kwargs
    ) -> "XGBoostModel":
        params, model_name = params or {}, model_name or f"{self.model_name}_configured"
        # Merge params with investment amount for threshold evaluation
        config = {**params, "investment_amount": self.threshold_evaluator.investment_amount}
        new_model = XGBoostModel(
            model_name=model_name,
            config=config,
            prediction_horizon=self.prediction_horizon,
            threshold_evaluator=self.threshold_evaluator,
        )
        # Copy current model's training settings to new model
        new_model.early_stopping_rounds, new_model.eval_metric = (
            self.early_stopping_rounds,
            self.eval_metric,
        )
        logger.info(f"Created model '{model_name}' with params: {params}")
        return new_model

    def _prepare_data(
        self, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series
    ) -> Tuple[xgb.DMatrix, xgb.DMatrix]:
        self.feature_names = list(x_train.columns)
        # Convert pandas DataFrames to XGBoost's optimized DMatrix format
        return xgb.DMatrix(x_train, label=y_train, feature_names=self.feature_names), xgb.DMatrix(
            x_test, label=y_test, feature_names=self.feature_names
        )

    def _auto_split(self, x_train, y_train, x_test, y_test, validation_split):
        """Auto-split training data if test data not provided."""
        if x_test is None or y_test is None:
            split_frac = validation_split if validation_split and 0 < validation_split < 1 else 0.2
            # Calculate split index from end of data (ensures at least 1 sample in validation)
            split_idx = max(1, len(x_train) - int(len(x_train) * split_frac))
            # Split tail portion for validation, rest for training
            x_test, y_test = x_train.iloc[split_idx:], y_train.iloc[split_idx:]
            x_train, y_train = x_train.iloc[:split_idx], y_train.iloc[:split_idx]
        return x_train, y_train, x_test, y_test

    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        params: Optional[Dict[str, Any]] = None,
        validation_split: Optional[float] = None,
        trial: Optional["optuna.Trial"] = None,
        prune_metric: Optional[str] = None,
    ) -> "XGBoostModel":
        # Auto-create validation split if test data not provided
        x_train, y_train, x_test, y_test = self._auto_split(
            x_train, y_train, x_test, y_test, validation_split
        )
        # Convert to XGBoost's DMatrix format
        dtrain, dtest = self._prepare_data(x_train, y_train, x_test, y_test)
        # Use provided params or get defaults from config
        params = params or self.hyperparameter_config.get_default_params()
        # Extract n_estimators separately (not a direct XGBoost param)
        n_estimators = params.pop("n_estimators", 1000)
        # Keep training deterministic/repeatable.
        seed = int(params.get("seed", params.get("random_state", 42)))

        # GPU support (recommended in XGBoost 2.x/3.x is tree_method=hist + device=cuda).
        use_gpu = bool(self.config.get("use_gpu", False))
        if use_gpu:
            params.setdefault("tree_method", "hist")
            params.setdefault("device", "cuda")
        else:
            params.setdefault("tree_method", "hist")

        # Quantile regression objective for conservative (upper-bound) return prediction.
        eval_metric = self._normalize_xgb_eval_metric(self.eval_metric)
        params.update(
            {
                "objective": "reg:quantileerror",
                "quantile_alpha": float(params.get("quantile_alpha", 0.8)),
                "eval_metric": eval_metric,
                "seed": seed,
                "verbosity": 0,
            }
        )

        callbacks = []
        if trial is not None:
            callbacks.append(
                _OptunaPruningCallback(
                    trial=trial,
                    metric_name=str(prune_metric or self.eval_metric),
                    dataset_name="test",
                )
            )

        # Train model with early stopping on validation set
        self.model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=n_estimators,
            evals=[(dtrain, "train"), (dtest, "test")],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False,
            callbacks=callbacks,
        )
        logger.info(
            f"Training done. Best iter: {self.model.best_iteration}, score: {self.model.best_score:.4f}"
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.model:
            raise RuntimeError("Model must be trained before prediction")
        if self.feature_names and list(X.columns) != self.feature_names:
            logger.warning("Feature mismatch")
        predictions = self.model.predict(xgb.DMatrix(X, feature_names=list(X.columns)))
        logger.info(f"Generated {len(predictions)} predictions")
        return predictions

    def get_feature_importance(self, importance_type: str = "weight") -> pd.DataFrame:
        if not self.model:
            raise RuntimeError("Model must be trained")
        # Convert XGBoost feature scores dict to sorted DataFrame (highest importance first)
        return pd.DataFrame(
            [
                {"feature": f, "importance": imp}
                for f, imp in self.model.get_score(importance_type=importance_type).items()
            ]
        ).sort_values("importance", ascending=False)

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        if self.model:
            info.update(
                {
                    "n_features": self.model.num_features(),
                    "best_iteration": self.model.best_iteration,
                    "best_score": self.model.best_score,
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
        if not self.model:
            raise RuntimeError("No model to save")

        experiment_name = experiment_name or "xgboost_stock_predictor"
        try:
            mlflow.end_run()
        except Exception:
            pass

        try:
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(
                run_name=f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M')}"
            ) as run:
                # Filter out device-specific params before logging
                mlflow.log_params(
                    {
                        k: v
                        for k, v in params.items()
                        if k
                        not in [
                            "device",
                            "objective",
                            "verbosity",
                            "seed",
                            "nthread",
                            "tree_method",
                        ]
                    }
                )
                mlflow.log_metrics(metrics)

                # Create input example from first 5 rows for model signature
                input_ex = x_eval.iloc[:5].copy()
                # Convert int columns to float64 to avoid MLflow signature issues
                if hasattr(input_ex, "dtypes"):
                    for col in input_ex.select_dtypes(include=["int"]).columns:
                        input_ex[col] = input_ex[col].astype("float64")

                # Infer model signature (input/output schema) from example
                signature = mlflow.models.infer_signature(
                    input_ex, self.model.predict(xgb.DMatrix(input_ex))
                )
                mlflow.xgboost.log_model(
                    self.model,
                    "model",
                    pip_requirements=[
                        "xgboost>=1.7.0",
                        "pandas>=1.5.0",
                        "numpy>=1.21.0",
                        "scikit-learn>=1.0.0",
                        "mlflow>=2.0.0",
                    ],
                    registered_model_name=f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M')}",
                    signature=signature,
                )
                logger.info(f"Model logged. Run ID: {run.info.run_id}")
                return run.info.run_id
        except Exception as e:
            logger.error(f"MLflow error: {e}")
            return None

    def load_model(self, run_id: str) -> None:
        try:
            artifacts = mlflow.tracking.MlflowClient().list_artifacts(run_id)
            # Find the model directory (default to "model" if not found)
            path = next(
                (a.path for a in artifacts if a.is_dir and "model" in a.path.lower()), "model"
            )
            self.model = mlflow.xgboost.load_model(f"runs:/{run_id}/{path}")
            logger.info(f"Model loaded from run: {run_id}")
        except Exception as e:
            logger.error(f"Load error: {e}")
            raise

    def _normalize_confidence(self, scores: np.ndarray) -> np.ndarray:
        """Normalize confidence scores to [0,1]."""
        min_s, max_s = scores.min(), scores.max()
        # Min-max normalization, or return 0.5 (neutral) if all scores are equal
        return (scores - min_s) / (max_s - min_s) if max_s > min_s else np.ones_like(scores) * 0.5

    def get_prediction_confidence(self, X: pd.DataFrame, method: str = "leaf_depth") -> np.ndarray:
        if not self.model:
            raise RuntimeError("Model must be trained")
        dtest = xgb.DMatrix(X, feature_names=list(X.columns))

        method = (method or "margin").strip().lower()

        if method in {"leaf_depth", "margin"}:
            if method == "leaf_depth":
                logger.warning("Confidence method 'leaf_depth' is deprecated; using 'margin'.")
            preds = self.model.predict(dtest)
            scores = np.abs(preds)
        elif method == "variance":
            # Confidence based on prediction stability across early iterations (low variance = high confidence)
            preds = np.array(
                [
                    self.model.predict(dtest, ntree_limit=i + 1)
                    for i in range(min(10, self.model.best_iteration))
                ]
            )
            scores = 1.0 / (1.0 + np.std(preds, axis=0))
        elif method == "simple":
            # Confidence based on distance from median (close to median = high confidence)
            preds = self.model.predict(dtest)
            scores = 1.0 - np.abs(preds - np.median(preds)) / (
                np.percentile(preds, 75) - np.percentile(preds, 25) + 1e-6
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # Normalize all scores to [0,1] range
        scores = self._normalize_confidence(scores)
        logger.info(
            f"Confidence [{method}] - Range: [{scores.min():.4f}, {scores.max():.4f}], Mean: {scores.mean():.4f}"
        )
        return scores

    def _evaluate_trial(
        self,
        trial_model,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        objective_col: str,
        trial_number: Optional[int] = None,
    ):
        """Evaluate trial model with threshold optimization using model wrapper.

        Uses the model's `optimize_prediction_threshold` convenience method so
        behavior matches other models (LightGBM, etc.). Returns (score, results).
        """
        tn = f" {trial_number}" if trial_number is not None else ""
        logger.info(f"Running threshold optimization for trial{tn}")
        try:
            if "close" not in x_test.columns:
                logger.warning(
                    "'close' column not found in x_test; skipping threshold optimization"
                )
                return 0.0, {}

            current_prices = x_test["close"].values
            res = trial_model.optimize_prediction_threshold(
                x_test=x_test,
                y_test=y_test,
                current_prices_test=current_prices,
                confidence_method="margin",
                threshold_range=(0.01, 0.99),
                n_thresholds=90,
            )
            best = res.get("best_result", {})
            score = float(best.get("test_profit_per_investment", 0.0))
            return score, res
        except Exception as e:
            logger.warning(f"Threshold optimization failed for trial{tn}: {e}")
            return 0.0, {}

    def objective(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        objective_column: str = "test_profit_per_investment",
    ) -> Callable:
        (
            self.best_investment_success_rate,
            self.best_trial_model,
            self.best_trial_params,
            self.best_threshold_info,
        ) = -np.inf, None, None, None

        def objective(trial):
            # Hyperparameter search aimed at selection quality / profit per investment.
            # Includes tree-shape knobs (grow_policy/max_leaves), histogram binning, and
            # finer-grained column subsampling.
            booster = trial.suggest_categorical("booster", ["gbtree", "dart"])
            grow_policy = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

            params = {
                "booster": booster,
                "grow_policy": grow_policy,
                "max_depth": trial.suggest_int("max_depth", 2, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 200, 3000, step=200),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
                "max_bin": trial.suggest_int("max_bin", 128, 512, step=64),
                "gamma": trial.suggest_float("gamma", 0.0, 10.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 50.0, log=True),
                "max_delta_step": trial.suggest_float("max_delta_step", 0.0, 10.0),
                "quantile_alpha": trial.suggest_float("quantile_alpha", 0.6, 0.95),
                "seed": 42,
            }

            if grow_policy == "lossguide":
                params["max_leaves"] = trial.suggest_int("max_leaves", 16, 512, log=True)

            if booster == "dart":
                # DART can help reduce overfitting; keep ranges conservative.
                params.update(
                    {
                        "sample_type": trial.suggest_categorical(
                            "sample_type", ["uniform", "weighted"]
                        ),
                        "normalize_type": trial.suggest_categorical(
                            "normalize_type", ["tree", "forest"]
                        ),
                        "rate_drop": trial.suggest_float("rate_drop", 0.0, 0.3),
                        "skip_drop": trial.suggest_float("skip_drop", 0.0, 0.3),
                    }
                )

            self._optuna_log_dataset_sizes(trial, x_train=x_train, x_test=x_test)

            trial_model = self._create_model(params=params, model_name=f"trial_{trial.number}")
            trial_model.fit(
                x_train,
                y_train,
                x_test,
                y_test,
                params=params,
                trial=trial,
                prune_metric=self.eval_metric,
            )
            profit_per_inv, threshold_result = self._evaluate_trial(
                trial_model, x_test, y_test, objective_column, trial_number=trial.number
            )

            self._optuna_log_trial_core_metrics(
                trial,
                profit_per_investment=profit_per_inv,
                best_iteration=getattr(getattr(trial_model, "model", None), "best_iteration", None),
                best_score=getattr(getattr(trial_model, "model", None), "best_score", None),
            )

            # Persist threshold optimization outputs to trial information for dashboards.
            if isinstance(threshold_result, dict):
                best = threshold_result.get("best_result") or {}
                self._optuna_log_threshold_metrics(
                    trial,
                    optimal_threshold=threshold_result.get("optimal_threshold"),
                    samples_kept_ratio=(
                        best.get("test_samples_ratio") if isinstance(best, dict) else None
                    ),
                    investment_success_rate=(
                        best.get("investment_success_rate") if isinstance(best, dict) else None
                    ),
                    custom_accuracy=(
                        best.get("test_custom_accuracy") if isinstance(best, dict) else None
                    ),
                    total_profit=(best.get("test_profit") if isinstance(best, dict) else None),
                    profitable_investments=(
                        best.get("profitable_investments") if isinstance(best, dict) else None
                    ),
                )

            # Track best performing model across all trials
            if profit_per_inv > self.best_investment_success_rate:
                (
                    self.best_investment_success_rate,
                    self.best_trial_model,
                    self.best_trial_params,
                    self.best_threshold_info,
                ) = profit_per_inv, trial_model.model, params, threshold_result
                logger.info(f"New best trial {trial.number}: ${profit_per_inv:.2f}")

            return profit_per_inv

        return objective

    def get_best_trial_info(self) -> Dict[str, Any]:
        if not hasattr(self, "best_investment_success_rate"):
            raise RuntimeError("No optimization run")
        info = {
            "best_investment_success_rate": self.best_investment_success_rate,
            "best_trial_params": self.best_trial_params,
            "has_best_model": self.best_trial_model is not None,
            "model_updated": self.model is not None,
        }
        if hasattr(self, "best_threshold_info") and self.best_threshold_info:
            info.update(
                {
                    "optimal_threshold": self.best_threshold_info.get("optimal_threshold"),
                    "threshold_metrics": self.best_threshold_info.get("metrics", {}),
                    "filtered_predictions_count": self.best_threshold_info.get(
                        "filtered_predictions_count"
                    ),
                }
            )
        return info

    def finalize_best_model(self) -> None:
        if hasattr(self, "best_trial_model") and self.best_trial_model:
            self.model = self.best_trial_model
            if hasattr(self, "best_threshold_info") and self.best_threshold_info:
                self.optimal_threshold = self.best_threshold_info.get(
                    "optimal_threshold", self.base_threshold
                )
                logger.info(f"Best model finalized with threshold: {self.optimal_threshold:.4f}")
        else:
            logger.warning("No best trial to finalize")


def main():
    logger.info("=" * 80 + "\nSTANDALONE XGBOOST HYPERTUNING\n" + "=" * 80)
    try:
        mlflow_manager = MLFlowManager()
        exp_name = "xgboost_stock_predictor"
        exp_id = mlflow_manager.setup_experiment(exp_name)
        logger.info(f"MLflow: {exp_id}, URI: {mlflow.get_tracking_uri()}")

        # Setup: 10-day prediction horizon, 20 optimization trials, maximize profit per investment
        pred_horizon, n_trials, obj_col = 10, 20, "test_profit_per_investment"
        data = load_and_prepare_data_polars_as_pandas(
            prediction_horizon=pred_horizon, split_date="2025-06-15", clean_features=True
        )
        x_train, x_test = data["x_train"], data["x_test"]
        y_train, y_test = data["y_train"], data["y_test"]
        logger.info(f"Data: {data['target_column']}")

        # Run Optuna hyperparameter optimization to find best XGBoost config.
        model = XGBoostModel(
            model_name="xgboost_hypertuned",
            prediction_horizon=pred_horizon,
            config={"use_gpu": True},
        )
        study_name = f"{model.model_name}_ph{pred_horizon}_profit"

        # Use the shared persistent Optuna study hook (sqlite-backed).
        study = model.create_optuna_study(
            prediction_horizon=pred_horizon,
            direction="maximize",
            objective="profit_per_investment",
            objective_slug="profit",
            split_date=str(data.get("split_date")),
            target_column=str(data.get("target_column")),
            study_name=study_name,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=25),
            n_features=int(x_train.shape[1]),
            n_train=int(len(x_train)),
            n_test=int(len(x_test)),
        )
        study.optimize(
            model.objective(x_train, y_train, x_test, y_test, obj_col), n_trials=n_trials, n_jobs=1
        )
        logger.info(f"Best profit: ${study.best_value:.2f}, Params: {study.best_params}")

        # Finalize by setting the best model from all trials as the active model
        model.finalize_best_model()
        y_pred = model.predict(x_test)

        # Evaluate with threshold optimization to filter low-confidence predictions
        has_thresh = hasattr(model, "best_threshold_info") and model.best_threshold_info
        if has_thresh:
            # Re-run threshold optimization on final model
            thresh_res = model.threshold_evaluator.find_optimal_threshold_for_investment(
                y_test, y_pred, investment_amount=model.threshold_evaluator.investment_amount
            )
            metrics = thresh_res["metrics"]
            profit, profit_per_inv = (
                metrics.get("test_total_profit", 0),
                metrics.get("test_profit_per_investment", 0),
            )
            thresh_metrics = {
                "optimal_threshold": model.best_threshold_info["optimal_threshold"],
                "filtered_predictions": thresh_res["filtered_predictions_count"],
            }
        else:
            profit, profit_per_inv, thresh_metrics = 0, 0, {}

        # Calculate traditional regression metrics for reference
        mse, mae = np.mean((y_test - y_pred) ** 2), np.mean(np.abs(y_test - y_pred))
        r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
        logger.info(
            f"Results - Profit: ${profit:.2f}, Profit/Inv: ${profit_per_inv:.2f}, MSE: {mse:.4f}, R2: {r2:.4f}"
        )

        # Extract and log top contributing features
        feat_imp = model.get_feature_importance("gain")
        logger.info(
            "Top 30 features:\n"
            + "\n".join(
                [
                    f"  {row['feature']}: {row['importance']:.2f}"
                    for _, row in feat_imp.head(30).iterrows()
                ]
            )
        )

        final_metrics = {
            "final_total_profit": profit,
            "final_profit_per_investment": profit_per_inv,
            "final_mse": mse,
            "final_mae": mae,
            "final_r2": r2,
            "final_test_samples": len(y_test),
            "hypertuning_best_profit": study.best_value,
            "hypertuning_trials_completed": n_trials,
            **thresh_metrics,
        }
        # Consolidate all parameters for MLflow logging (hyperparams + data info + config)
        final_params = {
            **study.best_params,
            "prediction_horizon": pred_horizon,
            "early_stopping_rounds": model.early_stopping_rounds,
            "eval_metric": model.eval_metric,
            "hypertuning_trials": n_trials,
            "target_column": data["target_column"],
            "split_date": data["split_date"],
            "feature_count": data["feature_count"],
            "train_samples": data["train_samples"],
            "test_samples": data["test_samples"],
            "threshold_optimization_enabled": has_thresh,
        }

        run_id = model.save_model(
            metrics=final_metrics, params=final_params, x_eval=x_test, experiment_name=exp_name
        )
        logger.info("=" * 80 + f"\nCOMPLETED! Profit: ${profit:.2f}, Run: {run_id}\n" + "=" * 80)

    except Exception as e:
        logger.error(f"ERROR: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
