"""Random Forest Model Implementation (compact).

Counting heuristics for savings: exclude docstring-only blocks, blank lines, and '#' comment-only lines.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import ast

import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src.models.base_model import BaseModel, load_and_prepare_data_polars_as_pandas
from src.models.evaluation.threshold_evaluator import ThresholdEvaluator
from src.utils.core.logger import get_logger

logger = get_logger(__name__)


def _as_dict(s: str) -> Dict[str, Any]:
    try:
        v = ast.literal_eval(s)
        return v if isinstance(v, dict) else {}
    except (ValueError, SyntaxError):
        return {}


def _as_list(s: str) -> list:
    try:
        v = ast.literal_eval(s)
        return v if isinstance(v, list) else []
    except (ValueError, SyntaxError):
        return []


class RandomForestModel(BaseModel):
    def __init__(
        self,
        model_name: str = "random_forest_stock_predictor",
        config: Optional[Dict[str, Any]] = None,
        prediction_horizon: int = 10,
        threshold_evaluator: Optional[ThresholdEvaluator] = None,
    ):
        cfg = dict(config or {})
        cfg["prediction_horizon"] = prediction_horizon
        super().__init__(model_name, cfg, threshold_evaluator=threshold_evaluator)
        self.prediction_horizon = int(self.config.get("prediction_horizon", prediction_horizon))

    def _create_model(self, **kwargs):
        p = {**self.config, **kwargs}
        p.pop("prediction_horizon", None)
        p.setdefault("min_samples_leaf", 1)
        p.setdefault("max_features", "sqrt")
        p.setdefault("random_state", 42)
        return RandomForestRegressor(**p)

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        x_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs,
    ):
        self.model = self._create_model(**kwargs)
        self.model.fit(x, y)
        self.is_trained = True
        self.feature_names = list(x.columns)
        self.feature_importance = getattr(self.model, "feature_importances_", None)
        self.training_history = {}
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("RandomForestModel (regression) does not support predict_proba.")

    def get_prediction_confidence(self, X: pd.DataFrame, method: str = "variance") -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before calculating confidence")
        if self.feature_names is not None:
            X = X[self.feature_names]
        if method != "variance":
            raise ValueError(f"Unknown confidence method: {method}")

        preds = np.array([t.predict(X) for t in self.model.estimators_])
        conf = 1.0 / (1.0 + np.var(preds, axis=0))
        lo, hi = float(conf.min()), float(conf.max())
        return (conf - lo) / (hi - lo) if hi > lo else np.full_like(conf, 0.5)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        current_prices: Optional[np.ndarray] = None,
        confidence_method: str = "variance",
    ) -> Dict[str, float]:
        return super().evaluate(
            X, y, current_prices=current_prices, confidence_method=confidence_method
        )

    def save_model(self, experiment_name: str = None) -> str:
        if experiment_name:
            self.experiment_name = experiment_name
        if self.run_id is None:
            active = mlflow.active_run()
            if active is not None:
                self.run_id = active.info.run_id
        return super().save_model(self.experiment_name)

    def load_model(self, run_id: str) -> "RandomForestModel":
        model_uri = f"runs:/{run_id}/model"
        self.model = self.mlflow_integration.load_sklearn_model(model_uri)
        params = self.mlflow_integration.get_run(run_id).data.params

        self.model_name = params.get("model_name", "Unknown")
        self.config = _as_dict(params.get("config", "{}"))
        features_raw = params.get("feature_names", "[]")
        self.feature_names = [] if features_raw == "[]" else _as_list(features_raw)
        self.is_trained = params.get("is_trained", "True") == "True"
        self.run_id = run_id
        return self

    def get_best_trial_info(self) -> Dict[str, Any]:
        if not hasattr(self, "best_score"):
            return {"message": "No hypertuning has been run yet"}
        return {
            "best_score": getattr(self, "best_score", None),
            "best_trial_params": getattr(self, "best_trial_params", None),
            "has_best_model": bool(getattr(self, "best_trial_model", None)),
            "model_updated": self.model is not None,
            "threshold_optimization": getattr(self, "best_threshold_info", None),
        }

    def finalize_best_model(self) -> None:
        best = getattr(self, "best_trial_model", None)
        if not best:
            logger.warning("No best model found to finalize")
            return
        self.model, self.feature_names = best.model, best.feature_names
        thr = getattr(self, "best_threshold_info", None)
        if thr and thr.get("optimal_threshold") is not None:
            self.optimal_threshold = thr["optimal_threshold"]
            self.confidence_method = getattr(self, "confidence_method", "variance")

    def select_features(
        self, X: pd.DataFrame, y: pd.Series, n_features_to_select: int = 50
    ) -> list:
        rf = RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1, min_samples_leaf=1, max_features="sqrt"
        )
        rf.fit(X, y)
        order = np.argsort(rf.feature_importances_)[::-1]
        selected = list(X.columns[order][:n_features_to_select])

        for required in ("close", "ticker_id"):
            if required in X.columns and required not in selected and selected:
                selected[-1] = required
        return selected

    @staticmethod
    def load_and_prepare_data(
        prediction_horizon: int = 10, split_date: str = None, clean_features: bool = True
    ):
        """Load and prepare data for RandomForestModel.

        Delegates to BaseModel's shared Polars pipeline helper.
        """

        return load_and_prepare_data_polars_as_pandas(
            prediction_horizon=prediction_horizon,
            split_date=split_date or "2025-05-01",
            clean_features=clean_features,
        )

    def objective(
        self, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series
    ) -> callable:
        self.best_score = -np.inf
        self.best_trial_model = None
        self.best_trial_params = None
        self.best_threshold_info = None

        def obj(trial: Any):
            self._optuna_log_dataset_sizes(trial, x_train=x_train, x_test=x_test)
            p = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_features": trial.suggest_categorical(
                    "max_features", ["auto", "sqrt", "log2", None]
                ),
                "random_state": 42,
                "n_jobs": -1,
            }
            try:
                m = RandomForestModel(
                    model_name=f"random_forest_trial_{trial.number}",
                    config=p,
                    prediction_horizon=self.prediction_horizon,
                    threshold_evaluator=self.threshold_evaluator,
                ).fit(x_train, y_train)

                prices = x_test["close"].values if "close" in x_test.columns else None
                if prices is None:
                    metric, thr = m.model.score(x_test, y_test), None
                else:
                    thr = m.threshold_evaluator.optimize_prediction_threshold(
                        model=m,
                        x_test=x_test,
                        y_test=y_test,
                        current_prices_test=prices,
                        confidence_method="variance",
                    )
                    metric = (
                        thr["best_result"].get("test_profit_per_investment", 0.0)
                        if thr.get("status") == "success"
                        else m.model.score(x_test, y_test)
                    )

                self._optuna_log_trial_core_metrics(trial, profit_per_investment=metric)
                if isinstance(thr, dict) and thr.get("status") == "success":
                    best = thr.get("best_result", {}) or {}
                    self._optuna_log_threshold_metrics(
                        trial,
                        optimal_threshold=thr.get("optimal_threshold"),
                        samples_kept_ratio=best.get("test_samples_ratio"),
                        investment_success_rate=best.get("investment_success_rate"),
                        custom_accuracy=best.get("test_custom_accuracy"),
                        total_profit=best.get("test_profit"),
                        profitable_investments=best.get("profitable_investments"),
                    )

                if metric > self.best_score:
                    self.best_score, self.best_trial_model = metric, m
                    self.best_trial_params, self.best_threshold_info = p.copy(), thr
                    self.model, self.feature_names = m.model, m.feature_names
                return metric
            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {e}")
                return -1e6

        return obj


def main():
    sep = "=" * 80
    logger.info(sep)
    logger.info("STANDALONE RANDOM FOREST HYPERTUNING & EVALUATION")
    logger.info(sep)

    data = RandomForestModel.load_and_prepare_data(prediction_horizon=10, split_date="2025-06-15")
    x_train, x_test = data["x_train"], data["x_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    fs = RandomForestModel(model_name="random_forest_feature_selector", prediction_horizon=10)
    selected = fs.select_features(x_train, y_train, 60)
    x_train_s, x_test_s = x_train[selected], x_test[selected]

    model = RandomForestModel(
        model_name="random_forest_standalone_hypertuned", prediction_horizon=10
    )
    study = model.create_optuna_study(
        direction="maximize",
        objective="profit_per_investment",
        objective_slug="profit",
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True,
    )
    study.optimize(model.objective(x_train_s, y_train, x_test_s, y_test), n_trials=50, n_jobs=1)

    model.finalize_best_model()
    prices = x_test_s["close"].values if "close" in x_test_s.columns else None
    _ = model.evaluate(x_test_s, y_test, current_prices=prices)
    _ = model.save_model(experiment_name="random_forest_stock_predictor_experiment")
    logger.info("STANDALONE RANDOM FOREST HYPERTUNING COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
