"""Standalone MLP hypertuning and evaluation."""

import optuna
import pandas as pd

from src.models.time_series.mlp.mlp_predictor import MLPPredictor
from src.utils.core.logger import get_logger
from src.utils.mlops.mlflow_utils import MLFlowManager
from src.models.base_model import load_and_prepare_data_polars_as_pandas
from src.models.time_series.mlp.mlp_architecture import MLPDataUtils

logger = get_logger(__name__)
EXPERIMENT_NAME = "mlp_stock_predictor"


def time_ordered_train_val_split_fast(
    X: pd.DataFrame, y: pd.Series, val_fraction: float = 0.2, time_col: str = "date_int"
):
    """Fast time-ordered train/val split.

    Uses the last `val_fraction` of rows as validation. If `time_col` exists,
    splits on unique time values to avoid mixing timestamps across splits.
    """
    if time_col in X.columns:
        unique_times = pd.Index(X[time_col].dropna().unique()).sort_values()
        cutoff_idx = int(len(unique_times) * (1.0 - val_fraction))
        cutoff_time = unique_times[max(0, min(len(unique_times) - 1, cutoff_idx))]
        train_mask = X[time_col] < cutoff_time
    else:
        cutoff_row = int(len(X) * (1.0 - val_fraction))
        train_mask = pd.Series(True, index=X.index)
        train_mask.iloc[cutoff_row:] = False

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_val, y_val = X.loc[~train_mask], y.loc[~train_mask]
    return X_train, y_train, X_val, y_val


def time_ordered_train_val_split_hardened(
    X: pd.DataFrame, y: pd.Series, val_fraction: float = 0.2, time_col: str = "date_int"
):
    """Hardened time-ordered train/val split with basic validation."""
    if X is None or y is None or len(X) == 0 or len(y) == 0:
        raise ValueError("X and y must be non-empty")
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
    if not (0.05 <= float(val_fraction) <= 0.4):
        raise ValueError("val_fraction must be between 0.05 and 0.4")

    X_aligned = X.copy()
    y_aligned = y.reindex(X_aligned.index)
    if y_aligned.isnull().any():
        raise ValueError("y contains NaNs after aligning to X index")

    X_train, y_train, X_val, y_val = time_ordered_train_val_split_fast(
        X_aligned, y_aligned, val_fraction=val_fraction, time_col=time_col
    )
    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError(
            f"Invalid split sizes: train={len(X_train)}, val={len(X_val)} (val_fraction={val_fraction})"
        )
    return X_train, y_train, X_val, y_val


def main():
    """Standalone MLP hypertuning and evaluation."""
    logger.info("STANDALONE MLP HYPERTUNING & EVALUATION")

    mlflow_manager = MLFlowManager()
    mlflow_manager.setup_experiment(EXPERIMENT_NAME)

    prediction_horizon = 20
    number_of_trials = 50

    data_result = load_and_prepare_data_polars_as_pandas(
        prediction_horizon=prediction_horizon, split_date="2025-06-15", clean_features=True
    )

    x_train, x_test = data_result["x_train"], data_result["x_test"]
    y_train, y_test = data_result["y_train"], data_result["y_test"]
    target_column = data_result["target_column"]

    y_lower, y_upper = y_train.quantile(0.05), y_train.quantile(0.95)
    mask = (y_train >= y_lower) & (y_train <= y_upper)
    x_train, y_train = x_train[mask], y_train[mask]

    x_train = MLPDataUtils.validate_and_clean_data(x_train)
    x_test = MLPDataUtils.validate_and_clean_data(x_test)

    mlp_model = MLPPredictor(model_name="MLP", config={"input_size": len(x_train.columns)})

    selected_features = [col for col in x_train.columns if x_train[col].dtype.kind in "fi"]
    x_train_selected = x_train[selected_features]
    x_test_selected = x_test[selected_features]
    mlp_model.config["input_size"] = len(selected_features)

    val_fraction = 0.2
    x_train_raw, y_train_raw, x_val_raw, y_val = time_ordered_train_val_split_hardened(
        x_train_selected, y_train, val_fraction=val_fraction, time_col="date_int"
    )

    x_train_scaled, scaler = MLPDataUtils.scale_data(x_train_raw, None, True)
    x_val_scaled, _ = MLPDataUtils.scale_data(x_val_raw, scaler, False)

    if "date_int" in x_test_selected.columns:
        date_cutoff = x_test_selected["date_int"].drop_duplicates().max() - 15
        mask = x_test_selected["date_int"] < date_cutoff
        x_test_selected, y_test = x_test_selected[mask], y_test[mask]

    mlp_model.scaler = scaler

    objective_function = mlp_model.objective(
        x_train_scaled, y_train_raw, x_val_raw, x_val_scaled, y_val, fitted_scaler=scaler
    )
    study = mlp_model.create_optuna_study(
        prediction_horizon=prediction_horizon,
        direction="maximize",
        objective="profit_per_investment",
        objective_slug="profit",
        split_date=data_result.get("split_date"),
        target_column=target_column,
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True, n_startup_trials=15),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2),
        load_if_exists=True,
        n_features=len(selected_features),
        n_train=len(x_train_scaled),
        n_test=len(x_val_scaled),
    )
    study.optimize(objective_function, n_trials=number_of_trials, n_jobs=1)

    logger.info(f"Optimization complete: Best profit ${study.best_value:.2f}")

    mlp_model.finalize_best_model()
    final_current_prices = x_test_selected["close"].values
    optimal_threshold = getattr(mlp_model, "optimal_threshold", 0.5)
    confidence_method = getattr(mlp_model, "confidence_method", "variance")

    threshold_performance = mlp_model.threshold_evaluator.evaluate_threshold_performance(
        model=mlp_model,
        x_test=x_test_selected,
        y_test=y_test,
        current_prices_test=final_current_prices,
        threshold=optimal_threshold,
        confidence_method=confidence_method,
    )

    baseline_predictions = mlp_model.predict(x_test_selected)
    baseline_profit = mlp_model.threshold_evaluator.calculate_profit_score(
        y_test.values, baseline_predictions, final_current_prices
    )
    baseline_profit_per_inv = baseline_profit / len(y_test)

    logger.info(f"Baseline profit/inv: ${baseline_profit_per_inv:.2f}")
    logger.info(
        f"Threshold-optimized profit/inv: ${threshold_performance['profit_per_investment']:.2f}"
    )

    final_metrics = {
        "final_total_profit": threshold_performance["total_profit"],
        "final_profit_per_investment": threshold_performance["profit_per_investment"],
        "final_mse": threshold_performance["mse"],
        "final_mae": threshold_performance["mae"],
        "final_r2": threshold_performance["r2_score"],
        "final_test_samples": len(y_test),
        "hypertuning_best_profit": study.best_value,
        "hypertuning_trials_completed": number_of_trials,
        "features_selected": len(selected_features),
        "final_optimal_threshold": mlp_model.best_threshold_info["optimal_threshold"],
        "final_samples_kept_ratio": threshold_performance["samples_kept_ratio"],
        "final_investment_success_rate": threshold_performance["investment_success_rate"],
        "final_baseline_profit_per_investment": baseline_profit_per_inv,
        "final_improvement_ratio": threshold_performance["profit_per_investment"]
        / baseline_profit_per_inv
        if baseline_profit_per_inv
        else 0,
    }

    final_params = {
        **study.best_params,
        "prediction_horizon": prediction_horizon,
        "hypertuning_trials": number_of_trials,
        "target_column": target_column,
        "split_date": data_result["split_date"],
        "feature_count": data_result["feature_count"],
        "train_samples": data_result["train_samples"],
        "test_samples": data_result["test_samples"],
        "val_fraction": val_fraction,
        "threshold_method": mlp_model.confidence_method,
        "threshold_optimization_during_hypertuning": True,
        "optimal_threshold_from_hypertuning": mlp_model.best_threshold_info["optimal_threshold"],
    }

    saved_run_id = mlp_model.save_model(
        metrics=final_metrics,
        params=final_params,
        x_eval=x_test_selected,
        experiment_name=EXPERIMENT_NAME,
        scaler=scaler,
    )

    logger.info(f"STANDALONE MLP HYPERTUNING COMPLETED - Run ID: {saved_run_id}")
    logger.info(
        f"Total Profit: ${final_metrics['final_total_profit']:.2f}, Profit/inv: ${final_metrics['final_profit_per_investment']:.2f}"
    )


if __name__ == "__main__":
    main()
