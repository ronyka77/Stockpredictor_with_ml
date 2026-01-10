"""Standalone MLP hypertuning and evaluation."""

import optuna
from src.models.time_series.mlp.mlp_predictor import MLPPredictor
from src.utils.core.logger import get_logger
from src.utils.mlops.mlflow_utils import MLFlowManager
from src.data_utils.polars_ml_pipeline import (
    prepare_ml_data_for_training_with_cleaning_polars,
    to_pandas,
)
from src.models.time_series.mlp.mlp_architecture import MLPDataUtils

logger = get_logger(__name__)
EXPERIMENT_NAME = "mlp_stock_predictor"


def main():
    """Standalone MLP hypertuning and evaluation."""
    logger.info("STANDALONE MLP HYPERTUNING & EVALUATION")

    mlflow_manager = MLFlowManager()
    mlflow_manager.setup_experiment(EXPERIMENT_NAME)

    prediction_horizon = 10
    number_of_trials = 200

    data_result = prepare_ml_data_for_training_with_cleaning_polars(
        prediction_horizon=prediction_horizon, split_date="2025-06-15", clean_features=True
    )

    x_train, x_test = to_pandas(data_result["x_train"]), to_pandas(data_result["x_test"])
    y_train, y_test = to_pandas(data_result["y_train"]), to_pandas(data_result["y_test"])
    target_column = data_result["target_column"]

    y_lower, y_upper = y_train.quantile(0.05), y_train.quantile(0.95)
    mask = (y_train >= y_lower) & (y_train <= y_upper)
    x_train, y_train = x_train[mask], y_train[mask]

    x_train = MLPDataUtils.validate_and_clean_data(x_train)
    x_test = MLPDataUtils.validate_and_clean_data(x_test)

    mlp_model = MLPPredictor(
        model_name="mlp_complete_pipeline", config={"input_size": len(x_train.columns)}
    )

    selected_features = [col for col in x_train.columns if x_train[col].dtype.kind in "fi"]
    x_train_selected = x_train[selected_features]
    x_test_selected = x_test[selected_features]
    mlp_model.config["input_size"] = len(selected_features)

    x_train_selected, scaler = MLPDataUtils.scale_data(x_train_selected, None, True)

    if "date_int" in x_test_selected.columns:
        threshold = x_test_selected["date_int"].drop_duplicates().max() - 15
        mask = x_test_selected["date_int"] < threshold
        x_test_selected, y_test = x_test_selected[mask], y_test[mask]

    x_test_scaled, _ = MLPDataUtils.scale_data(x_test_selected, scaler, False)
    mlp_model.scaler = scaler

    objective_function = mlp_model.objective(
        x_train_selected, y_train, x_test_selected, x_test_scaled, y_test, fitted_scaler=scaler
    )
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.RandomSampler(seed=42)
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
