"""
MLP All-Run Predictor

This module discovers all non-deleted MLflow runs for a given experiment name,
runs the prediction pipeline for each using the common `BasePredictor` flow
via `MLPPredictorWrapper`, and exports ONLY the best-performing run's predictions
(by highest average profit per $100 investment during prediction testing).
"""

from typing import List, Tuple, Optional
import mlflow
from mlflow.tracking import MlflowClient

from src.models.predictors.mlp_predictor import MLPPredictorWrapper
from src.utils.core.logger import get_logger


logger = get_logger(__name__)


def get_active_run_ids_for_experiment(experiment_name: str) -> List[str]:
    """
    Return all non-deleted run IDs for the given experiment name.
    """
    client = MlflowClient()

    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        logger.warning(f"⚠️ Experiment not found: {experiment_name}")
        return []

    runs = client.search_runs(
        [exp.experiment_id], run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY
    )
    run_ids = [r.info.run_id for r in runs]
    logger.info(f"🔎 Found {len(run_ids)} active runs in experiment '{experiment_name}'")
    return run_ids


def run_all_and_export_best(
    *, experiment_name: str, days_back: int = 30
) -> Optional[Tuple[str, float, str]]:
    """
    Evaluate all active runs in the experiment and export only the best one's predictions.

    Returns (best_run_id, best_avg_profit, output_file_path) if any run succeeded, else None.
    """
    run_ids = get_active_run_ids_for_experiment(experiment_name)
    if not run_ids:
        return None

    best_tuple: Optional[Tuple[str, float, Tuple]] = None  # (run_id, avg_profit, cached_data)

    for run_id in run_ids:
        logger.info("=" * 80)
        logger.info(f"▶️ Evaluating run: {run_id}")
        predictor = MLPPredictorWrapper(run_id=run_id)
        try:
            results_df, avg_profit, features_df, metadata_df, predictions = (
                predictor.evaluate_on_recent_data(days_back=days_back)
            )
        except Exception as e:
            logger.warning(f"   ❌ Skipping run due to error: {e}")
            continue

        if best_tuple is None or (avg_profit is not None and avg_profit > best_tuple[1]):
            best_tuple = (run_id, avg_profit, (features_df, metadata_df, predictions))
            logger.info(f"   ⭐ New best so far: {run_id} with avg profit ${avg_profit:.2f}")

    if best_tuple is None:
        logger.warning("No valid runs produced predictions.")
        return None

    best_run_id, best_avg_profit, cached = best_tuple
    logger.info("=" * 80)
    logger.info(f"🏁 Best run: {best_run_id} with avg profit ${best_avg_profit:.2f}")

    # Export only the best run's predictions using the cached results
    best_predictor = MLPPredictorWrapper(run_id=best_run_id)
    best_predictor.load_model_from_mlflow()
    best_predictor._load_metadata_from_mlflow()
    features_df, metadata_df, predictions = cached
    output_file = best_predictor.save_predictions_to_excel(features_df, metadata_df, predictions)
    logger.info(f"📄 Exported best predictions to: {output_file}")

    return best_run_id, best_avg_profit, output_file


def main():
    # Default experiment name used by MLP training
    experiment_name = "mlp_stock_predictor"
    days_back = 30
    result = run_all_and_export_best(experiment_name=experiment_name, days_back=days_back)
    if result is None:
        logger.warning("No predictions were exported.")
    else:
        best_run_id, best_avg_profit, output_file = result
        logger.info(
            f"Best run {best_run_id} exported with avg profit ${best_avg_profit:.2f} -> {output_file}"
        )


if __name__ == "__main__":
    main()
