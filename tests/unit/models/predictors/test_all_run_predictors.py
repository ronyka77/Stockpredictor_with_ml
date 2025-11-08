from unittest.mock import patch
from src.models.predictors.lightgbm_all_run_predictor import (
    get_active_run_ids_for_experiment,
    run_all_and_export_best,
)


def test_get_active_run_ids_for_nonexistent_experiment():
    """Return empty list when experiment name does not exist in mlflow."""
    with patch(
        "src.models.predictors.lightgbm_all_run_predictor.mlflow.get_experiment_by_name",
        return_value=None,
    ):
        ids = get_active_run_ids_for_experiment("nope")
        assert ids == []


def test_run_all_and_export_best_returns_none_when_no_runs():
    """Run exporter returns None when there are no active runs to evaluate."""
    with patch(
        "src.models.predictors.lightgbm_all_run_predictor.get_active_run_ids_for_experiment",
        return_value=[],
    ):
        res = run_all_and_export_best(experiment_name="x", days_back=1)
        assert res is None
