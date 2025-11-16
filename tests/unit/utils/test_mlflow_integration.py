from unittest.mock import Mock, patch

import pytest

from src.utils.mlops.mlflow_integration import (
    MLflowIntegration,
    cleanup_deleted_runs,
    cleanup_empty_experiments,
)


@pytest.mark.parametrize("existing_experiment", [True, False])
def test_setup_experiment_creates_or_uses_existing(existing_experiment, patch_mlflow_start):
    """Create or reuse an MLflow experiment depending on existence."""
    fake_get = Mock()
    if existing_experiment:
        fake_exp = Mock()
        fake_exp.experiment_id = "123"
        fake_get.return_value = fake_exp
    else:
        fake_get.return_value = None

    with (
        patch("mlflow.get_experiment_by_name", new=fake_get),
        patch("mlflow.create_experiment", new=Mock(return_value="created-id")),
        patch("mlflow.set_experiment", new=Mock()),
    ):
        mlfi = MLflowIntegration()
        result = mlfi.setup_experiment("my_experiment")

    assert (result == "123") if existing_experiment else (result == "created-id")


def test_setup_experiment_with_artifact_location_calls_create():
    """Create experiment with explicit artifact location when not present."""
    fake_get = Mock(return_value=None)
    fake_create = Mock(return_value="exp-created")

    with (
        patch("mlflow.get_experiment_by_name", new=fake_get),
        patch("mlflow.create_experiment", new=fake_create),
        patch("mlflow.set_experiment", new=Mock()),
    ):
        mlfi = MLflowIntegration()
        eid = mlfi.setup_experiment("e", artifact_location="/tmp/artifacts")

    fake_create.assert_called_once()
    assert eid == "exp-created"


def test_setup_experiment_propagates_exception():
    """Propagate exceptions raised while checking experiments during setup."""
    with patch("mlflow.get_experiment_by_name", side_effect=Exception("boom")):
        mlfi = MLflowIntegration()
        with pytest.raises(Exception):
            mlfi.setup_experiment("x")


def test_start_run_logs_and_returns(patch_mlflow_start):
    """Start an MLflow run and return the run object with IDs."""
    fake_run = Mock()
    fake_run.info.run_id = "run-1"
    fake_run.info.experiment_id = "exp-1"

    with patch("mlflow.start_run", new=Mock(return_value=fake_run)):
        mlfi = MLflowIntegration()
        run = mlfi.start_run(run_name="r", tags={"k": "v"})

    assert run.info.run_id == "run-1"


def test_start_run_propagates_exception():
    """Propagate exceptions from mlflow.start_run when it fails."""
    with patch("mlflow.start_run", side_effect=Exception("no-start")):
        mlfi = MLflowIntegration()
        with pytest.raises(Exception):
            mlfi.start_run()


def test_log_params_and_metrics_use_active_or_run():
    """Log parameters and metrics using mlflow APIs when no run_id provided."""
    with (
        patch("mlflow.log_params", new=Mock()) as fake_log_params,
        patch("mlflow.log_metrics", new=Mock()) as fake_log_metrics,
    ):
        mlfi = MLflowIntegration()
        mlfi.log_params({"p": 1})
        mlfi.log_metrics({"m": 0.5}, step=1)

    fake_log_params.assert_called_once_with({"p": 1})
    fake_log_metrics.assert_called_once_with({"m": 0.5}, step=1)


def test_log_params_and_metrics_with_run_id_uses_start_run(patch_mlflow_start):
    """When run_id provided, ensure start_run is invoked for logging calls."""
    mlfi = MLflowIntegration()
    mlfi.log_params({"k": "v"}, run_id="run-1")
    mlfi.log_metrics({"m": 0.1}, run_id="run-1")

    assert patch_mlflow_start.called


def test_log_model_unsupported_flavor_raises():
    """Raise ValueError for unsupported model flavors in logging."""
    mlfi = MLflowIntegration()
    with pytest.raises(ValueError):
        mlfi.log_model(object(), "model_path", flavor="unknown")


def test_log_and_load_model_flow():
    """Log a sklearn model artifact and load it back via pyfunc."""
    fake_model = object()

    with (
        patch("mlflow.sklearn.log_model", new=Mock()),
        patch("mlflow.pyfunc.load_model", new=Mock(return_value=fake_model)),
    ):
        mlfi = MLflowIntegration()
        mlfi.log_model(fake_model, "artifact", flavor="sklearn")
        loaded = mlfi.load_model("some-run", "model")

    assert loaded is fake_model


def test_log_model_xgboost_with_signature_and_run_id(patch_mlflow_start):
    """Log an XGBoost model including signature and run_id via mlflow.xgboost."""
    fake_signature = Mock()
    with patch("mlflow.xgboost.log_model", new=Mock()) as fake_log:
        mlfi = MLflowIntegration()
        mlfi.log_model(object(), "artifact", flavor="xgboost", signature=fake_signature, run_id="r")

    fake_log.assert_called_once()


def test_log_model_raises_and_logs_when_logfunc_fails():
    """Raise and propagate exceptions when mlflow model log function fails."""
    with patch("mlflow.sklearn.log_model", side_effect=Exception("bad")):
        mlfi = MLflowIntegration()
        with pytest.raises(Exception):
            mlfi.log_model(object(), "a", flavor="sklearn")


def test_log_artifact_with_run_id_calls_mlflow(patch_mlflow_start):
    """Log an artifact to mlflow using a specific run_id."""
    with patch("mlflow.log_artifact", new=Mock()) as fake_art:
        mlfi = MLflowIntegration()
        mlfi.log_artifact("/tmp/file.txt", artifact_path="a/path", run_id="r")

    fake_art.assert_called_once()


def test_log_artifact_raises_when_mlflow_fails():
    """Propagate exceptions raised while logging artifacts to mlflow."""
    with patch("mlflow.log_artifact", side_effect=Exception("nope")):
        mlfi = MLflowIntegration()
        with pytest.raises(Exception):
            mlfi.log_artifact("/tmp/x")


def test_load_model_raises_and_logs():
    """Propagate exceptions when loading a model via mlflow fails."""
    with patch("mlflow.pyfunc.load_model", side_effect=Exception("nope")):
        mlfi = MLflowIntegration()
        with pytest.raises(Exception):
            mlfi.load_model("bad-run")


def test_cleanup_deleted_and_empty_runs(tmp_mlruns_dir, patch_mlflow_client):
    """Remove deleted run directories and empty experiments from mlruns tree."""
    mlruns_dir = tmp_mlruns_dir
    exp_dir = mlruns_dir / "1"
    run_dir = exp_dir / "2"
    run_dir.mkdir(parents=True)
    (run_dir / "artifacts").mkdir()

    cleanup_deleted_runs(str(mlruns_dir))
    cleanup_empty_experiments(str(mlruns_dir))

    assert not run_dir.exists() or any(True for _ in run_dir.iterdir()) is False


def test_cleanup_deleted_runs_handles_rmtree_exception(tmp_mlruns_dir, patch_mlflow_client):
    """Handle exceptions thrown by rmtree when clearing deleted runs."""
    mlruns_dir = tmp_mlruns_dir
    exp_dir = mlruns_dir / "1"
    run_dir = exp_dir / "2"
    (run_dir / "artifacts").mkdir(parents=True)

    with (
        patch("os.path.exists", return_value=True),
        patch("shutil.rmtree", side_effect=OSError("boom")),
    ):
        cleanup_deleted_runs(str(mlruns_dir))


def test_cleanup_deleted_runs_handles_search_experiments_exception(patch_mlflow_client):
    """Raise when search_experiments fails during cleanup of deleted runs."""
    patch_mlflow_client.search_experiments.side_effect = Exception("boom")
    with pytest.raises(Exception):
        cleanup_deleted_runs("mlruns")


def test_cleanup_empty_experiments_handles_delete_exception(tmp_mlruns_dir, patch_mlflow_client):
    """Safely handle exceptions thrown when deleting empty experiments."""
    mlruns_dir = tmp_mlruns_dir
    (mlruns_dir / "1").mkdir(parents=True)

    patch_mlflow_client.search_experiments.return_value = [Mock(name="e", experiment_id="1")]
    patch_mlflow_client.search_runs.return_value = []
    patch_mlflow_client.delete_experiment.side_effect = Exception("del fail")

    with patch("os.path.exists", return_value=True), patch("shutil.rmtree", new=Mock()):
        cleanup_empty_experiments(str(mlruns_dir))


def test_cleanup_empty_experiments_handles_search_experiments_exception(patch_mlflow_client):
    """Raise when search_experiments fails during empty-experiment cleanup."""
    patch_mlflow_client.search_experiments.side_effect = Exception("boom")
    with pytest.raises(Exception):
        cleanup_empty_experiments("mlruns")
