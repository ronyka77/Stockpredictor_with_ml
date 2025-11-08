from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def tmp_mlruns_dir(tmp_path):
    """Create and return a temporary mlruns dir path (Path).

    Tests can create experiments/runs under this directory and pass its
    string path to functions that expect the mlruns root.
    """
    mlruns = tmp_path / "mlruns"
    mlruns.mkdir()
    return mlruns


@pytest.fixture
def dummy_run():
    class DummyRunInfo:
        def __init__(self, run_id="2", experiment_id="1"):
            self.run_id = run_id
            self.experiment_id = experiment_id

    class DummyRun:
        def __init__(self, run_id="2", experiment_id="1"):
            self.info = DummyRunInfo(run_id, experiment_id)

    return DummyRun()


@pytest.fixture
def fake_mlflow_client(dummy_run):
    """Return a Mock configured with common search_experiments/search_runs behavior."""
    client = Mock()
    fake_exp = Mock()
    fake_exp.name = "e"
    fake_exp.experiment_id = "1"
    client.search_experiments.return_value = [fake_exp]
    client.search_runs.return_value = [dummy_run]
    return client


@pytest.fixture
def patch_mlflow_client(fake_mlflow_client):
    """Patch the MlflowClient used in `src.utils.mlflow_integration` to return our fake client."""
    with patch("src.utils.mlops.mlflow_integration.MlflowClient", return_value=fake_mlflow_client):
        yield fake_mlflow_client


@pytest.fixture
def start_run_cm():
    cm = Mock()
    cm.__enter__ = Mock(return_value=Mock())
    cm.__exit__ = Mock(return_value=None)
    return cm


@pytest.fixture
def patch_mlflow_start(start_run_cm):
    with patch("mlflow.start_run", return_value=start_run_cm) as s:
        yield s
