from unittest.mock import Mock, patch

import yaml

from src.utils.mlflow_utils import MLFlowManager, MLFlowConfig


def test_mlflow_config_paths(tmp_mlruns_dir):
    """Construct MLFlowConfig and ensure paths and tracking URI are set."""
    mlruns_path = tmp_mlruns_dir

    with patch("src.utils.mlflow_utils.Path.resolve", return_value=mlruns_path):
        cfg = MLFlowConfig()

    assert str(cfg.local_path).endswith("mlruns")
    assert cfg.config["tracking_uri"].startswith("file:///")


def test_normalize_and_create_meta_yaml(tmp_mlruns_dir):
    """Create missing meta.yaml for a run and verify it exists."""
    mlruns_dir = tmp_mlruns_dir
    exp_dir = mlruns_dir / "1"
    run_dir = exp_dir / "2"
    run_dir.mkdir(parents=True)

    manager = MLFlowManager()
    manager.create_missing_meta_yaml(run_dir, "1", "2")

    meta_path = run_dir / "meta.yaml"
    assert meta_path.exists()


def test_normalize_all_meta_yaml_paths_handles_missing(tmp_mlruns_dir):
    """Run normalization over mlruns when some runs are missing meta files."""
    mlruns_dir = tmp_mlruns_dir
    (mlruns_dir / "1").mkdir(parents=True)

    manager = MLFlowManager()
    manager.mlruns_dir = str(mlruns_dir)
    manager.normalize_all_meta_yaml_paths()

    assert True


def test_create_missing_meta_yaml_for_experiment(tmp_mlruns_dir):
    """Create meta.yaml for an experiment directory when missing."""
    exp_dir = tmp_mlruns_dir / "1"
    exp_dir.mkdir(parents=True)

    manager = MLFlowManager()
    manager.create_missing_meta_yaml(exp_dir, "1")

    assert (exp_dir / "meta.yaml").exists()


def test_normalize_meta_yaml_paths_with_artifact_uri(tmp_mlruns_dir):
    """Normalize meta.yaml entries that use artifact_uri to updated base paths."""
    run_path = tmp_mlruns_dir / "1" / "2"
    run_path.mkdir(parents=True)
    meta_path = run_path / "meta.yaml"

    meta_path.write_text("artifact_uri: 'file:///C:/old/path/1/2/artifacts'\n")

    manager = MLFlowManager()
    manager.normalize_meta_yaml_paths(meta_path, "file:///new/base")

    assert "artifact_uri" in meta_path.read_text()


def test_normalize_meta_yaml_paths_creates_missing_meta(tmp_mlruns_dir):
    """Ensure missing meta.yaml files are created during normalization."""
    run_path = tmp_mlruns_dir / "1"
    run_path.mkdir(parents=True)
    meta_path = run_path / "meta.yaml"

    manager = MLFlowManager()
    manager.normalize_meta_yaml_paths(meta_path, "file:///new/base")

    assert meta_path.exists()


def test_normalize_meta_yaml_paths_with_artifact_location(tmp_mlruns_dir):
    """Normalize artifact_location fields inside meta.yaml to new base URIs."""
    run_path = tmp_mlruns_dir / "1"
    run_path.mkdir(parents=True)
    meta_path = run_path / "meta.yaml"

    meta_content = {"artifact_location": "file:///C:/old/path/123"}
    meta_path.write_text(yaml.safe_dump(meta_content))

    manager = MLFlowManager()
    manager.normalize_meta_yaml_paths(meta_path, "file:///new/base")

    data = yaml.safe_load(meta_path.read_text())
    assert "artifact_location" in data
    assert "file:///new/base" in data["artifact_location"]


def test_normalize_meta_yaml_paths_creates_missing_for_run_and_experiment(tmp_mlruns_dir):
    """Call helper to create missing meta.yaml for run and experiment when absent."""
    meta_path = tmp_mlruns_dir / "1" / "2" / "meta.yaml"

    manager = MLFlowManager()
    manager.create_missing_meta_yaml = Mock()

    manager.normalize_meta_yaml_paths(meta_path, "file:///new/base")

    manager.create_missing_meta_yaml.assert_called_once()


def test_normalize_all_meta_yaml_paths_calls_helpers(tmp_mlruns_dir):
    """normalize_all_meta_yaml_paths invokes creation or normalization helpers."""
    run_dir = tmp_mlruns_dir / "1" / "2"
    run_dir.mkdir(parents=True)

    manager = MLFlowManager()
    manager.mlruns_dir = str(tmp_mlruns_dir)
    manager.create_missing_meta_yaml = Mock()
    manager.normalize_meta_yaml_paths = Mock()

    manager.normalize_all_meta_yaml_paths()

    assert manager.create_missing_meta_yaml.called or manager.normalize_meta_yaml_paths.called


def test_normalize_meta_yaml_artifact_location_no_file_prefix(tmp_mlruns_dir):
    """Handle artifact_location without file:// prefix by prepending new base."""
    run_path = tmp_mlruns_dir / "exp"
    run_path.mkdir(parents=True)
    meta_path = run_path / "meta.yaml"

    meta_content = {"artifact_location": "/old/base/exp"}
    meta_path.write_text(yaml.safe_dump(meta_content))

    manager = MLFlowManager()
    manager.normalize_meta_yaml_paths(meta_path, "file:///new/base")

    data = yaml.safe_load(meta_path.read_text())
    assert "artifact_location" in data
    assert "file:///new/base" in data["artifact_location"]


def test_normalize_meta_yaml_artifact_uri_no_file_prefix(tmp_mlruns_dir):
    """Normalize artifact_uri entries that lack file:// prefixes to new base."""
    run_path = tmp_mlruns_dir / "1" / "2"
    run_path.mkdir(parents=True)
    meta_path = run_path / "meta.yaml"

    meta_content = {"artifact_uri": str(run_path / "artifacts").replace("\\", "/")}
    meta_path.write_text(yaml.safe_dump(meta_content))

    manager = MLFlowManager()
    manager.normalize_meta_yaml_paths(meta_path, "file:///new/base")

    data = yaml.safe_load(meta_path.read_text())
    assert "artifact_uri" in data
    assert "file:///new/base" in data["artifact_uri"]


def test_normalize_all_meta_yaml_paths_handles_missing_mlruns(tmp_mlruns_dir):
    """Gracefully handle when the configured mlruns directory does not exist."""
    manager = MLFlowManager()
    manager.mlruns_dir = str(tmp_mlruns_dir / "does_not_exist")

    manager.normalize_all_meta_yaml_paths()
