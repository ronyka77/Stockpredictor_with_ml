import json
from src.models.predictors.autogluon_predictor import AutoGluonPredictor


def test_load_model_from_mlflow_logs_warning_for_missing_dir(tmp_path):
    """Load model when model directory missing should not raise; logs a warning."""
    p = AutoGluonPredictor(model_dir=str(tmp_path / "nope"))
    # Should not raise
    p.load_model_from_mlflow()


def test__load_metadata_reads_json(tmp_path):
    """Read best_model_metadata.json and set predictor optimal_threshold accordingly."""
    md = {"optimal_threshold": 0.3, "best_model_name": "m"}
    d = tmp_path / "mymodel"
    d.mkdir()
    with open(d / "best_model_metadata.json", "w") as f:
        json.dump(md, f)
    p = AutoGluonPredictor(model_dir=str(d))
    p.load_model_from_mlflow()
    p._load_metadata()
    assert p.optimal_threshold == 0.3
