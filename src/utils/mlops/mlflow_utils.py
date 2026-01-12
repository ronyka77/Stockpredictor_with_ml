import os
import sys
import time
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass

import mlflow
import tempfile
import json
import yaml
from mlflow.tracking import MlflowClient

from ..core.logger import get_logger

logger = get_logger(__name__)

# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent.parent.parent
    if not project_root.exists():
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    sys.path.append(str(project_root))
except Exception as e:
    logger.error(f"Error setting project root path: {e}")
    sys.path.append(os.getcwd().parent)
    logger.info(f"Current directory mlflow_utils: {os.getcwd().parent}")


@dataclass
class PathIds:
    """Container for experiment and run IDs parsed from paths"""

    experiment_id: str
    run_id: Optional[str] = None


class MetaYamlHandler:
    """Handles all meta.yaml file operations for MLflow"""

    def __init__(self, mlruns_dir: str, tracking_uri: str):
        self.mlruns_dir = mlruns_dir
        self.tracking_uri = tracking_uri
        self.logger = get_logger(__name__)

    def _parse_path_ids(self, path: Path) -> Optional[PathIds]:
        """Extract experiment_id and optional run_id from path structure"""
        parts = path.parts
        if len(parts) >= 3 and parts[-3].isdigit() and parts[-2].isdigit():
            return PathIds(experiment_id=parts[-3], run_id=parts[-2])
        elif len(parts) >= 2 and parts[-2].isdigit():
            return PathIds(experiment_id=parts[-2])
        elif len(parts) >= 1 and parts[-1].isdigit():
            return PathIds(experiment_id=parts[-1])
        return None

    def _create_meta_data(self, run_path: Path, path_ids: PathIds) -> dict:
        """Generate meta.yaml content based on whether it's an experiment or run"""
        timestamp = int(time.time() * 1000)
        if path_ids.run_id is None:
            return {
                "artifact_location": f"file:///{run_path}",
                "creation_time": timestamp,
                "experiment_id": path_ids.experiment_id,
                "last_update_time": timestamp,
                "lifecycle_stage": "active",
                "name": f"experiment_{path_ids.experiment_id}",
            }
        return {
            "artifact_uri": f"file:///{run_path}/artifacts",
            "end_time": timestamp,
            "entry_point_name": "",
            "experiment_id": path_ids.experiment_id,
            "lifecycle_stage": "active",
            "run_id": path_ids.run_id,
            "run_name": f"run_{path_ids.run_id}",
            "run_uuid": path_ids.run_id,
            "source_name": "",
            "source_type": 4,
            "source_version": "",
            "start_time": timestamp,
            "status": 3,
            "tags": [],
            "user_id": os.getlogin(),
        }

    def _normalize_uri(self, uri: str, path_ids: PathIds) -> str:
        """Generate normalized URI based on path structure"""
        if path_ids.run_id:
            return f"{self.tracking_uri}/{path_ids.experiment_id}/{path_ids.run_id}/artifacts"
        return f"{self.tracking_uri}/{path_ids.experiment_id}"

    def ensure_and_normalize(self, meta_path: Path) -> None:
        """Create meta.yaml if missing, normalize if exists"""
        path_ids = self._parse_path_ids(meta_path.parent)
        if not path_ids:
            self.logger.warning(f"Cannot parse IDs from path: {meta_path.parent}")
            return

        if not meta_path.exists():
            self._create_meta_yaml(meta_path, path_ids)
        else:
            self._normalize_existing_meta(meta_path, path_ids)

    def _create_meta_yaml(self, meta_path: Path, path_ids: PathIds) -> None:
        """Create new meta.yaml file"""
        try:
            meta_data = self._create_meta_data(meta_path.parent, path_ids)
            with open(meta_path, "w") as f:
                yaml.safe_dump(meta_data, f)
            meta_type = "run" if path_ids.run_id else "experiment"
            self.logger.info(f"Created {meta_type} meta.yaml at {meta_path}")
        except Exception as e:
            self.logger.error(f"Error creating meta.yaml at {meta_path}: {e}")
            raise

    def _normalize_existing_meta(self, meta_path: Path, path_ids: PathIds) -> None:
        """Normalize paths in existing meta.yaml"""
        try:
            with open(meta_path) as f:
                meta_data = yaml.safe_load(f)

            key = "artifact_uri" if "artifact_uri" in meta_data else "artifact_location"
            meta_data[key] = self._normalize_uri(self.tracking_uri, path_ids).replace("\\", "/")

            with open(meta_path, "w") as f:
                yaml.safe_dump(meta_data, f)
        except Exception as e:
            self.logger.error(f"Error normalizing meta.yaml at {meta_path}: {e}")

    def process_all(self) -> None:
        """Process all meta.yaml files in mlruns directory"""
        mlruns_path = Path(self.mlruns_dir)
        if not mlruns_path.exists():
            self.logger.error(f"mlruns directory not found: {mlruns_path}")
            return

        for root, _dirs, _files in os.walk(mlruns_path):
            path_parts = Path(root).parts
            if not path_parts or not path_parts[-1].isdigit():
                continue

            meta_path = Path(root) / "meta.yaml"
            try:
                self.ensure_and_normalize(meta_path)
                self.logger.info(f"Processed: {meta_path}")
            except Exception as exc:
                self.logger.error(f"Error processing {meta_path}: {exc}")
                continue

        self.logger.info("Completed processing all meta.yaml files")


class MLFlowConfig:
    """MLflow configuration management"""

    def __init__(self):
        self.local_path = project_root / "mlruns"
        self.local_path_uri = str(self.local_path).replace("\\", "/")

        # Ensure paths are absolute and normalized
        self.local_path = self.local_path.resolve()
        logger.info(f"local_path: {self.local_path}")
        # Convert to proper MLflow URI format
        local_uri = f"file:///{self.local_path_uri}"

        self.config = {"tracking_uri": local_uri, "local_path_uri": self.local_path_uri}

        # Initialize logger
        self.logger = get_logger(__name__)

        # Ensure local directory exists
        self.local_path.mkdir(parents=True, exist_ok=True)


class MLFlowManager:
    """MLflow experiment and model management"""

    def __init__(self):
        self.config = MLFlowConfig()
        self.logger = self.config.logger
        self.mlruns_dir = self.config.config["local_path_uri"]
        self.meta_handler = MetaYamlHandler(self.mlruns_dir, self.config.config["tracking_uri"])

        # Set tracking URI on initialization
        mlflow.set_tracking_uri(self.config.config["tracking_uri"])

    def setup_experiment(self, experiment_name: Optional[str] = None) -> str:
        """Setup MLflow experiment"""
        name = experiment_name
        try:
            experiment = mlflow.get_experiment_by_name(name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(name)
                mlflow.set_experiment(name)
                self.logger.info(f"Created new experiment: {name} experiment_id: {experiment_id}")
            else:
                experiment_id = experiment.experiment_id
                mlflow.set_experiment(name)
                self.logger.info(
                    f"Using existing experiment: {name} experiment_id: {experiment_id}"
                )
            return experiment_id
        except Exception as e:
            self.logger.error(f"Error setting up experiment: {e}")
            raise

    def _get_package_versions(self) -> dict:
        """Return a mapping of package -> version for common ML packages."""
        try:
            import importlib.metadata as importlib_metadata
        except Exception:
            import importlib_metadata

        pkgs = ["scikit-learn", "skforecast", "xgboost", "lightgbm", "joblib", "mlflow"]
        versions = {}
        for pkg in pkgs:
            try:
                versions[pkg] = importlib_metadata.version(pkg)
            except Exception:
                # package not installed or version unavailable
                continue
        versions["python_version"] = sys.version.split()[0]
        return versions

    def _log_signature_and_input_example(
        self, signature: Optional[Any], input_example: Optional[Any], artifact_path: str
    ) -> None:
        """Persist signature and input_example as JSON artifacts to MLflow"""
        items = [
            ("signature", signature, lambda s: s.to_dict() if hasattr(s, "to_dict") else s),
            (
                "input_example",
                input_example,
                lambda ie: ie.to_dict()
                if hasattr(ie, "to_dict")
                else ie.tolist()
                if hasattr(ie, "tolist")
                else str(ie),
            ),
        ]

        for name, item, converter in items:
            if item is None:
                continue
            try:
                obj = converter(item)
                with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as fh:
                    json.dump(obj, fh, default=str)
                    temp_path = fh.name
                mlflow.log_artifact(temp_path, artifact_path=artifact_path)
                os.unlink(temp_path)
            except Exception:
                self.logger.exception(f"Failed to serialize/log {name}")

    def _safe_log_artifacts_with_signature(
        self,
        local_model_dir: Path,
        artifact_path: str,
        signature: Optional[Any],
        input_example: Optional[Any],
    ) -> None:
        """Fallback artifact logging with signature and input_example"""
        mlflow.log_artifacts(str(local_model_dir), artifact_path=artifact_path)
        self._log_signature_and_input_example(signature, input_example, artifact_path)

    def log_model_registry(
        self,
        local_model_dir: Path,
        sk_model: Optional[Any] = None,
        registered_model_name: Optional[str] = None,
        run_name: Optional[str] = None,
        artifact_path: str = "model",
        signature: Optional[Any] = None,
        input_example: Optional[Any] = None,
    ) -> dict:
        """Log model artifacts to MLflow and optionally register in Model Registry"""
        if not local_model_dir.exists():
            raise FileNotFoundError(local_model_dir)

        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id

            if sk_model is not None:
                try:
                    kwargs = {"artifact_path": artifact_path}
                    if signature is not None:
                        kwargs["signature"] = signature
                    if input_example is not None:
                        kwargs["input_example"] = input_example
                    mlflow.sklearn.log_model(sk_model, **kwargs)
                except Exception:
                    self.logger.exception("mlflow.sklearn.log_model failed, falling back")
                    self._safe_log_artifacts_with_signature(
                        local_model_dir, artifact_path, signature, input_example
                    )
            else:
                self._safe_log_artifacts_with_signature(
                    local_model_dir, artifact_path, signature, input_example
                )

            try:
                mlflow.log_dict(self._get_package_versions(), "package_versions.json")
            except Exception:
                self.logger.exception("Failed to log package versions")

            model_uri = f"runs:/{run_id}/{artifact_path}"
            registration_info = None
            if registered_model_name:
                try:
                    registration = mlflow.register_model(model_uri, registered_model_name)
                    registration_info = registration.to_dictionary()
                except Exception:
                    self.logger.exception("Failed to register model in MLflow Model Registry")

            return {"run_id": run_id, "model_uri": model_uri, "registration": registration_info}

    def download_registered_model_artifacts(
        self,
        registered_model_name: str,
        version: Optional[str] = None,
        dst_path: Optional[Path] = None,
    ) -> Path:
        """Download artifacts for a registered model version to a local directory.

        If version is not provided, the latest version is used.
        """
        client = MlflowClient()
        if version is None:
            versions = client.get_latest_versions(registered_model_name)
            if not versions:
                raise ValueError(f"No registered versions found for {registered_model_name}")
            version_obj = versions[0]
        else:
            version_obj = client.get_model_version(registered_model_name, version)

        source = getattr(version_obj, "source", None)
        if not source:
            raise ValueError("Registered model version has no source URI")

        # Expect source like "runs:/<run_id>/path"
        if source.startswith("runs:/"):
            _, rest = source.split("runs:/", 1)
            parts = rest.split("/", 1)
            run_id = parts[0]
            artifact_path = parts[1] if len(parts) > 1 else ""
        else:
            raise ValueError(f"Unsupported source scheme: {source}")

        if dst_path is None:
            dst_path = Path.cwd() / f"{registered_model_name}_{version_obj.version}"
        dst_path.mkdir(parents=True, exist_ok=True)

        client.download_artifacts(run_id, artifact_path, dst_path=str(dst_path))
        return dst_path

    def normalize_all_meta_yaml_paths(self) -> None:
        """Normalize paths in all meta.yaml files in the mlruns directory"""
        self.meta_handler.process_all()


# Usage example:
if __name__ == "__main__":
    manager = MLFlowManager()
    client = MlflowClient()

    try:
        manager.normalize_all_meta_yaml_paths()
        logger.info("Successfully processed all meta.yaml files")
    except Exception as e:
        logger.error(f"Error processing meta.yaml files: {e}")
