import os
import sys
import time
from pathlib import Path
from typing import Optional, Any

import mlflow
import tempfile
import json
import yaml
from mlflow.tracking import MlflowClient

from ..core.logger import get_logger

logger = get_logger(__name__)

# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent.parent
    if not project_root.exists():
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    sys.path.append(str(project_root))
except Exception as e:
    logger.error(f"Error setting project root path: {e}")
    sys.path.append(os.getcwd().parent)
    logger.info(f"Current directory mlflow_utils: {os.getcwd().parent}")


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
        """Persist signature and input_example as JSON artifacts to MLflow.

        This is a best-effort helper which will log exceptions but not raise.
        """
        try:
            if signature is not None:
                try:
                    sig_obj = signature.to_dict() if hasattr(signature, "to_dict") else signature
                    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as fh:
                        json.dump(sig_obj, fh, default=str)
                        sig_path = fh.name
                    mlflow.log_artifact(sig_path, artifact_path=artifact_path)
                    os.unlink(sig_path)
                except Exception:
                    self.logger.exception("Failed to serialize/log model signature")

            if input_example is not None:
                try:
                    ie = input_example
                    if hasattr(ie, "to_dict"):
                        ie_obj = ie.to_dict()
                    elif hasattr(ie, "tolist"):
                        ie_obj = ie.tolist()
                    else:
                        ie_obj = str(ie)

                    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as fh:
                        json.dump(ie_obj, fh, default=str)
                        ie_path = fh.name
                    mlflow.log_artifact(ie_path, artifact_path=artifact_path)
                    os.unlink(ie_path)
                except Exception:
                    self.logger.exception("Failed to serialize/log input_example")
        except Exception:
            self.logger.exception(
                "Failed while attempting to log signature/input_example artifacts"
            )

    def _safe_log_artifacts_with_signature(
        self,
        local_model_dir: Path,
        artifact_path: str,
        signature: Optional[Any],
        input_example: Optional[Any],
    ) -> None:
        """Fallback artifact logging that also attempts to persist signature and input_example."""
        mlflow.log_artifacts(str(local_model_dir), artifact_path=artifact_path)
        # Try to log signature and input_example, but do not fail the caller
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
        """Log local model artifacts to MLflow and optionally register as a model in the MLflow Model Registry.

        Returns a dict with run_id and model_uri and registration info when available.
        """
        if not local_model_dir.exists():
            raise FileNotFoundError(local_model_dir)

        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            # If a sklearn model object is provided, log it using the sklearn flavor
            if sk_model is not None:
                try:
                    kwargs = {"artifact_path": artifact_path}
                    if signature is not None:
                        kwargs["signature"] = signature
                    if input_example is not None:
                        kwargs["input_example"] = input_example
                    mlflow.sklearn.log_model(sk_model, **kwargs)
                except Exception:
                    self.logger.exception(
                        "mlflow.sklearn.log_model failed, falling back to log_artifacts"
                    )
                    self._safe_log_artifacts_with_signature(
                        local_model_dir, artifact_path, signature, input_example
                    )
            else:
                # Log artifacts under a stable artifact_path and attempt to persist signature/input_example
                self._safe_log_artifacts_with_signature(
                    local_model_dir, artifact_path, signature, input_example
                )
            # Log package versions as a JSON artifact for provenance
            try:
                pkg_versions = self._get_package_versions()
                mlflow.log_dict(pkg_versions, "package_versions.json")
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

    def normalize_meta_yaml_paths(self, meta_path: Path, new_base_path: str) -> None:
        """Normalize paths in meta.yaml to match the current machine"""
        try:
            if not meta_path.exists():
                logger.info(f"Meta.yaml not found at {meta_path}, creating a new one.")
                path_parts = str(meta_path).split("\\")
                if len(path_parts) >= 3 and path_parts[-3].isdigit() and path_parts[-2].isdigit():
                    experiment_id = path_parts[-3]
                    run_id = path_parts[-2]
                    self.create_missing_meta_yaml(meta_path.parent, experiment_id, run_id)
                    logger.info("Meta.yaml has experiment and run id")
                elif len(path_parts) >= 2 and path_parts[-2].isdigit():
                    experiment_id = path_parts[-2]
                    self.create_missing_meta_yaml(meta_path.parent, experiment_id)
                    logger.info("Meta.yaml has only experiment id")
                return

            with open(meta_path) as f:
                meta_data = yaml.safe_load(f)
            # Check if artifact_uri exists
            if "artifact_uri" not in meta_data:
                old_uri = meta_data["artifact_location"]
                # Extract the old URI and convert to Path object
                if old_uri.startswith("file:///"):
                    old_path = Path(old_uri[8:])  # Remove 'file:///' prefix
                else:
                    old_path = Path(old_uri)
                # Get the experiment and run ID from the path
                path_parts = old_path.parts
                experiment_id = path_parts[-1]
                # Construct new path using the base path and relative components
                new_uri = f"{new_base_path}/{experiment_id}"
                logger.info(f"new_uri: {new_uri}")
                # Update with normalized path
                meta_data["artifact_location"] = new_uri.replace("\\", "/")
            else:
                old_uri = meta_data["artifact_uri"]
                # Extract the old URI and convert to Path object
                if old_uri.startswith("file:///"):
                    old_path = Path(old_uri[8:])  # Remove 'file:///' prefix
                else:
                    old_path = Path(old_uri)

                # Get the experiment and run ID from the path
                path_parts = old_path.parts
                experiment_id = path_parts[-3]
                run_id = path_parts[-2]

                # Construct new path using the base path and relative components
                new_uri = f"{new_base_path}/{experiment_id}/{run_id}/artifacts"
                logger.info(f"new_uri: {new_uri}")
                meta_data["artifact_uri"] = new_uri.replace("\\", "/")
            with open(meta_path, "w") as f:
                yaml.safe_dump(meta_data, f)
        except Exception as e:
            self.logger.error(f"Error normalizing meta.yaml paths: {str(e)}")
            return

    def create_missing_meta_yaml(
        self, run_path: Path, experiment_id: str, run_id: Optional[str] = None
    ) -> None:
        """Create meta.yaml file if it doesn't exist"""
        try:
            if isinstance(run_path, str):
                run_path = Path(run_path)

            meta_path = run_path / "meta.yaml"

            if run_id is None:
                meta_data = {
                    "artifact_location": f"file:///{run_path}",
                    "creation_time": int(time.time() * 1000),
                    "experiment_id": experiment_id,
                    "last_update_time": int(time.time() * 1000),
                    "lifecycle_stage": "active",
                    "name": f"experiment_{experiment_id}",
                }
            else:
                meta_data = {
                    "artifact_uri": f"file:///{run_path}/artifacts",
                    "end_time": int(time.time() * 1000),
                    "entry_point_name": "",
                    "experiment_id": experiment_id,
                    "lifecycle_stage": "active",
                    "run_id": run_id,
                    "run_name": f"run_{run_id}",
                    "run_uuid": run_id,
                    "source_name": "",
                    "source_type": 4,
                    "source_version": "",
                    "start_time": int(time.time() * 1000),
                    "status": 3,
                    "tags": [],
                    "user_id": os.getlogin(),
                }
            # Write the file
            with open(meta_path, "w") as f:
                yaml.safe_dump(meta_data, f)
            self.logger.info(f"Created missing meta.yaml at {meta_path}")
        except Exception as e:
            self.logger.error(f"Error creating meta.yaml: {str(e)}")
            raise

    def _create_meta_yaml_if_missing(self, meta_path: Path, path_parts: tuple[str, ...]) -> None:
        """Create a meta.yaml file for the given path_parts if it doesn't exist.

        This centralizes logic for deciding whether a directory represents an
        experiment or a run and delegates to `create_missing_meta_yaml`.
        """
        # If meta already exists, nothing to do
        if meta_path.exists():
            return

        try:
            # path_parts is a tuple of path components; try to detect run vs experiment
            if len(path_parts) >= 3 and path_parts[-3].isdigit() and path_parts[-2].isdigit():
                experiment_id = path_parts[-3]
                run_id = path_parts[-2]
                self.create_missing_meta_yaml(meta_path.parent, experiment_id, run_id)
                self.logger.info(f"Created missing run meta.yaml at {meta_path}")
            elif len(path_parts) >= 2 and path_parts[-2].isdigit():
                experiment_id = path_parts[-2]
                self.create_missing_meta_yaml(meta_path.parent, experiment_id)
                self.logger.info(f"Created missing experiment meta.yaml at {meta_path}")
            elif len(path_parts) >= 1 and path_parts[-1].isdigit():
                # Fallback: last part looks like an experiment id
                experiment_id = path_parts[-1]
                self.create_missing_meta_yaml(meta_path.parent, experiment_id)
                self.logger.info(f"Created missing experiment meta.yaml at {meta_path}")
            else:
                # Unable to infer ids; create a generic meta.yaml for safety
                self.create_missing_meta_yaml(meta_path.parent, "unknown")
                self.logger.info(f"Created generic meta.yaml at {meta_path}")
        except Exception as exc:
            self.logger.error(f"Error creating meta.yaml at {meta_path}: {exc}")
            raise

    def normalize_all_meta_yaml_paths(self) -> None:
        """Normalize paths in all meta.yaml files in the mlruns directory"""
        mlruns_path = Path(self.mlruns_dir)
        if not mlruns_path.exists():
            self.logger.error(f"mlruns directory not found: {mlruns_path}")
            return

        local_path = self.config.config["tracking_uri"]

        for root, _dirs, _files in os.walk(mlruns_path):
            path_parts = Path(root).parts
            # Only consider experiment/run-like directories (last component is numeric)
            if not path_parts or not path_parts[-1].isdigit():
                continue

            meta_path = Path(root) / "meta.yaml"

            try:
                # Ensure meta.yaml exists for this path
                self._create_meta_yaml_if_missing(meta_path, path_parts)
            except Exception:
                # Error already logged in helper; skip this path
                continue

            # Normalize the paths in the meta.yaml
            try:
                self.normalize_meta_yaml_paths(meta_path, local_path)
                self.logger.info(f"Processed: {meta_path}")
            except Exception as exc:
                self.logger.error(f"Error processing {meta_path}: {exc}")
                continue

        self.logger.info("Completed processing all meta.yaml files")


# Usage example:
if __name__ == "__main__":
    manager = MLFlowManager()
    client = MlflowClient()

    try:
        manager.normalize_all_meta_yaml_paths()
        logger.info("Successfully processed all meta.yaml files")
    except Exception as e:
        logger.error(f"Error processing meta.yaml files: {e}")
