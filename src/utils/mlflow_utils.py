import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.xgboost
import xgboost as xgb
import yaml
from mlflow.tracking import MlflowClient

# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent.parent
    if not project_root.exists():
        # Handle network path by using raw string
        project_root = Path(r"\\".join(str(project_root).split("\\")))
    sys.path.append(str(project_root))
    print(f"Project root mlflow_utils: {project_root}")
except Exception as e:
    print(f"Error setting project root path: {e}")
    # Fallback to current directory if path resolution fails
    sys.path.append(os.getcwd().parent)
    print(f"Current directory mlflow_utils: {os.getcwd().parent}")

from src.utils.logger import get_logger


class MLFlowConfig:
    """MLflow configuration management"""

    def __init__(self):
        self.local_path = project_root / "mlruns"
        self.local_path_uri = str(self.local_path).replace("\\", "/")

        # Ensure paths are absolute and normalized
        self.local_path = self.local_path.resolve()
        print(f"local_path: {self.local_path}")
        # Convert to proper MLflow URI format
        local_uri = f"file:///{self.local_path_uri}"

        self.config = {
            "tracking_uri": local_uri,
            "local_path_uri": self.local_path_uri,
        }

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

    def load_latest_model(self, experiment_name: str):
        """Load the most recent model from the specified experiment"""
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                raise ValueError(f"Experiment {experiment_name} not found")

            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"]
            )
            if runs.empty:
                raise ValueError(f"No runs found for experiment {experiment_name}")
            
            latest_run_id = runs.iloc[0].run_id
            # Get the actual model path including experiment folders
            client = MlflowClient()
            client.get_run(latest_run_id)
            model_uri = f"runs:/{latest_run_id}/model_global"

            return mlflow.xgboost.load_model(model_uri), latest_run_id

        except Exception as e:
            self.logger.error(f"Error loading latest model: {e}")
            raise

    def normalize_meta_yaml_paths(self, meta_path: Path, new_base_path: str) -> None:
        """Normalize paths in meta.yaml to match the current machine"""
        try:
            if not meta_path.exists():
                print(f"Meta.yaml not found at {meta_path}, creating a new one.")
                path_parts = str(meta_path).split("\\")
                if len(path_parts) >= 3 and path_parts[-3].isdigit() and path_parts[-2].isdigit():
                    experiment_id = path_parts[-3]
                    run_id = path_parts[-2]
                    self.create_missing_meta_yaml(meta_path.parent, experiment_id, run_id)
                    print("Meta.yaml has experiment and run id")
                elif len(path_parts) >= 2 and path_parts[-2].isdigit():
                    experiment_id = path_parts[-2]
                    self.create_missing_meta_yaml(meta_path.parent, experiment_id)
                    print("Meta.yaml has only experiment id")
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
                print(f"new_uri: {new_uri}")
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
                print(f"new_uri: {new_uri}")
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

    def normalize_all_meta_yaml_paths(self) -> None:
        """Normalize paths in all meta.yaml files in the mlruns directory"""
        try:
            mlruns_path = Path(self.mlruns_dir)
            if not mlruns_path.exists():
                self.logger.error(f"mlruns directory not found: {mlruns_path}")
                return

            local_path = self.config.config["tracking_uri"]
            for root, _dirs, _files in os.walk(mlruns_path):
                path_parts = Path(root).parts
                if len(path_parts) >= 2 and path_parts[-1].isdigit():
                    # This is either an experiment or run directory
                    meta_path = Path(root) / "meta.yaml"
                    if not meta_path.exists():
                        try:
                            # Determine if this is an experiment or run directory
                            if len(path_parts) >= 3 and path_parts[-2].isdigit():
                                # This is a run directory
                                experiment_id = path_parts[-3]
                                run_id = path_parts[-2]
                                self.create_missing_meta_yaml(
                                    meta_path.parent, experiment_id, run_id
                                )
                                self.logger.info(f"Created missing run meta.yaml at {meta_path}")
                            else:
                                # This is an experiment directory
                                experiment_id = path_parts[-1]
                                self.create_missing_meta_yaml(meta_path.parent, experiment_id)
                                self.logger.info(
                                    f"Created missing experiment meta.yaml at {meta_path}"
                                )
                        except Exception as e:
                            self.logger.error(f"Error creating meta.yaml at {meta_path}: {e}")
                            continue

                    # Normalize the paths in the meta.yaml
                    try:
                        self.normalize_meta_yaml_paths(meta_path, local_path)
                        self.logger.info(f"Processed: {meta_path}")
                    except Exception as e:
                        self.logger.error(f"Error processing {meta_path}: {e}")
                        continue
            self.logger.info("Completed processing all meta.yaml files")

        except Exception as e:
            self.logger.error(f"Error in normalize_all_meta_yaml_paths: {e}")
            raise

    def get_run_artifact_uri(self, run_id: str) -> str:
        """Get the artifact URI for a specific MLflow run.
        Args:
            run_id: The MLflow run ID to locate
        Returns:
            String containing the artifact URI
        Raises:
            FileNotFoundError: If the run directory cannot be found
            ValueError: If the run ID is invalid
        """
        if not run_id or not isinstance(run_id, str) or len(run_id) != 32:
            raise ValueError(f"Invalid run ID format: {run_id}")

        mlruns_path = Path(self.mlruns_dir).resolve()
        tracking_uri = self.config.config.get("tracking_uri", "")
        if tracking_uri and Path(tracking_uri).exists():
            mlruns_path = Path(tracking_uri).resolve()
            self.logger.debug(f"Using tracking URI path: {mlruns_path}")

        # Search for the run directory
        try:
            # First try direct path construction (most efficient)
            for experiment_dir in mlruns_path.iterdir():
                if experiment_dir.is_dir():
                    # Skip .trash and non-experiment directories
                    if experiment_dir.name == ".trash" or not experiment_dir.name.isdigit():
                        continue

                    # Search within experiment directory
                    potential_paths = [p for p in experiment_dir.glob(f"**/{run_id}") if p.is_dir()]
                    if potential_paths:
                        run_path = potential_paths[0]
                        self.logger.info(f"Found run directory at: {run_path}")
                        return str(run_path / "artifacts")

            # If direct search fails, try global recursive search with more inclusive pattern
            all_potential_paths = [
                p
                for p in mlruns_path.glob(f"**/{run_id}*")  # Add wildcard to catch variations
                if p.is_dir()
                and ".trash" not in str(p)
                and any(parent.name.isdigit() for parent in p.parents)
                and run_id in str(p)  # Ensure run_id is actually in the path
            ]
            self.logger.debug(f"Global search found {len(all_potential_paths)} potential paths")
            self.logger.debug(f"Searching in mlruns path: {mlruns_path}")
            self.logger.debug(f"All potential paths: {all_potential_paths}")
            if all_potential_paths:
                run_path = all_potential_paths[0]
                self.logger.info(f"Found run directory via global search: {run_path}")
                return str(run_path / "artifacts")

            # Check trash as last resort
            trash_paths = list(mlruns_path.glob(f".trash/**/{run_id}"))
            if trash_paths:
                self.logger.warning(f"Run {run_id} found only in trash: {trash_paths[0]}")

            # Debug info before raising error
            self.logger.debug(f"Searched in mlruns path: {mlruns_path}")
            self.logger.debug(
                f"Available experiment directories: {[d.name for d in mlruns_path.iterdir() if d.is_dir()]}"
            )
            raise FileNotFoundError(f"Run directory not found for ID: {run_id}")

        except Exception as e:
            self.logger.error(f"Error locating run directory: {str(e)}")
            raise FileNotFoundError(
                f"Failed to locate run directory for ID {run_id}: {str(e)}"
            ) from e


def create_experiment_run(experiment_name: str, experiment_id: str):
    """Decorator for creating MLflow runs"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            manager = MLFlowManager()
            experiment_id = manager.setup_experiment(experiment_name)
            with mlflow.start_run(experiment_id=experiment_id):
                mlflow.set_tracking_uri(manager.config.config["tracking_uri"])
                result = func(*args, **kwargs)
            return result

        return wrapper

    return decorator


def train_model(X, y):
    # Create and train a dummy model
    dummy_model = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
    dummy_model.fit(X, y)

    return dummy_model


# Usage example:
if __name__ == "__main__":
    manager = MLFlowManager()
    client = MlflowClient()

    try:
        manager.normalize_all_meta_yaml_paths()
        print("Successfully processed all meta.yaml files")
    except Exception as e:
        print(f"Error processing meta.yaml files: {e}")
