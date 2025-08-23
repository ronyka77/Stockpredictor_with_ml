"""
MLflow integration utilities for the Model Context Protocol Server.

This module provides:
- MLflow experiment and run management
- Model logging and artifact tracking
- Integration with ExperimentLogger for structured logging
"""

import os
import shutil
from pathlib import Path
from typing import Any, Optional, Union

# MLflow imports
import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.tracking import MlflowClient

# Local imports
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)
project_root = Path(__file__).parent.parent.parent

class MLflowIntegration:
    """MLflow integration for experiment tracking and model management."""

    def __init__(self):
        """Initialize MLflow integration."""
        self.client = MlflowClient()
        self.mlruns_dir = project_root / "mlruns"
        self.mlruns_dir.mkdir(parents=True, exist_ok=True)

        # Set up MLflow tracking (use file:// URI for Windows compatibility)
        tracking_uri = self.mlruns_dir.absolute().as_uri()
        mlflow.set_tracking_uri(tracking_uri)

    def setup_experiment(
        self, experiment_name: str, artifact_location: Optional[str] = None) -> str:
        """Set up MLflow experiment.
        Args:
            experiment_name: Name of the experiment
            artifact_location: Optional custom artifact location
        Returns:
            Experiment ID
        """
        try:
            # Get or create experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                if artifact_location:
                    experiment_id = mlflow.create_experiment(
                        experiment_name, artifact_location=artifact_location
                    )
                else:
                    experiment_id = mlflow.create_experiment(experiment_name)

                mlflow.set_experiment(experiment_name)
                logger.info(
                    f"Created new experiment: {experiment_name}",
                    extra={"experiment_id": experiment_id},
                )
            else:
                experiment_id = experiment.experiment_id
                mlflow.set_experiment(experiment_name)
                logger.info(
                    f"Using existing experiment: {experiment_name}",
                    extra={"experiment_id": experiment_id},
                )
            return experiment_id

        except Exception as e:
            logger.error(f"Error setting up experiment: {str(e)}")
            raise

    def start_run(
        self,
        run_name: Optional[str] = None,
        experiment_id: Optional[str] = None,
        nested: bool = False,
        tags: Optional[dict[str, str]] = None,
    ) -> mlflow.ActiveRun:
        """Start a new MLflow run.

        Args:
            run_name: Optional name for the run
            experiment_id: Optional experiment ID
            nested: Whether this is a nested run
            tags: Optional tags for the run
        Returns:
            MLflow ActiveRun context
        """
        try:
            # Start run with specified parameters
            run = mlflow.start_run(
                run_name=run_name, experiment_id=experiment_id, nested=nested, tags=tags
            )

            logger.info(
                "Started MLflow run",
                extra={"run_id": run.info.run_id, "experiment_id": run.info.experiment_id},
            )
            return run

        except Exception as e:
            logger.error(f"Error starting run: {str(e)}")
            raise

    def log_params(self, params: dict[str, Any], run_id: Optional[str] = None) -> None:
        """Log parameters to MLflow.
        Args:
            params: Dictionary of parameters to log
            run_id: Optional run ID (uses active run if not specified)
        """
        try:
            if run_id:
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_params(params)
            else:
                mlflow.log_params(params)
        except Exception as e:
            logger.error(f"Error logging parameters: {str(e)}")
            raise

    def log_metrics(
        self, metrics: dict[str, float], step: Optional[int] = None, run_id: Optional[str] = None
    ) -> None:
        """Log metrics to MLflow.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
            run_id: Optional run ID (uses active run if not specified)
        """
        try:
            if run_id:
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_metrics(metrics, step=step)
            else:
                mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.error(f"Error logging metrics: {str(e)}")
            raise

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        flavor: str = "sklearn",
        signature: Optional[ModelSignature] = None,
        run_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Log a model to MLflow.
        Args:
            model: Model object to log
            artifact_path: Path for model artifact
            flavor: MLflow model flavor to use
            signature: Optional model signature
            run_id: Optional run ID (uses active run if not specified)
            **kwargs: Additional arguments for model logging
        """
        try:
            # Select appropriate logging function based on flavor
            if flavor == "sklearn":
                log_func = mlflow.sklearn.log_model
            elif flavor == "xgboost":
                log_func = mlflow.xgboost.log_model
            else:
                raise ValueError(f"Unsupported model flavor: {flavor}")

            # Log model with specified parameters
            if run_id:
                with mlflow.start_run(run_id=run_id):
                    if signature:
                        log_func(model, artifact_path, signature=signature, **kwargs)
                    else:
                        log_func(model, artifact_path, **kwargs)
            else:
                if signature:
                    log_func(model, artifact_path, signature=signature, **kwargs)
                else:
                    log_func(model, artifact_path, **kwargs)

            logger.info(
                f"Logged {flavor} model to MLflow",
                extra={"artifact_path": artifact_path, "has_signature": signature is not None},
            )

        except Exception as e:
            logger.error(f"Error logging model: {str(e)}")
            raise

    def log_artifact(
        self,
        local_path: Union[str, Path],
        artifact_path: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        """Log an artifact to MLflow.
        Args:
            local_path: Path to artifact file
            artifact_path: Optional path within artifact directory
            run_id: Optional run ID (uses active run if not specified)
        """
        try:
            if run_id:
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_artifact(local_path, artifact_path)
            else:
                mlflow.log_artifact(local_path, artifact_path)

            logger.info(
                "Logged artifact to MLflow",
                extra={"local_path": str(local_path), "artifact_path": artifact_path},
            )

        except Exception as e:
            logger.error(f"Error logging artifact: {str(e)}")
            raise

    def load_model(self, run_id: str, model_path: str = "model") -> Any:
        """Load a model from MLflow.
        Args:
            run_id: MLflow run ID
            model_path: Path to model within run artifacts
        Returns:
            Loaded model object
        """
        try:
            model_uri = f"runs:/{run_id}/{model_path}"
            model = mlflow.pyfunc.load_model(model_uri)

            logger.info(
                "Loaded model from MLflow", extra={"run_id": run_id, "model_path": model_path}
            )
            return model

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

def cleanup_deleted_runs(mlruns_dir="mlruns"):
    client = MlflowClient()
    experiments = client.search_experiments()
    for exp in experiments:
        logger.info(f"Processing Experiment: {exp.name} (ID: {exp.experiment_id})")
        try:
            # Search for runs with lifecycle_stage 'deleted'
            runs = client.search_runs(
                [exp.experiment_id], run_view_type=mlflow.entities.ViewType.DELETED_ONLY
            )
            logger.info(f"Found {len(runs)} runs in experiment {exp.name} (ID: {exp.experiment_id})")
            for run in runs:
                run_id = run.info.run_id
                run_path = os.path.join(mlruns_dir, exp.experiment_id, run_id)
                try:
                    if os.path.exists(run_path):
                        logger.info(f"Deleting run folder for run_id {run_id} at {run_path}")
                        shutil.rmtree(run_path)
                except Exception as e:
                    logger.error(f"Error deleting run folder for run_id {run_id} at {run_path}: {str(e)}")
                    continue
        except Exception as e:
            logger.error(f"Error processing experiment {exp.name} (ID: {exp.experiment_id}): {str(e)}")
            continue

def cleanup_empty_experiments(mlruns_dir="mlruns"):
    """Clean up experiments that have no runs.
    
    Args:
        mlruns_dir: Path to mlruns directory. Defaults to 'mlruns'.
    """
    client = MlflowClient()
    experiments = client.search_experiments()
    
    for exp in experiments:
        logger.info(f"Checking Experiment: {exp.name} (ID: {exp.experiment_id})")
        try:
            # Search for all runs (active and deleted)
            runs = client.search_runs(
                [exp.experiment_id],
                run_view_type=mlflow.entities.ViewType.ALL
            )
            
            if len(runs) == 0 and exp.experiment_id != "0":
                logger.info(f"Experiment {exp.name} has no runs - deleting...")
                exp_path = os.path.join(mlruns_dir, exp.experiment_id)
                
                # Delete from tracking server
                client.delete_experiment(exp.experiment_id)
                
                # Remove experiment directory if it exists
                if os.path.exists(exp_path):
                    logger.info(f"Removing experiment directory at {exp_path}")
                    shutil.rmtree(exp_path)
            else:
                logger.info(f"Experiment {exp.name} has {len(runs)} runs")
        except Exception as e:
            logger.error(f"Error processing experiment {exp.name} (ID: {exp.experiment_id}): {str(e)}")
            continue


if __name__ == "__main__":
    cleanup_deleted_runs()
    cleanup_empty_experiments()
