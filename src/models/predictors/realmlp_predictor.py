from src.models.predictors.base_predictor import BasePredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RealMLPPredictor(BasePredictor):
    """
    Production wrapper for RealMLP predictions and export.
    """

    def __init__(self, run_id: str):
        super().__init__(run_id=run_id, model_type="realmlp")

    def load_model_from_mlflow(self) -> None:
        # Placeholder: will be connected when RealMLP save/load is wired into MLflow
        logger.info("RealMLP model loading from MLflow is not yet implemented.")


