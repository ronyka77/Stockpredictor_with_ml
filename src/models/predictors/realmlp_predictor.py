from src.models.predictors.base_predictor import BasePredictor
from src.models.time_series.realmlp.realmlp_predictor import RealMLPPredictor as RealMLPInner
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RealMLPPredictor(BasePredictor):
    """
    Production wrapper for RealMLP predictions and export.
    """

    def __init__(self, run_id: str):
        super().__init__(run_id=run_id, model_type="realmlp")

    def load_model_from_mlflow(self) -> None:
        inner = RealMLPInner(model_name="RealMLP_inference", config={})
        ok = inner.load_model(self.run_id, experiment_name="realmlp_stock_predictor")
        if not ok:
            raise RuntimeError(f"Failed to load RealMLP model from MLflow run {self.run_id}")
        self.model = inner
        logger.info("✅ RealMLP model loaded from MLflow and ready for inference")

if __name__ == "__main__":
    run_id = "a052a3ad71ad4f858906534c69474832"
    days_back = 15

    predictor = RealMLPPredictor(run_id=run_id)
    output_file = predictor.run_prediction_pipeline(days_back=days_back)
    
    logger.info(f"Predictions saved to: {output_file}")


