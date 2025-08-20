"""
XGBoost Model Implementation

This module provides a complete XGBoost model implementation for stock prediction
with hyperparameter optimization, early stopping, and MLflow integration.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import xgboost as xgb
import mlflow
import mlflow.xgboost
import tempfile
import optuna
from datetime import datetime

from src.models.base_model import BaseModel
from src.models.evaluation import ThresholdEvaluator, CustomMetrics
from src.utils.logger import get_logger
from src.utils.mlflow_utils import MLFlowManager
from src.data_utils.ml_data_pipeline import prepare_ml_data_for_training_with_cleaning

logger = get_logger(__name__)
experiment_name = "xgboost_stock_predictor"

def log_to_mlflow(model, metrics, params, experiment_name, X_eval):
    """
    Log trained model, metrics, and parameters to MLflow.
    Requires X_eval DataFrame for signature generation.
    Args:
        model: Trained XGBoost model
        metrics: Model evaluation metrics
        params: Model parameters
        experiment_name: Experiment name
        X_eval (pd.DataFrame): Evaluation features for signature generation
    Returns:
        str: Run ID
    """
    try:
        # Define pip requirements for the model
        pip_requirements = [
            "xgboost>=1.7.0",
            "pandas>=1.5.0",
            "numpy>=1.21.0",
            "scikit-learn>=1.0.0",
            "mlflow>=2.0.0"
        ]
        
        # End any existing run before starting a new one
        try:
            mlflow.end_run()
            logger.info("Ended existing MLflow run before starting new one")
        except Exception as e:
            logger.info(f"No existing run to end: {e}")
        
        # Set up MLflow tracking
        mlflow.set_experiment(experiment_name)

        # Start a new run
        with mlflow.start_run(
            run_name=f"xgboost_final_{datetime.now().strftime('%Y%m%d_%H%M')}"
        ) as run:
            # Log parameters (excluding base params if desired)
            params_to_log = {
                k: v
                for k, v in params.items()
                if k not in ["device", "objective", "verbosity", "seed", "nthread", "tree_method"]
            }
            mlflow.log_params(params_to_log)
            mlflow.log_metrics(metrics)

            # Create input example using the DataFrame X_eval
            input_example = X_eval.iloc[:5].copy()

            # Identify and convert integer columns to float64
            if hasattr(input_example, "dtypes"):
                for col in input_example.columns:
                    if input_example[col].dtype.kind == "i":
                        logger.info(f"Converting integer column '{col}' to float64 for signature")
                        input_example[col] = input_example[col].astype("float64")

            # Infer signature - create DMatrix for XGBoost prediction
            dmatrix_example = xgb.DMatrix(input_example)
            predictions_example = model.predict(dmatrix_example)
            signature = mlflow.models.infer_signature(input_example, predictions_example)

            # Update model registration with signature
            model_info = mlflow.xgboost.log_model(
                model,
                "model",
                pip_requirements=pip_requirements,
                registered_model_name=f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M')}",
                signature=signature,
            )

            logger.info(f"Model logged to MLflow: {model_info.model_uri}")
            logger.info(f"Run ID: {run.info.run_id}")
            return run.info.run_id

    except Exception as e:
        logger.error(f"Error logging to MLflow: {str(e)}")
        return None

class XGBoostModel(BaseModel):
    """
    XGBoost model for stock price prediction with advanced features
    """
    
    def __init__(self, 
                model_name: str = "xgboost_stock_predictor",
                config: Optional[Dict[str, Any]] = None,
                prediction_horizon: int = 10,
                threshold_evaluator: Optional[ThresholdEvaluator] = None):
        """
        Initialize XGBoost model
        
        Args:
            model_name: Name for MLflow tracking
            config: Model configuration parameters
            prediction_horizon: Prediction horizon in days
            threshold_evaluator: Optional shared ThresholdEvaluator instance
        """
        # Add prediction_horizon to config
        if config is None:
            config = {}
        config['prediction_horizon'] = prediction_horizon
        
        super().__init__(model_name, config, threshold_evaluator=threshold_evaluator)
        
        # Initialize XGBoost-specific parameters
        self.prediction_horizon = self.config.get('prediction_horizon', 10)
        self.model = None
        self.feature_names = None
        self.early_stopping_rounds = self.config.get('early_stopping_rounds', 50)
        self.eval_metric = self.config.get('eval_metric', 'rmse')
        self.base_threshold = 0.5
        self.default_confidence_method = 'leaf_depth'
        
        # Initialize central evaluators
        # Use the threshold_evaluator from the base model instead of creating a new one
        self.custom_metrics = CustomMetrics()
    
    def _create_model(self, params: Optional[Dict[str, Any]] = None, model_name: Optional[str] = None, **kwargs) -> 'XGBoostModel':
        """
        Create a new XGBoost model instance with specified parameters
        
        Args:
            params: Dictionary of XGBoost parameters
            model_name: Optional custom name for the new model
            **kwargs: Additional parameters
            
        Returns:
            New XGBoostModel instance configured with the provided parameters
        """
        if params is None:
            params = {}
            
        if model_name is None:
            model_name = f"{self.model_name}_configured"
        
        # Create new model instance with the same prediction horizon and evaluator settings
        config = params.copy()
        config.update({
            'investment_amount': self.threshold_evaluator.investment_amount
        })
        
        new_model = XGBoostModel(
            model_name=model_name,
            config=config,
            prediction_horizon=self.prediction_horizon,
            threshold_evaluator=self.threshold_evaluator
        )
        
        # Copy any relevant settings from the current model
        new_model.early_stopping_rounds = self.early_stopping_rounds
        new_model.eval_metric = self.eval_metric
        
        logger.info(f"Created new XGBoost model '{model_name}' with parameters: {params}")
        
        return new_model
    
    def _prepare_data(self, X_train: pd.DataFrame, y_train: pd.Series, 
                        X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[xgb.DMatrix, xgb.DMatrix]:
        """
        Prepare data for XGBoost training using pre-split train/test data
        
        Args:
            X_train: Training feature matrix
            y_train: Training target values
            X_test: Test feature matrix (used for validation during training)
            y_test: Test target values (used for validation during training)
            
        Returns:
            Training and test DMatrix objects
        """
        # Store feature names for later use
        self.feature_names = list(X_train.columns)
        
        # Create DMatrix objects directly from the pre-split data
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=self.feature_names)
        
        return dtrain, dtest
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_test: Optional[pd.DataFrame] = None, y_test: Optional[pd.Series] = None,
            params: Optional[Dict[str, Any]] = None,
            validation_split: Optional[float] = None) -> 'XGBoostModel':
        """
        Train the XGBoost model using pre-split train/test data
        
        Args:
            X_train: Training feature matrix
            y_train: Training target values
            X_test: Test/validation feature matrix (optional if validation_split provided)
            y_test: Test/validation target values (optional if validation_split provided)
            params: Custom parameters (overrides defaults)
            validation_split: Optional fraction (0,1) to split from provided X_train/y_train for validation
            
        Returns:
            Self for method chaining
        """
        # logger.info(f"Training XGBoost model with {len(X_train)} samples, {len(X_train.columns)} features")
        
        # Handle optional validation split when X_test/y_test not provided
        if (X_test is None or y_test is None) and validation_split is not None:
            if not (0.0 < validation_split < 1.0):
                raise ValueError("validation_split must be in (0,1)")
            n_samples = len(X_train)
            val_size = max(1, int(n_samples * validation_split))
            split_idx = n_samples - val_size
            if split_idx <= 0:
                split_idx = n_samples - 1
            X_test = X_train.iloc[split_idx:].copy()
            y_test = y_train.iloc[split_idx:].copy()
            X_train = X_train.iloc[:split_idx].copy()
            y_train = y_train.iloc[:split_idx].copy()

        if X_test is None or y_test is None:
            # If still None (no split requested), create a tiny holdout from tail to satisfy API
            n_samples = len(X_train)
            val_size = max(1, int(n_samples * 0.2))
            split_idx = n_samples - val_size
            X_test = X_train.iloc[split_idx:].copy()
            y_test = y_train.iloc[split_idx:].copy()
            X_train = X_train.iloc[:split_idx].copy()

        # Prepare data
        dtrain, dtest = self._prepare_data(X_train, y_train, X_test, y_test)
        
        # Use provided parameters or defaults
        if params is None:
            params = self.hyperparameter_config.get_default_params()
        
        # Extract n_estimators before adding to XGBoost params (to avoid warning)
        n_estimators = params.pop('n_estimators', 1000)
        
        # Add objective and evaluation metric (don't override random_state if provided)
        base_params = {
            'objective': 'reg:quantileerror',
            'quantile_alpha': 0.8,
            'eval_metric': self.eval_metric
        }
        # Only add random_state if not already provided in params
        if 'random_state' not in params:
            base_params['random_state'] = 42
        
        params.update(base_params)
        
        # Set up evaluation
        evallist = [(dtrain, 'train'), (dtest, 'test')]
        
        # Train model with early stopping
        self.model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=n_estimators,
            evals=evallist,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False
        )
        
        # Log training completion
        best_iteration = self.model.best_iteration
        best_score = self.model.best_score
        
        logger.info(f"Training completed. Best iteration: {best_iteration}, Best score: {best_score:.4f}")
        
        # Log parameters to MLflow
        self.log_params(params)
        self.log_metrics({
            "best_iteration": best_iteration,
            "best_score": best_score
        })
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before making predictions")
        
        # Ensure feature names match
        if self.feature_names and list(X.columns) != self.feature_names:
            logger.warning("Feature names don't match training data")
        
        # Create DMatrix for prediction
        dtest = xgb.DMatrix(X, feature_names=list(X.columns))
        
        # Make predictions
        predictions = self.model.predict(dtest)
        
        logger.info(f"Generated {len(predictions)} predictions")
        return predictions
    
    def get_feature_importance(self, importance_type: str = 'weight') -> pd.DataFrame:
        """
        Get feature importance from the trained model
        
        Args:
            importance_type: Type of importance ('weight', 'gain', 'cover')
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before getting feature importance")
        
        # Get importance scores
        importance_dict = self.model.get_score(importance_type=importance_type)
        
        # Convert to DataFrame
        importance_df = pd.DataFrame([
            {'feature': feature, 'importance': importance}
            for feature, importance in importance_dict.items()
        ]).sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information
        
        Returns:
            Dictionary with model details
        """
        info = super().get_model_info()
        
        if self.model is not None:
            info.update({
                'n_features': self.model.num_features(),
                'best_iteration': self.model.best_iteration,
                'best_score': self.model.best_score,
                'early_stopping_rounds': self.early_stopping_rounds,
                'eval_metric': self.eval_metric
            })
        
        return info
    
    def save_model(self, metrics: Dict[str, float], params: Dict[str, Any], 
                    X_eval: pd.DataFrame, experiment_name: str = None) -> str:
        """
        Save the trained XGBoost model to MLflow using the universal logging function
        
        Args:
            metrics: Model evaluation metrics to log
            params: Model parameters to log
            X_eval: Evaluation features for signature generation
            experiment_name: Experiment name (uses default if None)
            
        Returns:
            MLflow run ID where the model was saved
        """
        if self.model is None:
            raise RuntimeError("No trained model to save")
        
        # Use the universal logging function via the class method
        run_id = self.log_model_to_mlflow(
            metrics=metrics,
            params=params,
            X_eval=X_eval,
            experiment_name=experiment_name
        )
        
        logger.info(f"✅ XGBoost model saved using universal logging function. Run ID: {run_id}")
        return run_id
    
    def load_model(self, run_id: str) -> None:
        """
        Load a saved XGBoost model from MLflow using run ID
        
        Args:
            run_id: MLflow run ID to load model from
        """
        try:
            logger.info(f"Loading XGBoost model from MLflow run: {run_id}")
            
            # Get the model artifact path from the run
            client = mlflow.tracking.MlflowClient()
            run_info = client.get_run(run_id)
            
            # List artifacts to find the model
            artifacts = client.list_artifacts(run_id)
            model_artifact_path = None
            
            # First, try to find the exact "model" directory (most common case)
            for artifact in artifacts:
                if artifact.is_dir and artifact.path == "model":
                    model_artifact_path = artifact.path
                    break
            
            # If not found, look for any directory with 'model' in the name
            if model_artifact_path is None:
                for artifact in artifacts:
                    if artifact.is_dir and 'model' in artifact.path.lower():
                        model_artifact_path = artifact.path
                        break
            
            # Default to "model" (the standard MLflow artifact path)
            if model_artifact_path is None:
                model_artifact_path = "model"
                logger.warning("⚠️ No model artifact found, defaulting to 'model' path")
            
            # Log available artifacts for debugging
            logger.info(f"Available artifacts in run {run_id}:")
            for artifact in artifacts:
                logger.info(f"  - {artifact.path} ({'dir' if artifact.is_dir else 'file'})")
            
            # Construct model URI and load
            model_uri = f"runs:/{run_id}/{model_artifact_path}"
            logger.info(f"Loading model from URI: {model_uri}")
            
            # Load the model
            loaded_model = mlflow.xgboost.load_model(model_uri)
            self.model = loaded_model
            
            # Load additional metadata from run
            self._load_metadata_from_run(run_info)
            
            logger.info(f"✅ XGBoost model loaded successfully from MLflow run: {run_id}")
                
        except Exception as e:
            logger.error(f"❌ Error loading XGBoost model from MLflow run {run_id}: {str(e)}")
            raise
    
    def _load_metadata_from_run(self, run_info) -> None:
        """
        Load model metadata from MLflow run info
        
        Args:
            run_info: MLflow run info object
        """
        try:
            params = run_info.data.params
            
            # Restore model configuration
            self.model_name = params.get('model_model_name', self.model_name)
            self.prediction_horizon = int(params.get('model_prediction_horizon', self.prediction_horizon))
            self.early_stopping_rounds = int(params.get('model_early_stopping_rounds', self.early_stopping_rounds))
            self.eval_metric = params.get('model_eval_metric', self.eval_metric)
            
            # Extract feature names from MLflow model signature
            try:
                client = mlflow.tracking.MlflowClient()
                
                # Direct approach: get signature from model URI
                model_uri = f"runs:/{run_info.info.run_id}/model"
                logger.info(f"Attempting to load model signature from: {model_uri}")
                
                # Load model info to get signature
                from mlflow.models import get_model_info
                model_info = get_model_info(model_uri)
                logger.info(f"Model info loaded: {model_info is not None}")
                
                if model_info and model_info.signature:
                    logger.info(f"Model signature found: {model_info.signature is not None}")
                    
                    if model_info.signature.inputs:
                        logger.info(f"Signature inputs found: {len(model_info.signature.inputs.inputs) if hasattr(model_info.signature.inputs, 'inputs') else 'No inputs attr'}")
                        
                        # Extract feature names from signature inputs
                        feature_names = []
                        
                        # Handle different signature input formats
                        if hasattr(model_info.signature.inputs, 'inputs'):
                            # Schema format
                            for input_spec in model_info.signature.inputs.inputs:
                                if hasattr(input_spec, 'name') and input_spec.name:
                                    feature_names.append(input_spec.name)
                        elif hasattr(model_info.signature.inputs, 'schema'):
                            # Alternative schema format
                            if hasattr(model_info.signature.inputs.schema, 'input_names'):
                                feature_names = model_info.signature.inputs.schema.input_names
                        else:
                            logger.info(f"Signature inputs type: {type(model_info.signature.inputs)}")
                            logger.info(f"Signature inputs attributes: {dir(model_info.signature.inputs)}")
                        
                        if feature_names:
                            self.feature_names = feature_names
                            logger.info(f"✅ Loaded {len(self.feature_names)} feature names from MLflow signature")
                        else:
                            logger.warning("⚠️ No feature names found in model signature inputs")
                    else:
                        logger.warning("⚠️ Model signature has no inputs")
                else:
                    logger.warning("⚠️ No model signature found in model info")
                        
            except Exception as signature_error:
                logger.warning(f"⚠️ Could not extract feature names from signature: {str(signature_error)}")
                logger.info(f"Signature error details: {type(signature_error).__name__}: {signature_error}")
                
                # Fallback: try to load from artifacts (original method)
                try:
                    artifacts = client.list_artifacts(run_info.info.run_id)
                    
                    # Look for feature names artifact
                    for artifact in artifacts:
                        if 'feature_names.txt' in artifact.path:
                            with tempfile.TemporaryDirectory() as temp_dir:
                                feature_file = client.download_artifacts(
                                    run_info.info.run_id, 
                                    artifact.path, 
                                    temp_dir
                                )
                                with open(feature_file, 'r') as f:
                                    self.feature_names = [line.strip() for line in f.readlines()]
                                logger.info(f"✅ Loaded {len(self.feature_names)} feature names from artifact file")
                                break
                except Exception as artifact_error:
                    logger.info(f"Could not load feature names from artifacts: {str(artifact_error)}")
                
            logger.info("Model metadata loaded successfully")
            
        except Exception as e:
            logger.warning(f"⚠ Could not load all metadata: {str(e)}")
    
    @classmethod
    def load_from_mlflow(cls, run_id: str) -> 'XGBoostModel':
        """
        Class method to create a new XGBoostModel instance and load from MLflow
        
        Args:
            run_id: MLflow run ID to load model from
            
        Returns:
            New XGBoostModel instance with loaded model
        """
        # Create new instance
        xgb_model = cls()
        
        # Load model from MLflow
        xgb_model.load_model(run_id=run_id)
        
        return xgb_model
    
    def get_prediction_confidence(self, X: pd.DataFrame, method: str = 'leaf_depth') -> np.ndarray:
        """
        Calculate confidence scores for predictions using various methods
        
        Args:
            X: Feature matrix
            method: Confidence calculation method ('leaf_depth', 'margin', 'variance')
            
        Returns:
            Array of confidence scores (higher = more confident)
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before calculating confidence")
        
        # Create DMatrix for prediction
        dtest = xgb.DMatrix(X, feature_names=list(X.columns))
        
        if method == 'leaf_depth':
            # Use leaf indices as confidence proxy (deeper leaves = more specific predictions)
            leaf_indices = self.model.predict(dtest, pred_leaf=True)
            # Calculate average leaf depth across all trees
            confidence_scores = np.mean(leaf_indices, axis=1)
            confidence_scores = np.power(confidence_scores, 2)
            
        elif method == 'margin':
            # Use prediction margin (distance from decision boundary)
            margins = self.model.predict(dtest, output_margin=True)
            confidence_scores = np.abs(margins)
            
        elif method == 'variance':
            # Use prediction variance across trees (requires ntree_limit)
            predictions = []
            n_trees = self.model.best_iteration or 100
            
            # Get predictions from different subsets of trees
            for i in range(10, min(n_trees, 100), 10):
                pred = self.model.predict(dtest, ntree_limit=i)
                predictions.append(pred)
            
            # Calculate variance across different tree counts
            predictions_array = np.array(predictions)
            variance = np.var(predictions_array, axis=0)
            confidence_scores = 1.0 / (1.0 + variance)  # Inverse variance as confidence
            
        elif method == 'simple':
            # Simple confidence based on prediction distance from current price
            predictions = self.model.predict(dtest)
            
            # Extract current prices from features (assuming 'close' column exists)
            current_prices = X['close'].values if 'close' in X.columns else np.ones_like(predictions)
            
            # Confidence = prediction change magnitude (normalized)
            relative_change = np.abs(predictions - current_prices) / current_prices
            confidence_scores = np.clip(relative_change, 0, 1)  # Clip to [0, 1]
            
        else:
            raise ValueError(f"Unknown confidence method: {method}")
        
        # Normalize confidence scores to [0, 1] range
        min_conf, max_conf = confidence_scores.min(), confidence_scores.max()
        if max_conf > min_conf:
            confidence_scores = (confidence_scores - min_conf) / (max_conf - min_conf)
        else:
            # All confidence scores are identical - this is problematic
            logger.warning(f"All confidence scores are identical ({min_conf:.4f}) - using uniform distribution")
            confidence_scores = np.full_like(confidence_scores, 0.5)
            
        logger.info(f"Final confidence - Range: [{confidence_scores.min():.4f}, {confidence_scores.max():.4f}]")
        logger.info(f"Final confidence - Mean: {confidence_scores.mean():.4f}, std: {confidence_scores.std():.4f}")
        
        return confidence_scores
    
    def objective(self, X_train: pd.DataFrame, y_train: pd.Series, 
                    X_test: pd.DataFrame, y_test: pd.Series, objective_column: str = 'test_profit_per_investment') -> callable:
        """
        Create Optuna objective function for hyperparameter optimization with threshold optimization
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features  
            y_test: Test targets
            
        Returns:
            Objective function for Optuna optimization with threshold optimization
        """
        # Initialize tracking variables for best model (optimizing for investment success rate)
        self.best_investment_success_rate = -np.inf
        self.best_trial_model = None
        self.best_trial_params = None
        self.best_threshold_info = None
        
        def objective(trial):
            """Objective function for Optuna optimization with threshold optimization for each trial"""
            # First suggest tree method to determine compatible parameters
            tree_method = trial.suggest_categorical('tree_method', ['auto', 'exact', 'approx', 'hist'])
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'gamma': trial.suggest_float('gamma', 0.0, 10.0),
                'tree_method': tree_method,
            }
            
            # Add colsample_bynode only for compatible tree methods (not 'exact')
            if tree_method != 'exact':
                params['colsample_bynode'] = trial.suggest_float('colsample_bynode', 0.4, 1.0)  # Only for non-exact methods
            
            try:
                # Create model with trial parameters using the _create_model method
                trial_model = self._create_model(
                    params=params,
                    model_name=f"xgboost_trial_{trial.number}"
                )
                
                # Disable MLflow for trial models to avoid clutter
                trial_model.disable_mlflow = True
                
                trial_model.fit(X_train, y_train, X_test, y_test, params=params)
                
                # Extract current prices for test sets
                test_current_prices = X_test['close'].values
                
                # Run threshold optimization for this trial
                logger.info(f"Running threshold optimization for trial {trial.number}")
                
                threshold_results = trial_model.optimize_prediction_threshold(
                    X_test=X_test,
                    y_test=y_test,
                    current_prices_test=test_current_prices,
                    confidence_method='leaf_depth',
                    threshold_range=(0.01, 0.99),
                    n_thresholds=90
                )
                
                if threshold_results['status'] == 'success':
                    # Use threshold-optimized investment success rate
                    optimized_profit_score = threshold_results['best_result'][objective_column]
                    
                    # Store additional threshold info for logging (use values from threshold optimization response)
                    threshold_info = {
                        'optimal_threshold': threshold_results['optimal_threshold'],
                        'samples_kept_ratio': threshold_results['best_result']['test_samples_ratio'],
                        'investment_success_rate': threshold_results['best_result']['investment_success_rate'],
                        'custom_accuracy': threshold_results['best_result']['test_custom_accuracy'],
                        'total_threshold_profit': threshold_results['best_result']['test_profit'],
                        'profitable_investments': threshold_results['best_result']['profitable_investments']
                    }
                    
                    # Log threshold optimization results for this trial
                    logger.info(f"Trial {trial.number} threshold optimization:")
                    logger.info(f"  Optimal threshold: {threshold_info['optimal_threshold']:.3f}")
                    logger.info(f"  Samples kept: {threshold_info['samples_kept_ratio']:.1%}")
                    logger.info(f"  Optimized {objective_column}: {optimized_profit_score:.3f}")
                
                # Check if this is the best trial so far
                if optimized_profit_score > self.best_investment_success_rate:
                    self.best_investment_success_rate = optimized_profit_score
                    self.best_trial_model = trial_model
                    self.best_trial_params = params.copy()
                    self.best_threshold_info = threshold_info.copy()
                    
                    # Update self.model with the best trial model
                    self.model = trial_model.model
                    self.feature_names = trial_model.feature_names
                    
                    # Store the optimal threshold information
                    if threshold_info['optimal_threshold'] is not None:
                        self.optimal_threshold = threshold_info['optimal_threshold']
                        self.confidence_method = 'leaf_depth'
                    
                    logger.info(f"🎯 NEW BEST TRIAL {trial.number}: {objective_column} = {optimized_profit_score:.3f}")
                    if threshold_info['optimal_threshold'] is not None:
                        logger.info(f"   Optimal threshold: {threshold_info['optimal_threshold']:.3f}")
                        logger.info(f"   Samples kept: {threshold_info['samples_kept_ratio']:.1%}")
                        logger.info(f"   Custom accuracy: {threshold_info['custom_accuracy']:.3f}")
                    
                    self.previous_best = optimized_profit_score
                else:
                    logger.info(f"Trial {trial.number}: Investment Success Rate = {optimized_profit_score:.3f} (Best: {self.best_investment_success_rate:.3f})")
                
                return optimized_profit_score
                
            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {e}")
                return -1e6
        
        return objective
    
    def get_best_trial_info(self) -> Dict[str, Any]:
        """
        Get information about the best trial from hyperparameter optimization with threshold info
        
        Returns:
            Dictionary with best trial information including threshold optimization details
        """
        if not hasattr(self, 'best_investment_success_rate'):
            return {"message": "No hyperparameter optimization has been run yet"}
        
        base_info = {
            "best_investment_success_rate": self.best_investment_success_rate,
            "best_trial_params": self.best_trial_params,
            "has_best_model": self.best_trial_model is not None,
            "model_updated": self.model is not None
        }
        
        # Add threshold optimization information if available
        if hasattr(self, 'best_threshold_info') and self.best_threshold_info is not None:
            base_info.update({
                "threshold_optimization": {
                    "optimal_threshold": self.best_threshold_info.get('optimal_threshold'),
                    "samples_kept_ratio": self.best_threshold_info.get('samples_kept_ratio'),
                    "investment_success_rate": self.best_threshold_info.get('investment_success_rate'),
                    "custom_accuracy": self.best_threshold_info.get('custom_accuracy'),
                    "total_threshold_profit": self.best_threshold_info.get('total_threshold_profit'),
                    "profitable_investments": self.best_threshold_info.get('profitable_investments'),
                    "confidence_method": getattr(self, 'confidence_method', 'leaf_depth')
                }
            })
        else:
            base_info["threshold_optimization"] = None
        
        return base_info
    
    def finalize_best_model(self) -> None:
        """
        Finalize the best model after hyperparameter optimization with threshold optimization
        This ensures the main model instance contains the best performing model and threshold info
        """
        if hasattr(self, 'best_trial_model') and self.best_trial_model is not None:
            # Copy the best model's state to this instance
            self.model = self.best_trial_model.model
            self.feature_names = self.best_trial_model.feature_names
            
            # Copy threshold optimization information if available
            if hasattr(self, 'best_threshold_info') and self.best_threshold_info is not None:
                if self.best_threshold_info.get('optimal_threshold') is not None:
                    self.optimal_threshold = self.best_threshold_info['optimal_threshold']
                    self.confidence_method = getattr(self, 'confidence_method', 'leaf_depth')
            
            # Log the finalization
            logger.info(f"✅ Best model finalized with investment success rate: {self.best_investment_success_rate:.3f}")
            logger.info(f"✅ Best parameters: {self.best_trial_params}")
            
            # Log threshold information if available
            if hasattr(self, 'best_threshold_info') and self.best_threshold_info is not None:
                threshold_info = self.best_threshold_info
                if threshold_info.get('optimal_threshold') is not None:
                    logger.info(f"✅ Optimal threshold: {threshold_info['optimal_threshold']:.3f}")
                    logger.info(f"✅ Samples kept ratio: {threshold_info['samples_kept_ratio']:.1%}")
                    logger.info(f"✅ Investment success rate: {threshold_info['investment_success_rate']:.3f}")
                    logger.info(f"✅ Custom accuracy: {threshold_info['custom_accuracy']:.3f}")
                    logger.info(f"✅ Profitable investments: {threshold_info['profitable_investments']}")
                else:
                    logger.info("✅ No threshold optimization was successful for the best trial")
            
            # Log to MLflow if enabled
            if not getattr(self, 'disable_mlflow', False):
                # Log best hyperparameters
                self.log_params({f"best_{k}": v for k, v in self.best_trial_params.items()})
                
                # Log best metrics
                metrics_to_log = {
                    "best_investment_success_rate": self.best_investment_success_rate,
                    "hypertuning_completed": 1
                }
                
                # Add threshold metrics if available
                if hasattr(self, 'best_threshold_info') and self.best_threshold_info is not None:
                    threshold_info = self.best_threshold_info
                    if threshold_info.get('optimal_threshold') is not None:
                        metrics_to_log.update({
                            "best_optimal_threshold": threshold_info['optimal_threshold'],
                            "best_samples_kept_ratio": threshold_info['samples_kept_ratio'],
                            "best_investment_success_rate": threshold_info['investment_success_rate'],
                            "best_custom_accuracy": threshold_info['custom_accuracy'],
                            "best_total_threshold_profit": threshold_info['total_threshold_profit'],
                            "best_profitable_investments": threshold_info['profitable_investments']
                        })
                
                self.log_metrics(metrics_to_log)
        else:
            logger.warning("⚠ No best model found to finalize")
    
    def log_model_to_mlflow(self, metrics: Dict[str, float], params: Dict[str, Any], 
                            X_eval: pd.DataFrame, experiment_name: str = None) -> str:
        """
        Log the trained XGBoost model to MLflow using the universal logging function
        
        Args:
            metrics: Model evaluation metrics
            params: Model parameters
            X_eval: Evaluation features for signature generation
            experiment_name: Experiment name (uses default if None)
            
        Returns:
            MLflow run ID
        """
        if self.model is None:
            raise RuntimeError("No trained model to log")
        
        if experiment_name is None:
            experiment_name = f"{self.model_name}_experiment"
        
        # Use the universal logging function
        run_id = log_to_mlflow(
            model=self.model,
            metrics=metrics,
            params=params,
            experiment_name=experiment_name,
            X_eval=X_eval
        )
        
        logger.info(f"✅ Model logged to MLflow using universal function. Run ID: {run_id}")
        return run_id

def main():
    """
    Standalone XGBoost hypertuning and evaluation using load_all_data
    """
    logger.info("=" * 80)
    logger.info("🎯 STANDALONE XGBOOST HYPERTUNING & EVALUATION")
    logger.info("=" * 80)
    
    try:
        # 0. Setup MLflow experiment tracking
        logger.info("0. Setting up MLflow experiment tracking...")
        mlflow_manager = MLFlowManager()
        experiment_id = mlflow_manager.setup_experiment(experiment_name)
        
        logger.info(f"✅ MLflow experiment setup completed: {experiment_id}")
        logger.info(f"✅ MLflow tracking URI: {mlflow.get_tracking_uri()}")
        
        # Define prediction horizon
        prediction_horizon = 10
        number_of_trials = 20
        objective_column = 'test_profit_per_investment'
        
        # OPTION 1: Use the enhanced data preparation function with cleaning (direct import)
        data_result = prepare_ml_data_for_training_with_cleaning(
            prediction_horizon=prediction_horizon,
            split_date='2025-02-01',
            ticker=None,
            clean_features=True,
        )
        
        # Extract prepared data
        X_train = data_result['X_train']
        X_test = data_result['X_test']
        y_train = data_result['y_train']
        y_test = data_result['y_test']
        target_column = data_result['target_column']
        train_date_range = data_result['train_date_range']
        test_date_range = data_result['test_date_range']
        
        logger.info(f"✅ Data preparation completed using target: {target_column}")
        
        # 2. Initialize XGBoost model
        logger.info("2. Initializing XGBoost model...")
        
        xgb_model = XGBoostModel(
            model_name="xgboost_standalone_hypertuned",
            prediction_horizon=prediction_horizon
        )
        
        # 3. Hyperparameter optimization
        logger.info("3. Starting hyperparameter optimization...")
        
        # Create objective function using the XGBoost model class method
        objective_function = xgb_model.objective(X_train, y_train, X_test, y_test, objective_column)
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_function, n_trials=number_of_trials, n_jobs=1)
        
        # Get best results from study
        best_params = study.best_params
        best_profit = study.best_value  # This is now threshold-optimized profit per investment
        
        logger.info("🎯 Hyperparameter optimization with threshold optimization completed!")
        logger.info(f"✅ Best Threshold-Optimized Profit per Investment: ${best_profit:.2f}")
        logger.info(f"✅ Best parameters: {best_params}")
        
        # Finalize the best model (ensure xgb_model contains the best performing model)
        xgb_model.finalize_best_model()
        
        # Get best trial info for verification (now includes threshold info)
        best_trial_info = xgb_model.get_best_trial_info()
        logger.info(f"✅ Best trial info: {best_trial_info}")
        
        # 4. Use the automatically selected best model for final evaluation
        logger.info("4. Making final evaluation with automatically selected best model...")
        
        # xgb_model now contains the best model with optimal threshold, no need to create a new one
        final_model = xgb_model
        
        # Check if the best model has threshold optimization results
        has_threshold_optimization = (hasattr(final_model, 'best_threshold_info') and 
                                    final_model.best_threshold_info is not None and
                                    final_model.best_threshold_info.get('optimal_threshold') is not None)
        
        if has_threshold_optimization:
            logger.info("✅ Best model includes threshold optimization results")
            
            # Extract current prices for evaluation
            if 'close' in X_test.columns:
                final_current_prices = X_test['close'].values
            elif 'Close' in X_test.columns:
                final_current_prices = X_test['Close'].values
            else:
                final_current_prices = y_test.values * 0.95  # Fallback
            
            # Evaluate with the optimal threshold from hyperparameter optimization
            # Use stored optimal values from hyperparameter optimization
            optimal_threshold = getattr(final_model, 'optimal_threshold', 0.5)
            confidence_method = getattr(final_model, 'confidence_method', 'leaf_depth')
            
            threshold_performance = final_model.threshold_evaluator.evaluate_threshold_performance(
                model=final_model,
                X_test=X_test,
                y_test=y_test,
                current_prices_test=final_current_prices,
                threshold=optimal_threshold,
                confidence_method=confidence_method
            )
            
            # Also get unfiltered baseline for comparison
            baseline_predictions = final_model.predict(X_test)
            baseline_profit = final_model.threshold_evaluator.calculate_profit_score(
                y_test.values, baseline_predictions, final_current_prices
            )
            baseline_profit_per_investment = baseline_profit / len(y_test)
            
            logger.info("📊 Final Results Comparison:")
            logger.info(f"   Baseline (unfiltered) profit per investment: ${baseline_profit_per_investment:.2f}")
            logger.info(f"   Threshold-optimized profit per investment: ${threshold_performance['profit_per_investment']:.2f}")
            logger.info(f"   Improvement ratio: {threshold_performance['profit_per_investment'] / baseline_profit_per_investment if baseline_profit_per_investment != 0 else 0:.2f}x")
            logger.info(f"   Samples kept: {threshold_performance['samples_evaluated']}/{len(X_test)} ({threshold_performance['samples_kept_ratio']:.1%})")
            logger.info(f"   Investment success rate: {threshold_performance['investment_success_rate']:.3f}")
            
            # Use threshold-optimized metrics for final evaluation
            final_profit_per_investment = threshold_performance['profit_per_investment']
            final_total_profit = threshold_performance['total_profit']
            final_investment_success_rate = threshold_performance['investment_success_rate']
            final_samples_kept = threshold_performance['samples_evaluated']
            
            # Traditional metrics on filtered data
            final_mse = threshold_performance['mse']
            final_mae = threshold_performance['mae']
            final_r2 = threshold_performance['r2_score']
            
            # Store threshold results for MLflow logging
            threshold_metrics = {
                'final_optimal_threshold': final_model.best_threshold_info['optimal_threshold'],
                'final_samples_kept_ratio': threshold_performance['samples_kept_ratio'],
                'final_investment_success_rate': final_investment_success_rate,
                'final_baseline_profit_per_investment': baseline_profit_per_investment,
                'final_improvement_ratio': threshold_performance['profit_per_investment'] / baseline_profit_per_investment if baseline_profit_per_investment != 0 else 0
            }
        else:
            logger.info("⚠️ Best model does not have threshold optimization results, using standard evaluation")
            
            # Standard evaluation without threshold optimization
            final_predictions = final_model.predict(X_test)
            
            # Extract current prices for final profit calculation
            if 'close' in X_test.columns:
                final_current_prices = X_test['close'].values
            elif 'Close' in X_test.columns:
                final_current_prices = X_test['Close'].values
            else:
                final_current_prices = y_test.values * 0.95  # Fallback
            
            # Calculate standard metrics
            final_total_profit = final_model.threshold_evaluator.calculate_profit_score(y_test.values, final_predictions, final_current_prices)
            final_profit_per_investment = final_total_profit / len(y_test)
            final_samples_kept = len(y_test)
            
            # Traditional metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            final_mse = mean_squared_error(y_test, final_predictions)
            final_mae = mean_absolute_error(y_test, final_predictions)
            final_r2 = r2_score(y_test, final_predictions)
            
            threshold_metrics = {}
        
        logger.info("📊 Final Optimized Results:")
        logger.info(f"   Total Profit: ${final_total_profit:.2f}")
        logger.info(f"   Profit per Investment: ${final_profit_per_investment:.2f}")
        if has_threshold_optimization:
            logger.info(f"   Samples Used: {final_samples_kept}/{len(y_test)} (threshold-filtered)")
        else:
            logger.info(f"   Samples Used: {final_samples_kept}/{len(y_test)} (all samples)")
        logger.info(f"   Traditional MSE: {final_mse:.4f}")
        logger.info(f"   Traditional MAE: {final_mae:.4f}")
        logger.info(f"   Traditional R²: {final_r2:.4f}")
        
        # 5. Feature importance analysis
        logger.info("5. Analyzing feature importance...")
        
        feature_importance = final_model.get_feature_importance('gain')
        
        logger.info("📊 Top 30 Most Important Features:")
        for idx, row in feature_importance.head(30).iterrows():
            logger.info(f"   {row['feature']}: {row['importance']:.2f}")
        
        # 6. Log model to MLflow using universal function
        logger.info("6. Logging model to MLflow using universal function...")
        
        # Prepare comprehensive metrics for logging
        final_metrics = {
            "final_total_profit": final_total_profit,
            "final_profit_per_investment": final_profit_per_investment,
            "final_mse": final_mse,
            "final_mae": final_mae,
            "final_r2": final_r2,
            "final_test_samples": len(y_test),
            "hypertuning_best_profit": best_profit,
            "hypertuning_trials_completed": number_of_trials
        }
        
        # Add threshold optimization metrics if successful
        final_metrics.update(threshold_metrics)
        
        # Add top feature importance to metrics
        top_features = feature_importance.head(10)
        for i, (_, row) in enumerate(top_features.iterrows()):
            final_metrics[f"feature_importance_{i+1}_{row['feature']}"] = row['importance']
        
        # Prepare comprehensive parameters for logging
        final_params = best_params.copy()
        final_params.update({
            "prediction_horizon": prediction_horizon,
            "early_stopping_rounds": final_model.early_stopping_rounds,
            "eval_metric": final_model.eval_metric,
            "hypertuning_trials": number_of_trials,
            "hypertuning_direction": "maximize",
            "hypertuning_metric": "profit_score",
            "target_column": target_column,
            "split_date": data_result['split_date'],
            "feature_count": data_result['feature_count'],
            "train_samples": data_result['train_samples'],
            "test_samples": data_result['test_samples'],
            "threshold_optimization_enabled": has_threshold_optimization
        })
        
        # Add threshold parameters if optimization was successful during hyperparameter optimization
        if has_threshold_optimization:
            final_params.update({
                "threshold_method": final_model.confidence_method,
                "threshold_optimization_during_hypertuning": True,
                "optimal_threshold_from_hypertuning": final_model.best_threshold_info['optimal_threshold']
            })
        
        # Use the universal logging function via save_model method
        saved_run_id = final_model.save_model(
            metrics=final_metrics,
            params=final_params,
            X_eval=X_test,
            experiment_name=experiment_name
        )
        
        logger.info("=" * 80)
        logger.info("🎉 STANDALONE XGBOOST HYPERTUNING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"📊 Dataset: {data_result['train_samples'] + data_result['test_samples']:,} samples, {data_result['feature_count']} features")
        logger.info(f"🎯 Target: {target_column} ({prediction_horizon}-day horizon)")
        logger.info(f"📅 Train period: {train_date_range}")
        logger.info(f"📅 Test period: {test_date_range}")
        logger.info(f"🔧 Hypertuning: {number_of_trials} trials completed (optimizing for profit)")
        logger.info(f"📈 Final Total Profit: ${final_total_profit:.2f}")
        logger.info(f"📈 Average Profit per Investment: ${final_profit_per_investment:.2f}")
        logger.info(f"📈 Traditional MSE: {final_mse:.4f}")
        logger.info("🎯 Threshold optimization included for prediction filtering")
        logger.info(f"💾 Model saved to MLflow run: {saved_run_id}")
        logger.info("📊 Used updated save_model method with universal logging function")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR in standalone XGBoost hypertuning: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main() 