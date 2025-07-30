"""
MLP Main Module

This module contains the main function and MLflow integration for standalone MLP execution.
Includes the complete pipeline for hypertuning and evaluation.
"""

import torch
import pandas as pd
import mlflow
import mlflow.pytorch
import optuna
import pickle
import os
from datetime import datetime
from typing import Dict, Any

from src.models.time_series.mlp.mlp_predictor import MLPPredictor
from src.utils.logger import get_logger
from src.utils.mlflow_utils import MLFlowManager
from src.data_utils.ml_data_pipeline import prepare_ml_data_for_training_with_cleaning
from src.models.time_series.mlp.mlp_evaluation import MLPEvaluationMixin
from src.models.time_series.mlp.mlp_architecture import MLPDataUtils
from src.models.time_series.mlp.mlp_optimization import MLPOptimizationMixin
from src.models.time_series.mlp.mlp_training import MLPTrainingMixin

logger = get_logger(__name__)
experiment_name = "mlp_stock_predictor"

# Remove feature outliers using IQR method (more robust than percentile)
def remove_feature_outliers(df, iqr_multiplier=1.5):
    """Remove outliers from features using IQR method."""
    df_clean = df.copy()
    for col in df.columns:
        if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR
            
            # Cap outliers instead of removing rows to preserve data
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df_clean
class MLPPredictorWithMLflow(MLPPredictor, MLPEvaluationMixin, MLPOptimizationMixin, MLPTrainingMixin):
    """
    MLPPredictor with MLflow integration capabilities.
    """
    
    def save_model(self, metrics: Dict[str, float], params: Dict[str, Any], 
                    X_eval: pd.DataFrame, experiment_name: str = None, scaler=None) -> str:
        """
        Save the trained MLP model to MLflow with all necessary components
        
        Args:
            metrics: Model evaluation metrics to log
            params: Model parameters to log
            X_eval: Evaluation features for signature generation
            experiment_name: Experiment name (uses default if None)
            scaler: Fitted StandardScaler instance to save with model
            
        Returns:
            MLflow run ID where the model was saved
        """
        if self.model is None:
            raise RuntimeError("No trained model to save")
        
        if experiment_name is None:
            experiment_name = f"{self.model_name}_experiment"
        
        try:
            # Define pip requirements for the model with explicit pip version
            pip_requirements = [
                "mlflow>=2.22.0",
                "torch>=2.0.0",
                "pandas>=2.3.0",
                "numpy>=2.3.0",
                "scikit-learn>=1.4.2"
            ]
            
            # End any existing run before starting a new one
            try:
                mlflow.end_run()
                logger.info("Ended existing MLflow run before starting new one")
            except Exception as e:
                logger.debug(f"No existing run to end: {e}")
            
            # Set up MLflow tracking
            mlflow.set_experiment(experiment_name)

            # Start a new run
            with mlflow.start_run(
                run_name=f"mlp_final_{datetime.now().strftime('%Y%m%d_%H%M')}"
            ) as run:
                # Log parameters (excluding base params if desired)
                params_to_log = {
                    k: v
                    for k, v in params.items()
                    if k not in ["device", "objective", "verbosity", "seed", "nthread", "verbose"]
                }
                mlflow.log_params(params_to_log)
                mlflow.log_metrics(metrics)

                if scaler is not None:
                    try:
                        # Create temporary directory and save scaler with specific name
                        temp_dir = "src/models/scalers"
                        os.makedirs(temp_dir, exist_ok=True)
                        scaler_path = os.path.join(temp_dir, 'scaler.pkl')
                        
                        with open(scaler_path, 'wb') as f:
                            pickle.dump(scaler, f)
                        
                        mlflow.log_artifact(scaler_path, artifact_path="scaler")
                        
                        # Clean up temp files
                        os.unlink(scaler_path)
                        logger.info("✅ Feature scaler saved to MLflow artifacts")
                    except Exception as scaler_error:
                        logger.warning(f"⚠️ Could not save scaler: {str(scaler_error)}")
                else:
                    logger.info("ℹ️ No scaler provided - model will use raw features for prediction")
                
                # Create input example using the DataFrame X_eval
                input_example = X_eval.iloc[:5].copy()

                # Identify and convert integer columns to float64
                if hasattr(input_example, "dtypes"):
                    for col in input_example.columns:
                        if input_example[col].dtype.kind == "i":
                            logger.info(f"Converting integer column '{col}' to float64 for signature")
                            input_example[col] = input_example[col].astype("float64")
                
                # Infer signature using the predictor's predict method
                try:
                    predictions_example = self.predict(input_example)
                    logger.info("✅ Signature generated using predictor predict method")
                    
                except Exception as predict_error:
                    logger.warning(f"⚠️ Could not use predictor predict method for signature: {str(predict_error)}")
                    logger.info("🔄 Falling back to direct model inference for signature generation")
                    
                    # Fallback to direct model inference
                    self.model.eval()
                    with torch.no_grad():
                        input_tensor = torch.FloatTensor(input_example.values)
                        device = next(self.model.parameters()).device
                        input_tensor = input_tensor.to(device)
                        
                        if input_tensor.device != device:
                            logger.warning(f"⚠️ Device mismatch detected: input_tensor on {input_tensor.device}, model on {device}")
                            input_tensor = input_tensor.to(device)
                        
                        predictions_example = self.model(input_tensor).cpu().numpy()
                
                signature = mlflow.models.infer_signature(input_example, predictions_example)
                
                # Log the predictor directly instead of using wrapper
                try:
                    mlflow.pytorch.log_model(
                        pytorch_model=self,
                        name="model",
                        signature=signature,
                    )
                    logger.info("✅ MLP Predictor successfully logged to MLflow")
                except RuntimeError as model_error:
                    if "device" in str(model_error).lower():
                        logger.error(f"❌ Device mismatch error during model logging: {str(model_error)}")
                        logger.info("🔄 Attempting to move model to CPU for logging...")
                        try:
                            # Create a temporary copy for CPU logging
                            model_cpu = self.model.cpu()
                            temp_predictor = MLPPredictor(
                                model_name=f"{self.model_name}_temp",
                                config=self.config,
                                threshold_evaluator=self.threshold_evaluator
                            )
                            temp_predictor.model = model_cpu
                            temp_predictor.scaler = scaler
                            temp_predictor.feature_names = self.feature_names
                            
                            mlflow.pytorch.log_model(
                                pytorch_model=temp_predictor,
                                name="model",
                                signature=signature,
                            )
                            self.model.to(device)
                            logger.info("✅ Model successfully logged to MLflow (CPU fallback)")
                        except Exception as cpu_error:
                            logger.error(f"❌ Failed to log model even with CPU fallback: {str(cpu_error)}")
                            raise
                    else:
                        logger.error(f"❌ Model logging failed: {str(model_error)}")
                        raise
                except Exception as e:
                    logger.error(f"❌ Unexpected error during model logging: {str(e)}")
                    raise
                
                logger.info(f"✅ MLP model saved successfully. Run ID: {run.info.run_id}")
                return run.info.run_id

        except Exception as e:
            logger.error(f"Error saving model to MLflow: {str(e)}")
            return None

    def load_model(self, run_id: str, experiment_name: str = None, model_id: str = None) -> bool:
        """
        Load a trained MLP model from MLflow based on the given run ID or model ID
        
        Args:
            run_id: MLflow run ID to load the model from (optional if model_id provided)
            experiment_name: Experiment name (uses default if None)
            model_id: Optional model ID for specific model loading (preferred over run_id)
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            if experiment_name is None:
                experiment_name = f"{self.model_name}_experiment"
            
            # Set the experiment
            mlflow.set_experiment(experiment_name)
            
            # Get MLflow client for artifact access
            client = mlflow.tracking.MlflowClient()
            
            # Load scaler from run artifacts if run_id is provided
            scaler_loaded = False
            if run_id:
                scaler_loaded = self._load_scaler_from_run(client, run_id)
            
            # Try to load model using model_id first (preferred method)
            model_loaded = False
            if model_id:
                try:
                    logger.info(f"🎯 Loading model using specific model ID: {model_id}")
                    
                    # Get the logged model directly using mlflow.get_logged_model()
                    logged_model_info = mlflow.get_logged_model(model_id)
                    logger.info(f"✅ Retrieved logged model: {logged_model_info.name}")

                    loaded_predictor = mlflow.pytorch.load_model(logged_model_info.model_uri)
                    
                    # Check if loaded model is a wrapper or raw model
                    if hasattr(loaded_predictor, 'model') and hasattr(loaded_predictor, 'scaler'):
                        # It's a wrapper - extract components
                        self.model = loaded_predictor.model
                        self.scaler = loaded_predictor.scaler
                        self.feature_names = loaded_predictor.feature_names
                        self.config.update(loaded_predictor.config)
                        logger.info("✅ Model wrapper loaded successfully with all components")
                    else:
                        # It's a raw model - use as is
                        self.model = loaded_predictor
                        logger.info("✅ Raw model loaded (no wrapper components)")
                    
                    self.model.to(self.device)
                    model_loaded = True
                    
                    logger.info("✅ Model loaded successfully from logged model")
                    
                    # Extract feature names from model signature if not already loaded
                    if not hasattr(self, 'feature_names') or not self.feature_names:
                        self._extract_feature_names_from_model(logged_model_info.model_uri, "logged model")
                    
                except Exception as model_error:
                    logger.error(f"❌ Failed to load model using model_id {model_id}: {str(model_error)}")
                    return False
            
            if not model_loaded:
                logger.error("❌ Failed to load model from all available sources")
                return False
            
            # Mark as trained
            self.is_trained = True
            
            logger.info("✅ Model loaded successfully")
            if model_id:
                logger.info(f"   Model ID: {model_id}")
            if run_id:
                logger.info(f"   Run ID: {run_id}")
            logger.info(f"   Experiment: {experiment_name}")
            logger.info(f"   Scaler loaded: {scaler_loaded}")
            logger.info(f"   Feature names loaded: {len(self.feature_names) if hasattr(self, 'feature_names') and self.feature_names else 0}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            return False

    def _load_scaler_from_run(self, client, run_id: str) -> bool:
        """
        Load scaler from run artifacts
        
        Args:
            client: MLflow client
            run_id: Run ID to load scaler from
            
        Returns:
            bool: True if scaler loaded successfully, False otherwise
        """
        try:
            artifacts = client.list_artifacts(run_id)
            logger.info(f"📦 Found {len(artifacts)} artifacts in run {run_id}")
            for artifact in artifacts:
                logger.info(f"   - {artifact.path}")
            
            # Find scaler artifact
            scaler_artifact_path = None
            for artifact in artifacts:
                if artifact.path == "scaler" or artifact.path.endswith("scaler.pkl"):
                    scaler_artifact_path = artifact.path
                    break
            
            if scaler_artifact_path:
                try:
                    # Download scaler artifact
                    local_path = "src/models/scalers"
                    client.download_artifacts(run_id, scaler_artifact_path, dst_path=local_path)
                    scaler_local_path = os.path.join(local_path, "scaler", "scaler.pkl")
                    with open(scaler_local_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                    
                    logger.info("✅ Feature scaler loaded from MLflow artifacts")
                    
                    # Clean up downloaded file
                    os.unlink(scaler_local_path)
                    
                    return True
                    
                except Exception as scaler_error:
                    logger.warning(f"⚠️ Could not load scaler: {str(scaler_error)}")
                    return False
            else:
                logger.info("ℹ️ No scaler artifact found - model will use raw features for prediction")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ Error listing artifacts for scaler loading: {str(e)}")
            return False

    def _extract_feature_names_from_model(self, model_uri: str, source: str):
        """
        Extract feature names from model signature
        
        Args:
            model_uri: Model URI to extract signature from
            source: Source description for logging
        """
        feature_names_from_signature = []
        try:
            # Direct approach: get signature from model URI
            logger.debug(f"Attempting to load model signature from: {model_uri}")
            
            # Load model info to get signature
            from mlflow.models import get_model_info
            model_info = get_model_info(model_uri)
            logger.debug(f"Model info loaded: {model_info is not None}")
            
            if model_info and model_info.signature:
                logger.debug(f"Model signature found: {model_info.signature is not None}")
                
                if model_info.signature.inputs:
                    logger.debug(f"Signature inputs found: {len(model_info.signature.inputs.inputs) if hasattr(model_info.signature.inputs, 'inputs') else 'No inputs attr'}")
                    
                    # Extract feature names from signature inputs
                    # Handle different signature input formats
                    if hasattr(model_info.signature.inputs, 'inputs'):
                        # Schema format
                        for input_spec in model_info.signature.inputs.inputs:
                            if hasattr(input_spec, 'name') and input_spec.name:
                                feature_names_from_signature.append(input_spec.name)
                    elif hasattr(model_info.signature.inputs, 'schema'):
                        # Alternative schema format
                        if hasattr(model_info.signature.inputs.schema, 'input_names'):
                            feature_names_from_signature = model_info.signature.inputs.schema.input_names
                    else:
                        logger.debug(f"Signature inputs type: {type(model_info.signature.inputs)}")
                        logger.debug(f"Signature inputs attributes: {dir(model_info.signature.inputs)}")
                    
                    if feature_names_from_signature:
                        self.feature_names = feature_names_from_signature
                        logger.info(f"✅ Loaded {len(self.feature_names)} feature names from {source} signature")
                        logger.debug(f"First 10 features: {self.feature_names[:10]}")
                    else:
                        logger.warning(f"⚠️ No feature names found in {source} signature inputs")
                else:
                    logger.warning(f"⚠️ {source} signature has no inputs")
            else:
                logger.warning(f"⚠️ No model signature found in {source} model info")
                    
        except Exception as signature_error:
            logger.warning(f"⚠️ Could not extract feature names from {source} signature: {str(signature_error)}")
            logger.debug(f"Signature error details: {type(signature_error).__name__}: {signature_error}")
            
            # Fallback: Try to extract feature names from the loaded model's input layer
            try:
                logger.info(f"🔄 Attempting fallback: extracting feature names from {source} model architecture...")
                if hasattr(self.model, 'input_size'):
                    # If the model has an input_size attribute, we can infer the number of features
                    input_size = self.model.input_size
                    logger.info(f"Model input size: {input_size}")
                    # Note: We can't get feature names from architecture, but we can at least validate the size
                    logger.warning(f"⚠️ Cannot extract feature names from {source} model architecture - will need to be provided externally")
            except Exception as fallback_error:
                logger.warning(f"⚠️ Fallback feature extraction for {source} also failed: {str(fallback_error)}")


def main():
    """
    Standalone MLP hypertuning and evaluation using load_all_data
    """
    logger.info("=" * 80)
    logger.info("🎯 STANDALONE MLP HYPERTUNING & EVALUATION")
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
        number_of_trials = 3
        n_features_to_select = 80
        
        # OPTION 1: Use the enhanced data preparation function with cleaning (direct import)
        data_result = prepare_ml_data_for_training_with_cleaning(
            prediction_horizon=prediction_horizon,
            split_date='2025-02-01',
            ticker=None, 
            clean_features=True,  
            use_cache=True, 
        )
        
        # Extract prepared data
        X_train = data_result['X_train']
        X_test = data_result['X_test']
        y_train = data_result['y_train']
        y_test = data_result['y_test']
        target_column = data_result['target_column']
        train_date_range = data_result['train_date_range']
        test_date_range = data_result['test_date_range']

        logger.info(f"✅ Data loaded successfully y_test range: {y_test.min()} to {y_test.max()} avg: {y_test.mean()}")
        
        # Remove outliers from target variables to improve prediction quality
        logger.info("🧹 Removing outliers from target variables...")
        
        # Calculate percentiles for outlier removal (keep middle 95%)
        y_train_lower = y_train.quantile(0.05)
        y_train_upper = y_train.quantile(0.95)
        y_test_lower = y_test.quantile(0.05)
        y_test_upper = y_test.quantile(0.95)
        
        # Create outlier masks
        y_train_outlier_mask = (y_train >= y_train_lower) & (y_train <= y_train_upper)
        y_test_outlier_mask = (y_test >= y_test_lower) & (y_test <= y_test_upper)
        
        # Apply outlier removal
        X_train_clean_outliers = X_train[y_train_outlier_mask]
        y_train_clean = y_train[y_train_outlier_mask]
        X_test_clean_outliers = X_test[y_test_outlier_mask]
        y_test_clean = y_test[y_test_outlier_mask]
        
        # Log outlier removal results
        logger.info("📊 Target outlier removal results:")
        logger.info(f"   Training: {len(y_train)} → {len(y_train_clean)} samples ({len(y_train) - len(y_train_clean)} outliers removed)")
        logger.info(f"   Testing: {len(y_test)} → {len(y_test_clean)} samples ({len(y_test) - len(y_test_clean)} outliers removed)")
        logger.info(f"   Training target range: {y_train_clean.min():.6f} to {y_train_clean.max():.6f} (avg: {y_train_clean.mean():.6f})")
        logger.info(f"   Testing target range: {y_test_clean.min():.6f} to {y_test_clean.max():.6f} (avg: {y_test_clean.mean():.6f})")
        
        # Update data with cleaned versions
        X_train = X_train_clean_outliers
        X_test = X_test_clean_outliers
        y_train = y_train_clean
        y_test = y_test_clean
        
        # Validate and clean the data
        X_train_clean= MLPDataUtils.validate_and_clean_data(X_train)
        X_train_scaled, _ = MLPDataUtils.scale_data(X_train_clean, None, True)
        X_test_clean = MLPDataUtils.validate_and_clean_data(X_test)
        
        # Extract cleaned data
        X_train = X_train_clean
        X_test = X_test_clean
        
        # Create SINGLE MLPPredictorWithMLflow instance for entire pipeline
        mlp_model = MLPPredictorWithMLflow(
            model_name="mlp_complete_pipeline",
            config={'input_size': len(X_train.columns)}
        )

        # 2. Perform feature selection using the same instance
        # selected_features = mlp_model.select_features(X_train_scaled, y_train, n_features_to_select)
        numerical_features = []
        for col in X_train.columns:
            if X_train[col].dtype in ['float64', 'float32', 'int64', 'int32', 'float', 'int']:
                numerical_features.append(col)
        
        selected_features = numerical_features
        logger.info(f"✅ Using {len(selected_features)} numerical features - no feature selection applied")
        # Create new DataFrames with only the selected features
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        logger.info(f"   DataFrames updated with {len(selected_features)} selected features.")
        
        # Update the model's input size for the selected features
        mlp_model.config['input_size'] = len(selected_features)
        
        # Fit a single StandardScaler on training data only to prevent data leakage
        X_train_selected, scaler = MLPDataUtils.scale_data(X_train_selected, None, True)
        X_test_scaled, _ = MLPDataUtils.scale_data(X_test_selected, scaler, False)
        # Store the fitted scaler in the SAME model instance
        mlp_model.scaler = scaler
        
        # Create objective function using the SAME MLP model instance with selected features
        objective_function = mlp_model.objective(X_train_selected, y_train, X_test_selected, X_test_scaled, y_test, fitted_scaler=scaler)
        sampler = optuna.samplers.RandomSampler(seed=42)

        # Run optimization
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective_function, n_trials=number_of_trials, n_jobs=1)
        
        # Get best results from study
        best_params = study.best_params
        best_profit = study.best_value  # This is now threshold-optimized profit per investment
        
        logger.info("🎯 Hyperparameter optimization with threshold optimization completed!")
        logger.info(f"✅ Best Threshold-Optimized Profit per Investment: ${best_profit:.2f}")
        logger.info(f"✅ Best parameters: {best_params}")
        
        # Finalize the best model (ensure mlp_model contains the best performing model)
        mlp_model.finalize_best_model()
        
        # Get best trial info for verification (now includes threshold info)
        best_trial_info = mlp_model.get_best_trial_info()
        logger.info(f"✅ Best trial info: {best_trial_info}")
        
        # Use the SAME instance throughout - no need to create final_model
        final_model = mlp_model

        # Extract current prices for evaluation
        final_current_prices = X_test_selected['close'].values

        # Evaluate with the optimal threshold from hyperparameter optimization
        optimal_threshold = getattr(final_model, 'optimal_threshold', 0.5)
        confidence_method = getattr(final_model, 'confidence_method', 'variance')
        
        threshold_performance = final_model.threshold_evaluator.evaluate_threshold_performance(
            model=final_model,
            X_test=X_test_selected,
            y_test=y_test,
            current_prices_test=final_current_prices,
            threshold=optimal_threshold,
            confidence_method=confidence_method
        )
        
        # Also get unfiltered baseline for comparison
        baseline_predictions = final_model.predict(X_test_selected)
        baseline_profit = final_model.threshold_evaluator.calculate_profit_score(
            y_test.values, baseline_predictions, final_current_prices
        )
        baseline_profit_per_investment = baseline_profit / len(y_test)
        
        logger.info("📊 Final Results Comparison:")
        logger.info(f"   Baseline (unfiltered) profit per investment: ${baseline_profit_per_investment:.2f}")
        logger.info(f"   Threshold-optimized profit per investment: ${threshold_performance['profit_per_investment']:.2f}")
        logger.info(f"   Improvement ratio: {threshold_performance['profit_per_investment'] / baseline_profit_per_investment if baseline_profit_per_investment != 0 else 0:.2f}x")
        logger.info(f"   Samples kept: {threshold_performance['samples_evaluated']}/{len(X_test_selected)} ({threshold_performance['samples_kept_ratio']:.1%})")
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
        
        logger.info("📊 Final Optimized Results:")
        logger.info(f"   Total Profit: ${final_total_profit:.2f}")
        logger.info(f"   Profit per Investment: ${final_profit_per_investment:.2f}")
        logger.info(f"   Samples Used: {final_samples_kept}/{len(X_test_selected)} (threshold-filtered)")
        logger.info(f"   Traditional MSE: {final_mse:.4f}")
        logger.info(f"   Traditional MAE: {final_mae:.4f}")
        logger.info(f"   Traditional R²: {final_r2:.4f}")
        
        
        # Prepare comprehensive metrics for logging
        final_metrics = {
            "final_total_profit": final_total_profit,
            "final_profit_per_investment": final_profit_per_investment,
            "final_mse": final_mse,
            "final_mae": final_mae,
            "final_r2": final_r2,
            "final_test_samples": len(y_test),
            "hypertuning_best_profit": best_profit,
            "hypertuning_trials_completed": number_of_trials,
            "features_selected": len(selected_features)
        }
        
        # Add threshold optimization metrics if successful
        final_metrics.update(threshold_metrics)
        
        # Prepare comprehensive parameters for logging
        final_params = best_params.copy()
        final_params.update({
            "prediction_horizon": prediction_horizon,
            "hypertuning_trials": number_of_trials,
            "target_column": target_column,
            "split_date": data_result['split_date'],
            "feature_count": data_result['feature_count'],
            "train_samples": data_result['train_samples'],
            "test_samples": data_result['test_samples'],
            "threshold_method": final_model.confidence_method,
            "threshold_optimization_during_hypertuning": True,
            "optimal_threshold_from_hypertuning": final_model.best_threshold_info['optimal_threshold']
        })
        
        # Use the universal logging function via save_model method
        saved_run_id = final_model.save_model(
            metrics=final_metrics,
            params=final_params,
            X_eval=X_test_selected,
            experiment_name=experiment_name,
            scaler=scaler
        )
        
        logger.info(f"✅ Model saved using updated save_model method. Run ID: {saved_run_id}")
        
        logger.info("=" * 80)
        logger.info("🎉 STANDALONE MLP HYPERTUNING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"📊 Dataset: {data_result['train_samples'] + data_result['test_samples']:,} samples, {data_result['feature_count']} features")
        logger.info(f"🎯 Target: {target_column} ({prediction_horizon}-day horizon)")
        logger.info(f"📅 Train period: {train_date_range}")
        logger.info(f"📅 Test period: {test_date_range}")
        logger.info(f"🔧 Hypertuning: {number_of_trials} trials completed (optimizing for profit)")
        logger.info(f"📈 Final Total Profit: ${final_total_profit:.2f}")
        logger.info(f"📈 Average Profit per Investment: ${final_profit_per_investment:.2f}")
        logger.info(f"📈 Traditional MSE: {final_mse:.4f}")
        logger.info(f"💾 Model saved to MLflow run: {saved_run_id}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR in standalone MLP hypertuning: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main() 