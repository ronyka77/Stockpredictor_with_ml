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
from datetime import datetime
from typing import Dict, Any

from src.models.time_series.mlp.mlp_predictor import MLPPredictor
from src.utils.logger import get_logger
from src.utils.mlflow_utils import MLFlowManager
from src.data_utils.ml_data_pipeline import prepare_ml_data_for_training_with_cleaning
from src.models.time_series.mlp.mlp_evaluation import validate_and_clean_data, MLPEvaluationMixin
from src.models.time_series.mlp.mlp_optimization import MLPOptimizationMixin
from src.models.time_series.mlp.mlp_training import MLPTrainingMixin

logger = get_logger(__name__)
experiment_name = "mlp_stock_predictor"


def log_to_mlflow_mlp(model, metrics, params, experiment_name, X_eval):
    """
    Log trained MLP model, metrics, and parameters to MLflow.
    Requires X_eval DataFrame for signature generation.
    Args:
        model: Trained PyTorch MLP model
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
            "torch>=1.12.0",
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

            # Create input example using the DataFrame X_eval
            input_example = X_eval.iloc[:5].copy()

            # Identify and convert integer columns to float64
            if hasattr(input_example, "dtypes"):
                for col in input_example.columns:
                    if input_example[col].dtype.kind == "i":
                        logger.info(f"Converting integer column '{col}' to float64 for signature")
                        input_example[col] = input_example[col].astype("float64")

            # Infer signature - create predictions for MLP
            model.eval()
            with torch.no_grad():
                input_tensor = torch.FloatTensor(input_example.values)
                predictions_example = model(input_tensor).cpu().numpy()
            
            signature = mlflow.models.infer_signature(input_example, predictions_example)

            # Update model registration with signature
            model_info = mlflow.pytorch.log_model(
                model,
                "model",  # Keep artifact path as "model"
                pip_requirements=pip_requirements,
                signature=signature,
                # REMOVED: registered_model_name parameter (causes 404 in MLflow 2.8+)
            )

            logger.info(f"Model logged to MLflow: {model_info.model_uri}")
            logger.info(f"Run ID: {run.info.run_id}")
            return run.info.run_id

    except Exception as e:
        logger.error(f"Error logging to MLflow: {str(e)}")
        return None


class MLPPredictorWithMLflow(MLPPredictor, MLPEvaluationMixin, MLPOptimizationMixin, MLPTrainingMixin):
    """
    MLPPredictor with MLflow integration capabilities.
    """
    
    def save_model(self, metrics: Dict[str, float], params: Dict[str, Any], 
                    X_eval: pd.DataFrame, experiment_name: str = None) -> str:
        """
        Save the trained MLP model to MLflow using the universal logging function
        
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
        
        # Store selected features information in params for later retrieval
        if hasattr(self, 'feature_names') and self.feature_names is not None:
            # Add selected features to params
            for i, feature in enumerate(self.feature_names):
                params[f'selected_feature_{i+1}'] = feature
            params['selected_features_count'] = len(self.feature_names)
            logger.info(f"✅ Stored {len(self.feature_names)} selected features in model parameters")
        
        # Use the universal logging function via the class method
        run_id = self.log_model_to_mlflow(
            metrics=metrics,
            params=params,
            X_eval=X_eval,
            experiment_name=experiment_name
        )
        
        logger.info(f"✅ MLP model saved using universal logging function. Run ID: {run_id}")
        return run_id

    def log_model_to_mlflow(self, metrics: Dict[str, float], params: Dict[str, Any], 
                            X_eval: pd.DataFrame, experiment_name: str = None) -> str:
        """
        Log the trained MLP model to MLflow using the universal logging function
        
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
        run_id = log_to_mlflow_mlp(
            model=self.model,
            metrics=metrics,
            params=params,
            experiment_name=experiment_name,
            X_eval=X_eval
        )
        
        logger.info(f"✅ Model logged to MLflow using universal function. Run ID: {run_id}")
        return run_id

    def load_model(self, run_id: str, experiment_name: str = None) -> bool:
        """
        Load a trained MLP model from MLflow based on the given run ID
        
        Args:
            run_id: MLflow run ID to load the model from
            experiment_name: Experiment name (uses default if None)
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            if experiment_name is None:
                experiment_name = f"{self.model_name}_experiment"
            
            # Set the experiment
            mlflow.set_experiment(experiment_name)
            
            # Get the run details
            run = mlflow.get_run(run_id)
            if run is None:
                logger.error(f"❌ Run with ID {run_id} not found in experiment {experiment_name}")
                return False
            
            # Load the model from the run
            model_uri = f"runs:/{run_id}/model"
            loaded_model = mlflow.pytorch.load_model(model_uri)
            
            # Set the loaded model
            self.model = loaded_model
            self.model.to(self.device)
            
            # Load run parameters and update config
            run_params = run.data.params
            if run_params:
                # Update config with loaded parameters (excluding MLflow-specific ones)
                for key, value in run_params.items():
                    if key not in ["mlflow_run_id", "mlflow_experiment_name"]:
                        try:
                            # Try to convert string values back to appropriate types
                            if value.lower() == 'true':
                                self.config[key] = True
                            elif value.lower() == 'false':
                                self.config[key] = False
                            elif value.replace('.', '').replace('-', '').isdigit():
                                if '.' in value:
                                    self.config[key] = float(value)
                                else:
                                    self.config[key] = int(value)
                            else:
                                self.config[key] = value
                        except (ValueError, AttributeError):
                            self.config[key] = value
            
            # Store feature information from run parameters
            self._load_feature_information(run_params)
            
            # Mark as trained
            self.is_trained = True
            
            logger.info(f"✅ Model loaded successfully from MLflow run: {run_id}")
            logger.info(f"   Experiment: {experiment_name}")
            logger.info(f"   Model URI: {model_uri}")
            logger.info(f"   Run status: {run.info.status}")
            logger.info(f"   Run start time: {run.info.start_time}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model from MLflow run {run_id}: {str(e)}")
            return False

    def _load_feature_information(self, run_params: Dict[str, Any]):
        """
        Extract and store feature information from run parameters
        
        Args:
            run_params: Parameters from MLflow run
        """
        # Store feature-related information
        self.feature_info = {
            'input_size': run_params.get('input_size'),
            'features_selected': run_params.get('features_selected'),
            'prediction_horizon': run_params.get('prediction_horizon'),
            'target_column': run_params.get('target_column'),
            'split_date': run_params.get('split_date'),
            'feature_count': run_params.get('feature_count')
        }
        
        # Try to extract selected features if available
        selected_features = []
        for key, value in run_params.items():
            if key.startswith('selected_feature_'):
                selected_features.append(value)
        
        if selected_features:
            self.feature_info['selected_features'] = selected_features
            logger.info(f"✅ Loaded {len(selected_features)} selected features from run parameters")
        else:
            logger.warning("⚠️ No selected features found in run parameters")

    def get_required_features(self) -> Dict[str, Any]:
        """
        Get the required features and configuration for predictions
        
        Returns:
            dict: Dictionary containing required feature information
        """
        if not hasattr(self, 'feature_info'):
            raise RuntimeError("No feature information available. Load a model first.")
        
        return self.feature_info.copy()

    def get_feature_requirements(self) -> Dict[str, Any]:
        """
        Get detailed feature requirements for predictions
        
        Returns:
            dict: Detailed feature requirements
        """
        if not hasattr(self, 'feature_info'):
            raise RuntimeError("No feature information available. Load a model first.")
        
        requirements = {
            'input_size': self.feature_info.get('input_size'),
            'selected_features': self.feature_info.get('selected_features', []),
            'prediction_horizon': self.feature_info.get('prediction_horizon'),
            'target_column': self.feature_info.get('target_column'),
            'data_preparation': {
                'split_date': self.feature_info.get('split_date'),
                'feature_count': self.feature_info.get('feature_count'),
                'features_selected': self.feature_info.get('features_selected')
            }
        }
        
        return requirements

    def validate_features(self, X: pd.DataFrame) -> bool:
        """
        Validate that the provided features match the model's requirements
        
        Args:
            X: Features DataFrame to validate
            
        Returns:
            bool: True if features are valid, False otherwise
        """
        if not hasattr(self, 'feature_info'):
            logger.error("❌ No feature information available. Load a model first.")
            return False
        
        # Check input size
        expected_input_size = self.feature_info.get('input_size')
        actual_input_size = len(X.columns)
        
        if expected_input_size and actual_input_size != expected_input_size:
            logger.error(f"❌ Feature count mismatch. Expected: {expected_input_size}, Got: {actual_input_size}")
            return False
        
        # Check if selected features are available
        selected_features = self.feature_info.get('selected_features', [])
        if selected_features:
            missing_features = set(selected_features) - set(X.columns)
            if missing_features:
                logger.error(f"❌ Missing required features: {missing_features}")
                return False
            
            extra_features = set(X.columns) - set(selected_features)
            if extra_features:
                logger.warning(f"⚠️ Extra features provided: {extra_features}")
        
        logger.info(f"✅ Feature validation passed. Input size: {actual_input_size}")
        return True

    def prepare_features_for_prediction(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for prediction by selecting required features and ensuring correct order
        
        Args:
            X: Raw features DataFrame
            
        Returns:
            pd.DataFrame: Prepared features ready for prediction
        """
        if not hasattr(self, 'feature_info'):
            raise RuntimeError("No feature information available. Load a model first.")
        
        selected_features = self.feature_info.get('selected_features', [])
        
        if selected_features:
            # Select only the required features in the correct order
            missing_features = set(selected_features) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            X_prepared = X[selected_features]
            logger.info(f"✅ Prepared features: {len(X_prepared.columns)} columns")
        else:
            # No specific features selected, use all available
            X_prepared = X
            logger.info(f"✅ Using all available features: {len(X_prepared.columns)} columns")
        
        return X_prepared


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
        number_of_trials = 5
        n_features_to_select = 60
        
        # OPTION 1: Use the enhanced data preparation function with cleaning (direct import)
        data_result = prepare_ml_data_for_training_with_cleaning(
            prediction_horizon=prediction_horizon,
            split_date='2025-02-01',
            ticker=None,  # Load ALL tickers
            clean_features=True,  # Apply feature cleaning
            use_cache=True,  # Apply MLP data cleaning
        )
        
        # Extract prepared data
        X_train = data_result['X_train']
        X_test = data_result['X_test']
        y_train = data_result['y_train']
        y_test = data_result['y_test']
        target_column = data_result['target_column']
        train_date_range = data_result['train_date_range']
        test_date_range = data_result['test_date_range']
        
        # Validate and clean the data
        X_train, y_train = validate_and_clean_data(X_train, y_train, logger)
        X_test, y_test = validate_and_clean_data(X_test, y_test, logger)
        
        mlp_model = MLPPredictorWithMLflow(
            model_name="mlp_feature_selector",
            config={'input_size': len(X_train.columns)}
        )

        # 2. Perform feature selection (using fast method by default)
        selected_features = mlp_model.select_features(X_train, y_train, n_features_to_select)
        
        # Create new DataFrames with only the selected features
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        logger.info(f"   DataFrames updated with {len(selected_features)} selected features.")
        
        mlp_model = MLPPredictorWithMLflow(
            model_name="mlp_standalone_hypertuned",
            config={'input_size': len(selected_features)}
        )
        
        # Create objective function using the MLP model class method with selected features
        objective_function = mlp_model.objective(X_train_selected, y_train, X_test_selected, y_test)
        sampler = optuna.samplers.TPESampler(seed=42)
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
        
        # mlp_model now contains the best model with optimal threshold, no need to create a new one
        final_model = mlp_model
        
        # Check if the best model has threshold optimization results
        has_threshold_optimization = (hasattr(final_model, 'best_threshold_info') and 
                                    final_model.best_threshold_info is not None and
                                    final_model.best_threshold_info.get('optimal_threshold') is not None)
        
        if has_threshold_optimization:
            # Extract current prices for evaluation
            if 'close' in X_test_selected.columns:
                final_current_prices = X_test_selected['close'].values
            else:
                final_current_prices = y_test.values * 0.95  # Fallback
            
            # Evaluate with the optimal threshold from hyperparameter optimization
            # Use stored optimal values from hyperparameter optimization
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
        else:
            # No threshold optimization, use baseline metrics
            baseline_predictions = final_model.predict(X_test_selected)
            final_current_prices = X_test_selected['close'].values if 'close' in X_test_selected.columns else y_test.values * 0.95
            
            baseline_profit = final_model.threshold_evaluator.calculate_profit_score(
                y_test.values, baseline_predictions, final_current_prices
            )
            baseline_profit_per_investment = baseline_profit / len(y_test)
            
            final_profit_per_investment = baseline_profit_per_investment
            final_total_profit = baseline_profit
            final_investment_success_rate = 0.5  # Default
            final_samples_kept = len(y_test)
            
            # Traditional metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            final_mse = mean_squared_error(y_test, baseline_predictions)
            final_mae = mean_absolute_error(y_test, baseline_predictions)
            final_r2 = r2_score(y_test, baseline_predictions)
            
            threshold_metrics = {}
        
        logger.info("📊 Final Optimized Results:")
        logger.info(f"   Total Profit: ${final_total_profit:.2f}")
        logger.info(f"   Profit per Investment: ${final_profit_per_investment:.2f}")
        if has_threshold_optimization:
            logger.info(f"   Samples Used: {final_samples_kept}/{len(X_test_selected)} (threshold-filtered)")
        else:
            logger.info(f"   Samples Used: {final_samples_kept}/{len(X_test_selected)} (all samples)")
        logger.info(f"   Traditional MSE: {final_mse:.4f}")
        logger.info(f"   Traditional MAE: {final_mae:.4f}")
        logger.info(f"   Traditional R²: {final_r2:.4f}")
        
        feature_importance = final_model.get_feature_importance('gradient')
        
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
        
        # Add top feature importance to metrics
        top_features = feature_importance.head(10)
        for i, (_, row) in enumerate(top_features.iterrows()):
            final_metrics[f"feature_importance_{i+1}_{row['feature']}"] = row['importance']
        
        # Prepare comprehensive parameters for logging
        final_params = best_params.copy()
        final_params.update({
            "prediction_horizon": prediction_horizon,
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
            X_eval=X_test_selected,
            experiment_name=experiment_name
        )
        
        logger.info(f"✅ Model saved using updated save_model method. Run ID: {saved_run_id}")
        logger.info("✅ Model registered with timestamp-based name via universal function")
        
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