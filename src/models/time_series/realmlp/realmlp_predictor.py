from typing import Dict, Any, Optional, List

import os
import json
import hashlib
import tempfile
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import mlflow
import mlflow.pytorch
import optuna

from src.models.time_series.base_pytorch_model import PyTorchBasePredictor
from src.models.time_series.realmlp.realmlp_architecture import RealMLPModule
from src.models.time_series.realmlp.realmlp_preprocessing import RealMLPPreprocessor
from src.models.time_series.realmlp.realmlp_trainer import RealMLPTrainingMixin
from src.utils.logger import get_logger
from src.models.evaluation.threshold_evaluator import ThresholdEvaluator

logger = get_logger(__name__)


class RealMLPPredictor(RealMLPTrainingMixin, PyTorchBasePredictor):
    def __init__(self, model_name: str = "RealMLP", config: Optional[Dict[str, Any]] = None, threshold_evaluator=None):
        super().__init__(model_name=model_name, config=config or {}, threshold_evaluator=threshold_evaluator)
        self.preprocessor: Optional[RealMLPPreprocessor] = None
        self.feature_names: List[str] = []
        self.threshold_evaluator: ThresholdEvaluator = threshold_evaluator or ThresholdEvaluator(
            investment_amount=(config or {}).get("investment_amount", 100.0)
        )
        self.optimal_threshold: Optional[float] = None
        self.confidence_method: str = "variance"
        self.best_threshold_info: Optional[Dict[str, Any]] = None
        self.threshold_eval_metrics: Optional[Dict[str, Any]] = None
        self.threshold_all_results = None
        self.best_investment_success_rate: float = float("-inf")
        self.best_trial_model: Optional["RealMLPPredictor"] = None
        self.best_trial_params: Optional[Dict[str, Any]] = None
        self.best_score: float = float("-inf")

    def _create_model(self) -> nn.Module:
        cfg = self.config
        input_size = cfg.get("input_size")
        if input_size is None:
            raise ValueError("Config must include 'input_size'")
        hidden_sizes = cfg.get("hidden_sizes", [512, 256, 128, 64])
        activation = cfg.get("activation", "gelu")
        dropout = cfg.get("dropout", 0.1)
        batch_norm = cfg.get("batch_norm", True)
        use_diagonal = cfg.get("use_diagonal", True)
        use_numeric_embedding = cfg.get("use_numeric_embedding", True)
        numeric_embedding_dim = cfg.get("numeric_embedding_dim", 16)

        num_categories = self.config.get("num_categories")
        cat_embed_dim = self.config.get("cat_embed_dim", 32)
        embedding_dropout = self.config.get("embedding_dropout", 0.1)

        return RealMLPModule(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
            use_diagonal=use_diagonal,
            use_numeric_embedding=use_numeric_embedding,
            numeric_embedding_dim=numeric_embedding_dim,
            numeric_embedding_out_dim=hidden_sizes[0] if use_numeric_embedding else input_size,
            num_categories=num_categories,
            cat_embed_dim=cat_embed_dim,
            embedding_dropout=embedding_dropout,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained/loaded before prediction")
        if getattr(self, "preprocessor", None) is None:
            raise RuntimeError("Preprocessor not available on predictor; cannot transform features for inference")

        numeric_cols = list(self.preprocessor.feature_names)
        missing = [c for c in numeric_cols if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required numeric feature columns for inference: {missing}")

        X_num, cat_idx = self.preprocessor.transform(X, numeric_cols=numeric_cols)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.model.to(device)

        x_tensor = torch.as_tensor(X_num, dtype=torch.float32, device=device)
        cat_tensor = None
        if cat_idx is not None:
            cat_tensor = torch.as_tensor(cat_idx, dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = self.model(x_tensor, cat_tensor)
            preds = outputs.detach().cpu().numpy().squeeze()
        return preds

    def get_prediction_confidence(self, X: pd.DataFrame, method: str = "variance") -> np.ndarray:
        # Reuse PyTorchBasePredictor's variance/simple/margin conventions by enabling dropout for variance
        if method == "variance":
            if getattr(self, "preprocessor", None) is None:
                raise RuntimeError("Preprocessor not available; cannot compute confidence")
            # Prepare tensors without calling predict() which switches to eval()
            numeric_cols = list(self.preprocessor.feature_names)
            missing = [c for c in numeric_cols if c not in X.columns]
            if missing:
                raise ValueError(f"Missing required numeric feature columns for confidence: {missing}")
            X_num, cat_idx = self.preprocessor.transform(X, numeric_cols=numeric_cols)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            x_tensor = torch.as_tensor(X_num, dtype=torch.float32, device=device)
            cat_tensor = None
            if cat_idx is not None:
                cat_tensor = torch.as_tensor(cat_idx, dtype=torch.long, device=device)

            original_training = self.model.training
            self.model.train()  # enable dropout
            try:
                n_passes = int(self.config.get("mc_dropout_passes", 5))
                preds: List[np.ndarray] = []
                with torch.no_grad():
                    for _ in range(max(1, n_passes)):
                        outputs = self.model(x_tensor, cat_tensor)
                        preds.append(outputs.detach().cpu().numpy().squeeze())
                arr = np.stack(preds, axis=0)
                var = arr.var(axis=0)
                conf = 1.0 / (1.0 + var)
            finally:
                if not original_training:
                    self.model.eval()
            # Normalize to [0,1]
            mn, mx = float(np.min(conf)), float(np.max(conf))
            return (conf - mn) / (mx - mn) if mx > mn else np.full_like(conf, 0.5)
        elif method == "simple":
            p = self.predict(X)
            return np.abs(p)
        elif method == "margin":
            p = self.predict(X)
            return np.abs(p - p.mean())
        else:
            raise ValueError(f"Unknown confidence method: {method}")

    # ------------------------- Threshold optimization & evaluation -------------------------
    def optimize_and_evaluate_threshold(
        self,
        *,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        current_prices_col: str = "close",
        confidence_method: str = "variance",
        threshold_range: tuple = (0.01, 0.99),
        n_thresholds: int = 99) -> Dict[str, Any]:
        """
        Run confidence-threshold optimization on test data, then evaluate performance at the best threshold.

        Returns a dict containing optimization and evaluation results.
        """
        if self.model is None or not self.is_trained:
            raise RuntimeError("Model must be trained/loaded before threshold evaluation")
        if getattr(self, "preprocessor", None) is None:
            raise RuntimeError("Preprocessor not available; cannot align feature columns for evaluation")
        if current_prices_col not in X_test.columns:
            raise ValueError(f"'{current_prices_col}' column is required in X_test to compute profits")

        current_prices_test = X_test[current_prices_col].to_numpy()

        logger.info(
            f"Starting threshold optimization (method={confidence_method}, range={threshold_range}, n={n_thresholds})"
        )
        opt = self.threshold_evaluator.optimize_prediction_threshold(
            model=self,
            X_test=X_test,
            y_test=y_test,
            current_prices_test=current_prices_test,
            confidence_method=confidence_method,
            threshold_range=threshold_range,
            n_thresholds=n_thresholds,
        )

        results: Dict[str, Any] = {"optimization": opt}

        if opt.get("status") == "success":
            best_thr = float(opt["optimal_threshold"])  # type: ignore[index]
            self.optimal_threshold = best_thr
            self.confidence_method = confidence_method
            self.best_threshold_info = opt.get("best_result", {})  # type: ignore[assignment]
            self.threshold_all_results = opt.get("all_results")

            logger.info(f"Evaluating performance at optimal threshold={best_thr:.3f}")
            eval_res = self.threshold_evaluator.evaluate_threshold_performance(
                model=self,
                X_test=X_test,
                y_test=y_test,
                current_prices_test=current_prices_test,
                threshold=best_thr,
                confidence_method=confidence_method,
            )
            self.threshold_eval_metrics = eval_res
            results["evaluation"] = eval_res
        else:
            logger.warning(f"Threshold optimization failed: {opt.get('message')}")

        return results

    # ------------------------- Optuna hypertuning (no data loading/saving) -------------------------
    def optuna_hypertune(
        self,
        *,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        preprocessor: RealMLPPreprocessor,
        n_trials: int = 30,
        timeout: Optional[int] = None,
        confidence_method: str = "variance",
    ) -> Dict[str, Any]:
        """
        Hyperparameter optimization using Optuna.
        - Creates a new RealMLPPredictor per trial
        - Fits the model (no data loading here)
        - Evaluates via optimize_and_evaluate_threshold on the provided validation set
        - Maximizes profit_per_investment
        - Does not save models
        """

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=42))

        base_config = dict(self.config) if isinstance(self.config, dict) else {}

        def objective(trial: optuna.Trial) -> float:
            # Sample hyperparameters
            hidden_sizes_options = [
                [512, 256, 128, 64],
                [384, 192, 96],
                [256, 128, 64],
                [256, 128],
                [768, 384, 192, 96],
            ]
            hidden_sizes = trial.suggest_categorical("hidden_sizes", hidden_sizes_options)
            dropout = trial.suggest_float("dropout", 0.05, 0.5)
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
            use_huber = trial.suggest_categorical("use_huber", [True, False])
            huber_delta = trial.suggest_float("huber_delta", 0.02, 0.2)
            numeric_embedding_dim = trial.suggest_categorical("numeric_embedding_dim", [8, 12, 16, 24, 32])
            epochs = trial.suggest_int("epochs", 10, 40)

            trial_config = {
                **base_config,
                "hidden_sizes": hidden_sizes,
                "dropout": dropout,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "use_huber": use_huber,
                "huber_delta": huber_delta,
                "numeric_embedding_dim": numeric_embedding_dim,
                "epochs": epochs,
            }
            # Ensure required keys
            trial_config["input_size"] = trial_config.get("input_size") or len(preprocessor.feature_names)

            # Fresh predictor per trial
            predictor = RealMLPPredictor(model_name="RealMLP", config=trial_config)

            try:
                predictor.fit(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    preprocessor=preprocessor,
                )

                eval_pack = predictor.optimize_and_evaluate_threshold(
                    X_test=X_val,
                    y_test=y_val,
                    confidence_method=confidence_method,
                )
                metrics = eval_pack.get("evaluation", {}) if isinstance(eval_pack, dict) else {}
                optimized_profit_score = None

                # Extract optimization dict, whether nested (wrapper) or direct
                opt_dict = None
                if isinstance(eval_pack, dict):
                    if "optimization" in eval_pack and isinstance(eval_pack["optimization"], dict):
                        opt_dict = eval_pack["optimization"]
                    elif "status" in eval_pack and "best_result" in eval_pack:
                        # Direct result from optimize_prediction_threshold
                        opt_dict = eval_pack

                best_res = opt_dict.get("best_result", {}) if isinstance(opt_dict, dict) else {}
                if not isinstance(best_res, dict):
                    best_res = {}

                # Primary score from evaluation metrics if available
                if isinstance(metrics, dict) and "profit_per_investment" in metrics:
                    try:
                        optimized_profit_score = float(metrics["profit_per_investment"]) 
                    except Exception:
                        optimized_profit_score = None
                # Fallback score from best_result (optimizer) when no evaluation present
                if optimized_profit_score is None:
                    try:
                        optimized_profit_score = float(best_res.get("test_profit_per_investment", 0.0))
                    except Exception:
                        optimized_profit_score = 0.0

                # Build threshold_info from optimization results for logging/selection
                optimal_threshold = None
                if isinstance(opt_dict, dict):
                    optimal_threshold = opt_dict.get("optimal_threshold") or best_res.get("threshold")
                threshold_info = {
                    "optimal_threshold": optimal_threshold,
                    "samples_kept_ratio": best_res.get("test_samples_ratio"),
                    "investment_success_rate": best_res.get("investment_success_rate"),
                    "custom_accuracy": best_res.get("test_custom_accuracy"),
                    "total_threshold_profit": best_res.get("test_profit"),
                    "profitable_investments": best_res.get("profitable_investments"),
                }

                # Check if this is the best trial so far (maximize profit per investment)
                if float(optimized_profit_score) > self.best_investment_success_rate:
                    self.best_investment_success_rate = float(optimized_profit_score)
                    self.best_trial_model = predictor
                    self.best_trial_params = dict(trial_config)
                    self.best_threshold_info = dict(threshold_info)

                    # Update current predictor with best trial's trained model and metadata
                    self.model = predictor.model
                    self.feature_names = list(getattr(predictor, "feature_names", []))

                    # Store the optimal threshold and confidence method
                    if threshold_info.get("optimal_threshold") is not None:
                        self.optimal_threshold = float(threshold_info["optimal_threshold"])  # type: ignore[index]
                        self.confidence_method = confidence_method

                    logger.info(f"\U0001F3AF NEW BEST TRIAL {trial.number}: Profit Per Investment = {float(optimized_profit_score):.3f}")
                    if threshold_info.get("optimal_threshold") is not None:
                        logger.info(
                            f"   Optimal threshold: {float(threshold_info['optimal_threshold']):.3f}"
                        )
                        logger.info(
                            f"   Samples kept: {float(threshold_info['samples_kept_ratio'])*100:.1f}%"
                        )
                        logger.info(
                            f"   Investment success rate: {float(threshold_info['investment_success_rate']):.3f}"
                        )
                        logger.info(
                            f"   Custom accuracy: {float(threshold_info['custom_accuracy']):.3f}"
                        )

                    self.best_score = float(optimized_profit_score)
                else:
                    logger.info(
                        f"Trial {trial.number}: Profit Per Investment = {float(optimized_profit_score):.3f} (Best: {self.best_investment_success_rate:.3f})"
                    )
            except Exception as e:
                logger.warning(f"Trial failed with error: {e}")
                score = -1e9

            # Report to Optuna (optional pruning hook)
            try:
                trial.report(score, step=0)
            except Exception:
                pass
            return float(optimized_profit_score)

        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        best_params = study.best_params if len(study.trials) else {}

        return {
            "study": study,
            "best_params": best_params,
            "best_value": self.best_score,
        }

    # ------------------------- MLflow save/load -------------------------
    def save_model(
        self,
        *,
        metrics: Dict[str, float],
        params: Dict[str, Any],
        X_eval: pd.DataFrame,
        preprocessor: Optional[RealMLPPreprocessor] = None) -> Optional[str]:
        """
        Save the trained RealMLP model and preprocessor artifacts to MLflow.
        Logs:
            - PyTorch model at artifact path 'model'
            - Preprocessor artifacts under 'preprocessor/'
            - feature_names.json (redundant with preprocessor but convenient)
            - feature_schema_hash param for strict schema matching
        """
        if self.model is None:
            raise RuntimeError("No trained model to save")

        experiment = "realmlp_stock_predictor"

        try:
            numeric_feature_names = (
                list(preprocessor.feature_names) if (preprocessor and preprocessor.feature_names) else list(X_eval.columns)
            )
        except Exception:
            numeric_feature_names = list(getattr(self, "feature_names", [])) or list(X_eval.columns)

        combined_feature_names = list(numeric_feature_names)
        if preprocessor is not None and getattr(preprocessor, "categorical_cols", None):
            for c in preprocessor.categorical_cols:
                if c not in combined_feature_names:
                    combined_feature_names.append(c)

        # Compute a stable hash for the feature schema
        try:
            joined = "\u241F".join([str(c) for c in numeric_feature_names])  # unit separator to avoid collisions
            feature_schema_hash = hashlib.sha256(joined.encode("utf-8")).hexdigest()
        except Exception:
            feature_schema_hash = ""

        # Prepare input example and signature (use numeric_feature_names for model signature)
        input_example = X_eval.loc[:, numeric_feature_names].iloc[:5].copy()
        for col in input_example.columns:
            if getattr(input_example[col], "dtype", None) is not None and input_example[col].dtype.kind == "i":
                input_example[col] = input_example[col].astype("float64")

        # Ensure model on CPU for signature inference
        self.model.eval()
        try:
            original_device = next(self.model.parameters()).device
        except Exception:
            original_device = torch.device("cpu")
        self.model.to(torch.device("cpu"))
        try:
            with torch.no_grad():
                input_tensor = torch.as_tensor(input_example.values, dtype=torch.float32)
                # If model includes ticker embedding, generate predictions with a dummy OOV categorical index vector
                has_cat = hasattr(self.model, "cat_embedding") and getattr(self.model, "cat_embedding") is not None
                if has_cat:
                    dummy_cat = torch.zeros((input_tensor.shape[0],), dtype=torch.long)
                    predictions_example = self.model(input_tensor, dummy_cat).cpu().numpy()
                else:
                    predictions_example = self.model(input_tensor).cpu().numpy()
            signature = mlflow.models.infer_signature(input_example if not has_cat else input_example, predictions_example)
        finally:
            try:
                if original_device is not None:
                    self.model.to(original_device)
            except Exception:
                pass

        # End any existing run
        try:
            mlflow.end_run()
        except Exception:
            pass

        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name="realmlp_final") as run:
            # Optionally silence MLflow env var info message
            try:
                os.environ.setdefault("MLFLOW_RECORD_ENV_VARS_IN_MODEL_LOGGING", "false")
            except Exception:
                pass
            # Log params/metrics
            try:
                params_to_log = {k: v for k, v in params.items()}
                params_to_log["feature_schema_hash"] = feature_schema_hash
                # Include threshold params if available
                if self.optimal_threshold is not None:
                    params_to_log["optimal_threshold"] = float(self.optimal_threshold)
                    params_to_log["confidence_method"] = str(getattr(self, "confidence_method", "variance"))
                mlflow.log_params(params_to_log)
            except Exception as e:
                logger.warning(f"Could not log params: {e}")
            try:
                # Log training metrics provided by caller
                mlflow.log_metrics(metrics)
                # Log evaluation metrics if available
                if isinstance(self.threshold_eval_metrics, dict):
                    metric_keys = [
                        "mse","mae","r2_score","total_profit","profit_per_investment",
                        "custom_accuracy","investment_success_rate","samples_evaluated","samples_kept_ratio",
                        "average_confidence","avg_confidence_all","avg_confidence_filtered"
                    ]
                    loggable = {}
                    for k in metric_keys:
                        if k in self.threshold_eval_metrics and self.threshold_eval_metrics[k] is not None:
                            val = self.threshold_eval_metrics[k]
                            try:
                                loggable[k] = float(val)
                            except Exception:
                                pass
                    if loggable:
                        mlflow.log_metrics(loggable)
                # Persist JSON artifacts for completeness
                if self.best_threshold_info is not None:
                    mlflow.log_dict(self.best_threshold_info, "evaluation/best_threshold_info.json")
                if isinstance(self.threshold_eval_metrics, dict):
                    mlflow.log_dict(self.threshold_eval_metrics, "evaluation/threshold_evaluation_metrics.json")
            except Exception as e:
                logger.warning(f"Could not log metrics/artifacts: {e}")

            # Log feature names explicitly (alongside preprocessor copy)
            try:
                # Save both numeric feature names and combined export names
                mlflow.log_dict(
                    {"feature_names": numeric_feature_names, "export_feature_names": combined_feature_names},
                    "preprocessor/feature_names.json",
                )
            except Exception as e:
                logger.warning(f"Could not log feature_names.json: {e}")

            # Log preprocessor artifacts
            if preprocessor is not None:
                try:
                    tmp_dir = tempfile.mkdtemp(prefix="realmlp_preproc_")
                    preproc_dir = os.path.join(tmp_dir, "preprocessor")
                    os.makedirs(preproc_dir, exist_ok=True)
                    preprocessor.save_artifacts(Path(preproc_dir))
                    # Log the entire directory
                    mlflow.log_artifacts(preproc_dir, artifact_path="preprocessor")
                except Exception as e:
                    logger.warning(f"Could not save/log preprocessor artifacts: {e}")
                finally:
                    try:
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                    except Exception:
                        pass

            # Log the PyTorch model
            try:
                # Temporarily move to CPU for reliable logging
                try:
                    original_device = next(self.model.parameters()).device
                except Exception:
                    original_device = torch.device("cpu")
                # self.model.to(torch.device("cpu"))
                if hasattr(self.model, "cat_embedding") and getattr(self.model, "cat_embedding") is not None:
                    mlflow.pytorch.log_model(
                        pytorch_model=self.model,
                        artifact_path="model",
                        signature=signature,
                    )
                else:
                    mlflow.pytorch.log_model(
                        pytorch_model=self.model,
                        artifact_path="model",
                        signature=signature,
                        input_example=input_example,
                    )
                # Optionally validate serving input example for clarity
                try:
                    from mlflow.models import validate_serving_input, convert_input_example_to_serving_input
                    serving_example = convert_input_example_to_serving_input(input_example)
                    validate_serving_input(f"runs:/{run.info.run_id}/model", serving_example)
                except Exception as _val_e:
                    logger.warning(f"Serving input validation warning: {_val_e}")
            except Exception as e:
                logger.error(f"Model logging failed: {e}")
                raise
            finally:
                try:
                    if original_device is not None:
                        self.model.to(original_device)
                except Exception:
                    pass

            # Persist local state: expose combined feature names to consumers, keep preprocessor for transforms
            self.feature_names = combined_feature_names
            if preprocessor is not None:
                self.preprocessor = preprocessor

            run_id = run.info.run_id
            logger.info(f"✅ RealMLP saved to MLflow. run_id={run_id}")
            return run_id

    def load_model(self, run_id: str, experiment_name: Optional[str] = None) -> bool:
        """
        Load RealMLP model and preprocessor artifacts from MLflow.
        """
        try:
            experiment = experiment_name or "realmlp_stock_predictor"
            mlflow.set_experiment(experiment)

            # Load torch model
            model_uri = f"runs:/{run_id}/model"
            self.model = mlflow.pytorch.load_model(model_uri)
            self.model.to(self.device)

            # Try to load preprocessor artifacts
            self.preprocessor = None
            self.feature_names = []
            try:
                downloaded_dir = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/preprocessor")
                # Load from the downloaded directory
                preproc = RealMLPPreprocessor().load_artifacts(Path(downloaded_dir))
                self.preprocessor = preproc
                # Prefer feature_names from artifact if available
                if getattr(preproc, "feature_names", None):
                    self.feature_names = list(preproc.feature_names)
                else:
                    # Fallback: look for explicit feature_names.json we logged
                    try:
                        with open(os.path.join(downloaded_dir, "feature_names.json"), "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if isinstance(data, list):
                            self.feature_names = data
                        elif isinstance(data, dict) and "feature_names" in data:
                            self.feature_names = data["feature_names"]
                    except Exception:
                        pass
                logger.info("✅ RealMLP preprocessor artifacts loaded from MLflow")
            except Exception as e:
                logger.info(f"No preprocessor artifacts found or failed to load: {e}")

            self.is_trained = True
            logger.info("✅ RealMLP model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading RealMLP model: {e}")
            return False


