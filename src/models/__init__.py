"""
Models Package

This package contains machine learning models for stock prediction,
including gradient boosting models, ensemble strategies, and evaluation utilities.
"""

from .base_model import BaseModel
from .evaluation.metrics import CustomMetrics
from .gradient_boosting import (
    XGBoostModel, LightGBMModel, CatBoostModel, EnsembleModel,
    XGBoostHyperparameterConfig, LightGBMHyperparameterConfig, 
    CatBoostHyperparameterConfig, HyperparameterOptimizer
)

__all__ = [
    'BaseModel',
    'CustomMetrics',
    'XGBoostModel',
    'LightGBMModel', 
    'CatBoostModel',
    'EnsembleModel',
    'XGBoostHyperparameterConfig',
    'LightGBMHyperparameterConfig',
    'CatBoostHyperparameterConfig',
    'HyperparameterOptimizer'
]

__version__ = '1.0.0' 