"""
Evaluation Package

This package contains model evaluation utilities including custom metrics
for financial data, threshold-based evaluation, and model performance assessment tools.
"""

from .metrics import CustomMetrics
from .threshold_evaluator import ThresholdEvaluator, ModelProtocol

__all__ = ['CustomMetrics', 'ThresholdEvaluator', 'ModelProtocol'] 