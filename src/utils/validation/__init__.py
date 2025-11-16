"""
Validation Utilities Package

This package contains specialized validation utilities for the StockPredictor system,
including tensor shape validation for machine learning models.
"""

from .shape_validator import ShapeValidator

__all__ = ["ShapeValidator"]
