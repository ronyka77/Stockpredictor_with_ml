"""
Fundamental Indicators Module

This module provides comprehensive fundamental analysis capabilities including:
- Financial ratios calculation
- Growth metrics analysis  
- Advanced scoring systems (Altman Z-Score, Piotroski F-Score)
- Sector analysis and cross-sectional comparisons
"""

from .base import (
    BaseFundamentalCalculator,
    FundamentalCalculationResult,
    FundamentalCalculatorRegistry
)

# Import all calculator modules to trigger decorator registration
# The decorators will automatically register calculators with FundamentalCalculatorRegistry
from . import ratios
from . import growth_metrics
from . import scoring_systems
from . import sector_analysis

# Export specific classes for direct import if needed
from .ratios import FundamentalRatiosCalculator
from .growth_metrics import GrowthMetricsCalculator
from .scoring_systems import ScoringSystemsCalculator
from .sector_analysis import SectorAnalysisCalculator

__all__ = [
    'BaseFundamentalCalculator',
    'FundamentalCalculationResult', 
    'FundamentalCalculatorRegistry',
    'FundamentalRatiosCalculator',
    'GrowthMetricsCalculator',
    'ScoringSystemsCalculator',
    'SectorAnalysisCalculator'
]

# DEPRECATED: Use FundamentalCalculatorRegistry.get_all_calculators() instead
# This is kept for backward compatibility but will be removed in future versions
def get_available_calculators():
    """
    Get all available calculators from the registry
    
    Returns:
        Dict[str, Type[BaseFundamentalCalculator]]: Dictionary of calculator name to class
    """
    import warnings
    warnings.warn(
        "get_available_calculators() is deprecated. Use FundamentalCalculatorRegistry.get_all_calculators() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return FundamentalCalculatorRegistry.get_all_calculators()

# For backward compatibility - this will be removed in future versions
# Note: This creates the dictionary when the module is imported, after decorators have run
def _get_available_calculators_dict():
    """Lazy getter for backward compatibility"""
    return FundamentalCalculatorRegistry.get_all_calculators()

# This will be populated after imports complete
AVAILABLE_CALCULATORS = None

def _populate_available_calculators():
    """Populate AVAILABLE_CALCULATORS after all imports are complete"""
    global AVAILABLE_CALCULATORS
    AVAILABLE_CALCULATORS = FundamentalCalculatorRegistry.get_all_calculators()

# Populate after imports
_populate_available_calculators() 