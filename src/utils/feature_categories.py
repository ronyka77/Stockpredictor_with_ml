"""
Centralized feature category taxonomy and helpers.

This module deduplicates the keyword matching logic used across the codebase
for classifying and filtering feature columns by category.
"""

from typing import Dict, List


# Canonical taxonomy: update here to propagate everywhere
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "trend": ["sma", "ema", "macd", "ichimoku"],
    "momentum": ["rsi", "stoch", "roc", "williams"],
    "volatility": ["bb", "bollinger", "atr", "volatility"],
    "volume": ["obv", "vpt", "ad_line", "volume", "mfi"],
}


def classify_feature_name(name: str) -> str:
    """Return the category for a given feature name based on keyword tokens.

    If no category is matched, returns 'basic'.
    """
    lowered = name.lower()
    for category, tokens in CATEGORY_KEYWORDS.items():
        if any(token in lowered for token in tokens):
            return category
    return "basic"


def filter_columns_by_categories(
    columns: List[str], categories: List[str]
) -> List[str]:
    """Filter a list of column names to those whose inferred category is selected.

    Args:
        columns: Column names to evaluate
        categories: Allowed categories (e.g., ["trend", "momentum"]).

    Returns:
        A list of column names whose classified category is in the provided set.
    """
    allowed = set(categories)
    return [col for col in columns if classify_feature_name(col) in allowed]
