def test_feature_calculator_integrates_components(mocker):
    """Verify FeatureCalculator merges component outputs into a combined result"""
    import pandas as pd
    from src.data_collector.indicator_pipeline.base import IndicatorResult
    from src.data_collector.indicator_pipeline.feature_calculator import (
        FeatureCalculator,
    )

    # Create simple price DataFrame
    idx = pd.date_range("2025-01-01", periods=5)
    price_df = pd.DataFrame(
        {
            "open": [1, 2, 3, 4, 5],
            "high": [2, 3, 4, 5, 6],
            "low": [1, 1, 2, 3, 4],
            "close": [1.5, 2.5, 3.5, 4.5, 5.5],
            "volume": [10, 20, 30, 40, 50],
        },
        index=idx,
    )

    # Simple result to return from each calculator
    simple_df = pd.DataFrame({"trend_sma": [1, 1, 1, 1, 1]}, index=idx)
    simple_result = IndicatorResult(
        data=simple_df,
        metadata={},
        quality_score=80.0,
        warnings=[],
        calculation_time=0.0,
    )

    # Patch BaseIndicator.validate_data to skip minimum data checks
    import src.data_collector.indicator_pipeline.base as base_mod

    mocker.patch.object(base_mod.BaseIndicator, "validate_data", return_value=None)

    # Patch all calculators to return the simple result
    mocker.patch(
        "src.data_collector.indicator_pipeline.trend_indicators.TrendIndicatorCalculator.calculate",
        return_value=simple_result,
    )
    mocker.patch(
        "src.data_collector.indicator_pipeline.momentum_indicators.MomentumIndicatorCalculator.calculate",
        return_value=simple_result,
    )
    mocker.patch(
        "src.data_collector.indicator_pipeline.volatility_indicators.VolatilityIndicatorCalculator.calculate",
        return_value=simple_result,
    )
    mocker.patch(
        "src.data_collector.indicator_pipeline.volume_indicators.VolumeIndicatorCalculator.calculate",
        return_value=simple_result,
    )

    fc = FeatureCalculator()
    result = fc.calculate_all_features(price_df)
    assert hasattr(result, "data")
    assert (
        "trend" in result.metadata.get("components", [])
        or len(result.metadata.get("components", [])) >= 1
    )
