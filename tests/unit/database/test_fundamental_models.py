import pytest
from datetime import date
from unittest.mock import MagicMock

from src.database import fundamental_models as fm
from tests.fixtures.factories import (
    FundamentalRatiosFactory,
    FundamentalGrowthMetricsFactory,
    FundamentalScoresFactory,
    build_fundamental_sector,
)


@pytest.fixture
def db_session_mock():
    """Return a simple mock session with configurable query().first() and .all()."""
    session = MagicMock()
    return session


@pytest.mark.parametrize(
    "has_growth,has_scores,has_sector",
    [(True, True, True), (True, False, False), (False, False, False)],
)
def test_get_latest_fundamental_data_various_combinations(
    db_session_mock, has_growth, has_scores, has_sector
):
    """Return latest fundamental data combining optional growth/scores/sector sections"""
    ratios = FundamentalRatiosFactory.build(ticker="AAPL", date=date(2025, 1, 1))
    growth = (
        FundamentalGrowthMetricsFactory.build(ticker="AAPL", date=date(2025, 1, 1))
        if has_growth
        else None
    )
    scores = (
        FundamentalScoresFactory.build(ticker="AAPL", date=date(2025, 1, 1)) if has_scores else None
    )
    sector = (
        build_fundamental_sector.build(ticker="AAPL", date=date(2025, 1, 1)) if has_sector else None
    )

    # Configure session mock to return these objects in sequence for the four queries
    # The function under test queries ratios, growth, scores, sector in that order.
    db_session_mock.query.return_value.filter_by.return_value.order_by.return_value.first.side_effect = [
        ratios,
        growth,
        scores,
        sector,
    ]

    # Execution
    result = fm.get_latest_fundamental_data(db_session_mock, "AAPL")

    # Verification
    if not ratios:
        assert result is None
        return

    assert result["ticker"] == "AAPL"
    assert isinstance(result["ratios"], fm.FundamentalRatios)
    assert (result["growth"] is None) is (not has_growth) is False or True
    # Explicitly check presence/absence of optional sections
    if has_growth:
        assert isinstance(result["growth"], fm.FundamentalGrowthMetrics)
    else:
        assert result["growth"] is None

    if has_scores:
        assert isinstance(result["scores"], fm.FundamentalScores)
    else:
        assert result["scores"] is None

    if has_sector:
        assert isinstance(result["sector"], fm.FundamentalSectorAnalysis)
    else:
        assert result["sector"] is None


def test_get_fundamental_data_by_date_returns_none_when_no_ratios(db_session_mock):
    """Return None when no ratios exist for given date"""
    db_session_mock.query.return_value.filter_by.return_value.first.return_value = None

    # Execution
    result = fm.get_fundamental_data_by_date(db_session_mock, "MSFT", date(2024, 12, 31))

    # Verification
    assert result is None


def test_get_fundamental_data_by_date_returns_all_sections_when_present(db_session_mock):
    """Return all fundamental sections when ratios, growth, scores and sector present"""
    target = date(2025, 6, 30)
    ratios = FundamentalRatiosFactory.build(ticker="MSFT", date=target)
    growth = FundamentalGrowthMetricsFactory.build(ticker="MSFT", date=target)
    scores = FundamentalScoresFactory.build(ticker="MSFT", date=target)
    sector = build_fundamental_sector.build(ticker="MSFT", date=target)

    # Configure per-model filters: the test relies on the implementation calling query().filter_by(...).first()
    def first_side_effect(*args, **kwargs):
        # side effect will be consumed in order: ratios, growth, scores, sector
        return first_side_effect.values.pop(0)

    first_side_effect.values = [ratios, growth, scores, sector]
    db_session_mock.query.return_value.filter_by.return_value.first.side_effect = first_side_effect

    # Execution
    result = fm.get_fundamental_data_by_date(db_session_mock, "MSFT", target)

    # Verification
    assert result["ticker"] == "MSFT"
    assert result["date"] == target
    assert isinstance(result["ratios"], fm.FundamentalRatios)
    assert isinstance(result["growth"], fm.FundamentalGrowthMetrics)
    assert isinstance(result["scores"], fm.FundamentalScores)
    assert isinstance(result["sector"], fm.FundamentalSectorAnalysis)


def test_get_sector_companies_returns_list(db_session_mock):
    """Return list of FundamentalSectorAnalysis items for a sector and date"""
    target = date(2025, 1, 1)
    items = [build_fundamental_sector.build(gics_sector="Tech", date=target) for _ in range(3)]
    db_session_mock.query.return_value.filter_by.return_value.all.return_value = items

    # Execution
    result = fm.get_sector_companies(db_session_mock, "Tech", target)

    # Verification
    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(x, fm.FundamentalSectorAnalysis) for x in result)


@pytest.mark.parametrize(
    "missing,total,expected", [(0, 10, 1.0), (5, 10, 0.5), (10, 10, 0.0), (0, 0, 0.0)]
)
def test_calculate_data_quality_score_variants(missing, total, expected):
    """Calculate data quality score for various missing/total combinations"""
    result = fm.calculate_data_quality_score(missing, total)

    # Verification
    assert result == expected
