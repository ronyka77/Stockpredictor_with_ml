from dataclasses import dataclass
from typing import List

from src.data_collector.polygon_fundamentals.data_models import FundamentalDataResponse
from src.data_collector.polygon_fundamentals.data_validator import (
    FundamentalDataValidator,
    ValidationResult,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExtendedValidation:
    base: ValidationResult
    cross_errors: List[str]
    cross_warnings: List[str]

    @property
    def is_valid(self) -> bool:
        return self.base.is_valid and not self.cross_errors

    @property
    def quality_score(self) -> float:
        return self.base.quality_score


class FundamentalDataValidatorV2:
    """Wraps V1 validator and adds cross-statement integrity rules."""

    def __init__(self) -> None:
        self.base = FundamentalDataValidator()

    def validate(self, resp: FundamentalDataResponse) -> ExtendedValidation:
        base = self.base.validate_response(resp)
        cross_errors: List[str] = []
        cross_warnings: List[str] = []

        inc = resp.get_latest_income_statement()
        bs = resp.get_latest_balance_sheet()
        cf = resp.get_latest_cash_flow()

        # Period alignment check (warning)
        if inc and bs:
            if (inc.end_date != bs.end_date) or (inc.fiscal_period != bs.fiscal_period):
                cross_warnings.append("Latest IS and BS period misalignment")
        if inc and cf:
            if (inc.end_date != cf.end_date) or (inc.fiscal_period != cf.fiscal_period):
                cross_warnings.append("Latest IS and CF period misalignment")

        # Basic assets ≈ liabilities + equity (warning level here; base may mark strong)
        try:
            if bs and bs.assets and bs.liabilities and bs.equity:
                assets = bs.assets.value or 0.0
                liabilities = bs.liabilities.value or 0.0
                equity = bs.equity.value or 0.0
                denom = abs(assets) if assets else 1.0
                if abs(assets - (liabilities + equity)) > 0.02 * denom:
                    cross_warnings.append("Balance sheet equation off by >2%")
        except Exception as e:
            logger.error(f"Error in balance sheet equation: {e}")
            pass

        # CF components sum (warning)
        try:
            if (
                cf
                and cf.net_cash_flow_from_operating_activities
                and cf.net_cash_flow_from_investing_activities
                and cf.net_cash_flow_from_financing_activities
                and cf.net_cash_flow
            ):
                op = cf.net_cash_flow_from_operating_activities.value or 0.0
                inv = cf.net_cash_flow_from_investing_activities.value or 0.0
                fin = cf.net_cash_flow_from_financing_activities.value or 0.0
                total = op + inv + fin
                net = cf.net_cash_flow.value or 0.0
                denom = abs(total) if total else 1.0
                if abs(net - total) > 0.05 * denom:
                    cross_warnings.append("Cash flow components don't sum within 5% tolerance")
        except Exception as e:
            logger.error(f"Error in cash flow components sum: {e}")
            pass

        return ExtendedValidation(
            base=base, cross_errors=cross_errors, cross_warnings=cross_warnings
        )
