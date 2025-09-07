

from typing import Any, Dict, Optional

from src.data_collector.polygon_fundamentals.data_models import (
    FundamentalDataResponse,
    IncomeStatement,
    BalanceSheet,
    CashFlowStatement,
    FinancialValue,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


class FundamentalsParser:
    """
    Translates raw Polygon fundamentals JSON (nested or legacy) into DTOs.
    Focuses on essential fields required for downstream correctness.
    """

    def parse(self, raw: Dict[str, Any], ticker: str) -> FundamentalDataResponse:
        response = FundamentalDataResponse(
            status=raw.get("status", "OK"),
            request_id=raw.get("request_id"),
            count=raw.get("count"),
            next_url=raw.get("next_url"),
        )

        results = raw.get("results") or []
        if not isinstance(results, list) or not results:
            logger.warning(f"No 'results' in fundamentals response for {ticker}")
            return response

        for r in results:
            fin = r.get("financials") or {}

            # metadata common
            meta = {
                "start_date": r.get("start_date"),
                "end_date": r.get("end_date"),
                "filing_date": r.get("filing_date"),
                "timeframe": r.get("timeframe"),
                "fiscal_period": r.get("fiscal_period"),
                "fiscal_year": r.get("fiscal_year"),
                "cik": r.get("cik"),
                "company_name": r.get("company_name"),
                "ticker": ticker,
                "source_filing_url": r.get("source_filing_url"),
                "source_filing_file_url": r.get("source_filing_file_url"),
            }

            # income
            inc_src = fin.get("income_statement") or {}
            if inc_src or any(k in r for k in ("revenues", "net_income_loss")):
                response.income_statements.append(
                    IncomeStatement(
                        **meta,
                        revenues=self._fv(self._pick(inc_src, r, "revenues")),
                        cost_of_revenue=self._fv(self._pick(inc_src, r, "cost_of_revenue")),
                        gross_profit=self._fv(self._pick(inc_src, r, "gross_profit")),
                        operating_expenses=self._fv(self._pick(inc_src, r, "operating_expenses")),
                        operating_income_loss=self._fv(self._pick(inc_src, r, "operating_income_loss")),
                        interest_expense=self._fv(self._pick(inc_src, r, "interest_expense")),
                        interest_income=self._fv(self._pick(inc_src, r, "interest_income")),
                        income_tax_expense_benefit=self._fv(
                            self._pick(inc_src, r, "income_tax_expense_benefit")
                        ),
                        net_income_loss=self._fv(self._pick(inc_src, r, "net_income_loss")),
                        earnings_per_share_basic=self._fv(
                            self._pick(inc_src, r, "earnings_per_share_basic")
                        ),
                        earnings_per_share_diluted=self._fv(
                            self._pick(inc_src, r, "earnings_per_share_diluted")
                        ),
                        weighted_average_shares_outstanding=self._fv(
                            self._pick(inc_src, r, "weighted_average_shares_outstanding")
                        ),
                        weighted_average_shares_outstanding_diluted=self._fv(
                            self._pick(inc_src, r, "weighted_average_shares_outstanding_diluted")
                        ),
                    )
                )

            # balance sheet
            bs_src = fin.get("balance_sheet") or {}
            if bs_src or any(k in r for k in ("assets", "equity")):
                response.balance_sheets.append(
                    BalanceSheet(
                        **meta,
                        assets=self._fv(self._pick(bs_src, r, "assets")),
                        current_assets=self._fv(self._pick(bs_src, r, "current_assets")),
                        noncurrent_assets=self._fv(self._pick(bs_src, r, "noncurrent_assets")),
                        liabilities=self._fv(self._pick(bs_src, r, "liabilities")),
                        current_liabilities=self._fv(self._pick(bs_src, r, "current_liabilities")),
                        noncurrent_liabilities=self._fv(
                            self._pick(bs_src, r, "noncurrent_liabilities")
                        ),
                        equity=self._fv(self._pick(bs_src, r, "equity")),
                        long_term_debt_noncurrent=self._fv(
                            self._pick(bs_src, r, "long_term_debt")
                        ),
                    )
                )

            # cash flow
            cf_src = fin.get("cash_flow_statement") or {}
            if cf_src or any(
                k in r for k in ("net_cash_flow_from_operating_activities", "net_cash_flow")
            ):
                response.cash_flow_statements.append(
                    CashFlowStatement(
                        **meta,
                        net_cash_flow_from_operating_activities=self._fv(
                            self._pick(cf_src, r, "net_cash_flow_from_operating_activities")
                        ),
                        net_cash_flow_from_investing_activities=self._fv(
                            self._pick(cf_src, r, "net_cash_flow_from_investing_activities")
                        ),
                        net_cash_flow_from_financing_activities=self._fv(
                            self._pick(cf_src, r, "net_cash_flow_from_financing_activities")
                        ),
                        net_cash_flow=self._fv(self._pick(cf_src, r, "net_cash_flow")),
                    )
                )

        # compute overall quality on essential fields only
        try:
            response.calculate_data_quality()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Quality calculation error: {e}")
        return response

    @staticmethod
    def _pick(nested: Dict[str, Any], legacy: Dict[str, Any], field: str) -> Optional[Any]:
        if isinstance(nested, dict) and field in nested:
            return nested[field]
        return legacy.get(field)

    @staticmethod
    def _fv(value: Optional[Any]) -> Optional[FinancialValue]:
        if value is None:
            return None
        if isinstance(value, dict):
            # accommodates {'value': x, 'unit': 'USD', 'source': 'direct_report', ...}
            try:
                return FinancialValue(**value)
            except Exception:  # noqa: BLE001
                # best-effort fallback
                if "value" in value:
                    return FinancialValue(value=value.get("value"))
                return None
        # primitives
        try:
            return FinancialValue(value=float(value))
        except Exception:  # noqa: BLE001
            return None


