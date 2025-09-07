from typing import Any, Dict, Optional, List

from src.data_collector.polygon_fundamentals_v2.parser import FundamentalsParser
from src.data_collector.polygon_fundamentals_v2.repository import FundamentalsRepository
from src.database.connection import execute
from src.utils.logger import get_logger


logger = get_logger(__name__)


class FundamentalsExtractor:
    """Extracts structured facts from raw_fundamental_json and upserts into public.fundamental_facts_v2."""

    def __init__(self) -> None:
        self.repo = FundamentalsRepository()
        self.parser = FundamentalsParser()

    def _flatten_value(self, obj: Optional[Dict[str, Any]]) -> Optional[float]:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get("value")
        if isinstance(obj, (int, float)):
            return float(obj)
        return None

    def _upsert_fact_row(
        self, ticker_id: int, meta: Dict[str, Any], row: Dict[str, Any]
    ) -> None:
        # Metadata fields
        params: Dict[str, Any] = {
            "ticker_id": ticker_id,
            "date": meta.get("end_date"),
            "filing_date": meta.get("filing_date"),
            "fiscal_period": meta.get("fiscal_period"),
            "fiscal_year": meta.get("fiscal_year"),
            "timeframe": meta.get("timeframe"),
            "source_filing_url": meta.get("source_filing_url"),
            "source_filing_file_url": meta.get("source_filing_file_url"),
            "acceptance_datetime": meta.get("acceptance_datetime"),
        }

        # Ensure all SQL placeholders exist (default to None), then overlay row values
        value_placeholders: List[str] = [
            # income
            "revenues",
            "cost_of_revenue",
            "gross_profit",
            "operating_expenses",
            "selling_general_and_administrative_expenses",
            "research_and_development",
            "operating_income_loss",
            "nonoperating_income_loss",
            "income_loss_from_continuing_operations_before_tax",
            "income_tax_expense_benefit",
            "income_loss_from_continuing_operations_after_tax",
            "net_income_loss",
            "net_income_loss_attributable_to_parent",
            "basic_earnings_per_share",
            "diluted_earnings_per_share",
            "basic_average_shares",
            "diluted_average_shares",
            # balance
            "assets",
            "current_assets",
            "noncurrent_assets",
            "inventory",
            "other_current_assets",
            "fixed_assets",
            "other_noncurrent_assets",
            "liabilities",
            "current_liabilities",
            "noncurrent_liabilities",
            "accounts_payable",
            "other_current_liabilities",
            "long_term_debt",
            "other_noncurrent_liabilities",
            "equity",
            "equity_attributable_to_parent",
            # cash flow
            "net_cash_flow_from_operating_activities",
            "net_cash_flow_from_investing_activities",
            "net_cash_flow_from_financing_activities",
            "net_cash_flow",
            "net_cash_flow_continuing",
            "net_cash_flow_from_operating_activities_continuing",
            "net_cash_flow_from_investing_activities_continuing",
            "net_cash_flow_from_financing_activities_continuing",
            # comprehensive income
            "comprehensive_income_loss",
            "comprehensive_income_loss_attributable_to_parent",
            "other_comprehensive_income_loss",
        ]
        for key in value_placeholders:
            if key not in params:
                params[key] = None
        params.update(row)

        sql = """
            INSERT INTO public.fundamental_facts_v2 (
                ticker_id, date, filing_date, fiscal_period, fiscal_year, timeframe,
                source_filing_url, source_filing_file_url, acceptance_datetime,
                revenues, cost_of_revenue, gross_profit, operating_expenses,
                selling_general_and_administrative_expenses, research_and_development,
                operating_income_loss, nonoperating_income_loss,
                income_loss_from_continuing_operations_before_tax, income_tax_expense_benefit,
                income_loss_from_continuing_operations_after_tax, net_income_loss,
                net_income_loss_attributable_to_parent, basic_earnings_per_share, diluted_earnings_per_share,
                basic_average_shares, diluted_average_shares,
                assets, current_assets, noncurrent_assets, inventory, other_current_assets, fixed_assets,
                other_noncurrent_assets, liabilities, current_liabilities, noncurrent_liabilities,
                accounts_payable, other_current_liabilities, long_term_debt, other_noncurrent_liabilities,
                equity, equity_attributable_to_parent,
                net_cash_flow_from_operating_activities, net_cash_flow_from_investing_activities,
                net_cash_flow_from_financing_activities, net_cash_flow, net_cash_flow_continuing,
                net_cash_flow_from_operating_activities_continuing, net_cash_flow_from_investing_activities_continuing,
                net_cash_flow_from_financing_activities_continuing,
                comprehensive_income_loss, comprehensive_income_loss_attributable_to_parent, other_comprehensive_income_loss
            ) VALUES (
                %(ticker_id)s, %(date)s, %(filing_date)s, %(fiscal_period)s, %(fiscal_year)s, %(timeframe)s,
                %(source_filing_url)s, %(source_filing_file_url)s, %(acceptance_datetime)s,
                %(revenues)s, %(cost_of_revenue)s, %(gross_profit)s, %(operating_expenses)s,
                %(selling_general_and_administrative_expenses)s, %(research_and_development)s,
                %(operating_income_loss)s, %(nonoperating_income_loss)s,
                %(income_loss_from_continuing_operations_before_tax)s, %(income_tax_expense_benefit)s,
                %(income_loss_from_continuing_operations_after_tax)s, %(net_income_loss)s,
                %(net_income_loss_attributable_to_parent)s, %(basic_earnings_per_share)s, %(diluted_earnings_per_share)s,
                %(basic_average_shares)s, %(diluted_average_shares)s,
                %(assets)s, %(current_assets)s, %(noncurrent_assets)s, %(inventory)s, %(other_current_assets)s, %(fixed_assets)s,
                %(other_noncurrent_assets)s, %(liabilities)s, %(current_liabilities)s, %(noncurrent_liabilities)s,
                %(accounts_payable)s, %(other_current_liabilities)s, %(long_term_debt)s, %(other_noncurrent_liabilities)s,
                %(equity)s, %(equity_attributable_to_parent)s,
                %(net_cash_flow_from_operating_activities)s, %(net_cash_flow_from_investing_activities)s,
                %(net_cash_flow_from_financing_activities)s, %(net_cash_flow)s, %(net_cash_flow_continuing)s,
                %(net_cash_flow_from_operating_activities_continuing)s, %(net_cash_flow_from_investing_activities_continuing)s,
                %(net_cash_flow_from_financing_activities_continuing)s,
                %(comprehensive_income_loss)s, %(comprehensive_income_loss_attributable_to_parent)s, %(other_comprehensive_income_loss)s
            )
            ON CONFLICT (ticker_id, date) DO UPDATE SET
                filing_date = EXCLUDED.filing_date,
                fiscal_period = EXCLUDED.fiscal_period,
                fiscal_year = EXCLUDED.fiscal_year,
                timeframe = EXCLUDED.timeframe,
                source_filing_url = EXCLUDED.source_filing_url,
                source_filing_file_url = EXCLUDED.source_filing_file_url,
                acceptance_datetime = EXCLUDED.acceptance_datetime,
                revenues = COALESCE(EXCLUDED.revenues, public.fundamental_facts_v2.revenues),
                cost_of_revenue = COALESCE(EXCLUDED.cost_of_revenue, public.fundamental_facts_v2.cost_of_revenue),
                gross_profit = COALESCE(EXCLUDED.gross_profit, public.fundamental_facts_v2.gross_profit),
                operating_expenses = COALESCE(EXCLUDED.operating_expenses, public.fundamental_facts_v2.operating_expenses),
                selling_general_and_administrative_expenses = COALESCE(EXCLUDED.selling_general_and_administrative_expenses, public.fundamental_facts_v2.selling_general_and_administrative_expenses),
                research_and_development = COALESCE(EXCLUDED.research_and_development, public.fundamental_facts_v2.research_and_development),
                operating_income_loss = COALESCE(EXCLUDED.operating_income_loss, public.fundamental_facts_v2.operating_income_loss),
                nonoperating_income_loss = COALESCE(EXCLUDED.nonoperating_income_loss, public.fundamental_facts_v2.nonoperating_income_loss),
                income_loss_from_continuing_operations_before_tax = COALESCE(EXCLUDED.income_loss_from_continuing_operations_before_tax, public.fundamental_facts_v2.income_loss_from_continuing_operations_before_tax),
                income_tax_expense_benefit = COALESCE(EXCLUDED.income_tax_expense_benefit, public.fundamental_facts_v2.income_tax_expense_benefit),
                income_loss_from_continuing_operations_after_tax = COALESCE(EXCLUDED.income_loss_from_continuing_operations_after_tax, public.fundamental_facts_v2.income_loss_from_continuing_operations_after_tax),
                net_income_loss = COALESCE(EXCLUDED.net_income_loss, public.fundamental_facts_v2.net_income_loss),
                net_income_loss_attributable_to_parent = COALESCE(EXCLUDED.net_income_loss_attributable_to_parent, public.fundamental_facts_v2.net_income_loss_attributable_to_parent),
                basic_earnings_per_share = COALESCE(EXCLUDED.basic_earnings_per_share, public.fundamental_facts_v2.basic_earnings_per_share),
                diluted_earnings_per_share = COALESCE(EXCLUDED.diluted_earnings_per_share, public.fundamental_facts_v2.diluted_earnings_per_share),
                basic_average_shares = COALESCE(EXCLUDED.basic_average_shares, public.fundamental_facts_v2.basic_average_shares),
                diluted_average_shares = COALESCE(EXCLUDED.diluted_average_shares, public.fundamental_facts_v2.diluted_average_shares),
                assets = COALESCE(EXCLUDED.assets, public.fundamental_facts_v2.assets),
                current_assets = COALESCE(EXCLUDED.current_assets, public.fundamental_facts_v2.current_assets),
                noncurrent_assets = COALESCE(EXCLUDED.noncurrent_assets, public.fundamental_facts_v2.noncurrent_assets),
                inventory = COALESCE(EXCLUDED.inventory, public.fundamental_facts_v2.inventory),
                other_current_assets = COALESCE(EXCLUDED.other_current_assets, public.fundamental_facts_v2.other_current_assets),
                fixed_assets = COALESCE(EXCLUDED.fixed_assets, public.fundamental_facts_v2.fixed_assets),
                other_noncurrent_assets = COALESCE(EXCLUDED.other_noncurrent_assets, public.fundamental_facts_v2.other_noncurrent_assets),
                liabilities = COALESCE(EXCLUDED.liabilities, public.fundamental_facts_v2.liabilities),
                current_liabilities = COALESCE(EXCLUDED.current_liabilities, public.fundamental_facts_v2.current_liabilities),
                noncurrent_liabilities = COALESCE(EXCLUDED.noncurrent_liabilities, public.fundamental_facts_v2.noncurrent_liabilities),
                accounts_payable = COALESCE(EXCLUDED.accounts_payable, public.fundamental_facts_v2.accounts_payable),
                other_current_liabilities = COALESCE(EXCLUDED.other_current_liabilities, public.fundamental_facts_v2.other_current_liabilities),
                long_term_debt = COALESCE(EXCLUDED.long_term_debt, public.fundamental_facts_v2.long_term_debt),
                other_noncurrent_liabilities = COALESCE(EXCLUDED.other_noncurrent_liabilities, public.fundamental_facts_v2.other_noncurrent_liabilities),
                equity = COALESCE(EXCLUDED.equity, public.fundamental_facts_v2.equity),
                equity_attributable_to_parent = COALESCE(EXCLUDED.equity_attributable_to_parent, public.fundamental_facts_v2.equity_attributable_to_parent),
                net_cash_flow_from_operating_activities = COALESCE(EXCLUDED.net_cash_flow_from_operating_activities, public.fundamental_facts_v2.net_cash_flow_from_operating_activities),
                net_cash_flow_from_investing_activities = COALESCE(EXCLUDED.net_cash_flow_from_investing_activities, public.fundamental_facts_v2.net_cash_flow_from_investing_activities),
                net_cash_flow_from_financing_activities = COALESCE(EXCLUDED.net_cash_flow_from_financing_activities, public.fundamental_facts_v2.net_cash_flow_from_financing_activities),
                net_cash_flow = COALESCE(EXCLUDED.net_cash_flow, public.fundamental_facts_v2.net_cash_flow),
                net_cash_flow_continuing = COALESCE(EXCLUDED.net_cash_flow_continuing, public.fundamental_facts_v2.net_cash_flow_continuing),
                net_cash_flow_from_operating_activities_continuing = COALESCE(EXCLUDED.net_cash_flow_from_operating_activities_continuing, public.fundamental_facts_v2.net_cash_flow_from_operating_activities_continuing),
                net_cash_flow_from_investing_activities_continuing = COALESCE(EXCLUDED.net_cash_flow_from_investing_activities_continuing, public.fundamental_facts_v2.net_cash_flow_from_investing_activities_continuing),
                net_cash_flow_from_financing_activities_continuing = COALESCE(EXCLUDED.net_cash_flow_from_financing_activities_continuing, public.fundamental_facts_v2.net_cash_flow_from_financing_activities_continuing),
                comprehensive_income_loss = COALESCE(EXCLUDED.comprehensive_income_loss, public.fundamental_facts_v2.comprehensive_income_loss),
                comprehensive_income_loss_attributable_to_parent = COALESCE(EXCLUDED.comprehensive_income_loss_attributable_to_parent, public.fundamental_facts_v2.comprehensive_income_loss_attributable_to_parent),
                other_comprehensive_income_loss = COALESCE(EXCLUDED.other_comprehensive_income_loss, public.fundamental_facts_v2.other_comprehensive_income_loss),
                updated_at = CURRENT_TIMESTAMP
        """

        # Use centralized helper to execute and commit
        execute(sql, params)

    def extract_from_payload(self, ticker: str, raw: Dict[str, Any]) -> Dict[str, Any]:
        ticker_id = self.repo.get_ticker_id(ticker)
        if ticker_id is None:
            logger.warning(f"Ticker not found: {ticker}")
            return {"ticker": ticker, "success": False, "error": "unknown_ticker"}

        dto = self.parser.parse(raw, ticker)

        count = 0
        for stmt in dto.income_statements:
            meta = stmt.model_dump()
            row: Dict[str, Any] = {
                "revenues": self._flatten_value(meta.get("revenues")),
                "cost_of_revenue": self._flatten_value(meta.get("cost_of_revenue")),
                "gross_profit": self._flatten_value(meta.get("gross_profit")),
                "operating_expenses": self._flatten_value(
                    meta.get("operating_expenses")
                ),
                "selling_general_and_administrative_expenses": self._flatten_value(
                    meta.get("selling_general_and_administrative_expenses")
                ),
                "research_and_development": self._flatten_value(
                    meta.get("research_and_development")
                ),
                "operating_income_loss": self._flatten_value(
                    meta.get("operating_income_loss")
                ),
                "nonoperating_income_loss": self._flatten_value(
                    meta.get("other_income_expense")
                ),
                "income_loss_from_continuing_operations_before_tax": self._flatten_value(
                    meta.get("income_loss_before_income_tax_expense_benefit")
                ),
                "income_tax_expense_benefit": self._flatten_value(
                    meta.get("income_tax_expense_benefit")
                ),
                "income_loss_from_continuing_operations_after_tax": None,
                "net_income_loss": self._flatten_value(meta.get("net_income_loss")),
                "net_income_loss_attributable_to_parent": self._flatten_value(
                    meta.get("net_income_loss_attributable_to_parent")
                ),
                "basic_earnings_per_share": self._flatten_value(
                    meta.get("earnings_per_share_basic")
                ),
                "diluted_earnings_per_share": self._flatten_value(
                    meta.get("earnings_per_share_diluted")
                ),
                "basic_average_shares": self._flatten_value(
                    meta.get("weighted_average_shares_outstanding")
                ),
                "diluted_average_shares": self._flatten_value(
                    meta.get("weighted_average_shares_outstanding_diluted")
                ),
                "comprehensive_income_loss": self._flatten_value(
                    meta.get("comprehensive_income_loss")
                ),
                "comprehensive_income_loss_attributable_to_parent": self._flatten_value(
                    meta.get("comprehensive_income_loss_attributable_to_parent")
                ),
            }
            self._upsert_fact_row(ticker_id, meta, row)
            count += 1

        for stmt in dto.balance_sheets:
            meta = stmt.model_dump()
            row = {
                "assets": self._flatten_value(meta.get("assets")),
                "current_assets": self._flatten_value(meta.get("current_assets")),
                "noncurrent_assets": self._flatten_value(meta.get("noncurrent_assets")),
                "inventory": self._flatten_value(meta.get("inventory_net")),
                "other_current_assets": self._flatten_value(
                    meta.get("other_assets_current")
                ),
                "fixed_assets": self._flatten_value(
                    meta.get("property_plant_equipment_net")
                ),
                "other_noncurrent_assets": self._flatten_value(
                    meta.get("other_assets_noncurrent")
                ),
                "liabilities": self._flatten_value(meta.get("liabilities")),
                "current_liabilities": self._flatten_value(
                    meta.get("current_liabilities")
                ),
                "noncurrent_liabilities": self._flatten_value(
                    meta.get("noncurrent_liabilities")
                ),
                "accounts_payable": self._flatten_value(
                    meta.get("accounts_payable_current")
                ),
                "other_current_liabilities": self._flatten_value(
                    meta.get("other_liabilities_current")
                ),
                "long_term_debt": self._flatten_value(
                    meta.get("long_term_debt_noncurrent")
                ),
                "other_noncurrent_liabilities": self._flatten_value(
                    meta.get("other_liabilities_noncurrent")
                ),
                "equity": self._flatten_value(meta.get("equity")),
                "equity_attributable_to_parent": self._flatten_value(
                    meta.get("equity_attributable_to_parent")
                ),
            }
            self._upsert_fact_row(ticker_id, meta, row)
            count += 1

        for stmt in dto.cash_flow_statements:
            meta = stmt.model_dump()
            row = {
                "net_cash_flow_from_operating_activities": self._flatten_value(
                    meta.get("net_cash_flow_from_operating_activities")
                ),
                "net_cash_flow_from_investing_activities": self._flatten_value(
                    meta.get("net_cash_flow_from_investing_activities")
                ),
                "net_cash_flow_from_financing_activities": self._flatten_value(
                    meta.get("net_cash_flow_from_financing_activities")
                ),
                "net_cash_flow": self._flatten_value(meta.get("net_cash_flow")),
            }
            self._upsert_fact_row(ticker_id, meta, row)
            count += 1

        return {"ticker": ticker, "success": True, "rows_upserted": count}

    def extract_pending_from_db(self) -> Dict[str, Any]:
        """Load ALL raw JSON payloads that are missing/outdated in facts and upsert them sequentially."""
        pending = self.repo.fetch_pending_raw()
        processed = 0
        errors: List[str] = []
        for row in pending:
            try:
                ticker_id = int(row["ticker_id"]) if "ticker_id" in row else None
                payload = row.get("payload_json")
                if not ticker_id or not payload:
                    continue
                ticker = self.repo.get_ticker_symbol(ticker_id) or ""
                if not ticker:
                    continue
                self.extract_from_payload(ticker, payload)
                processed += 1
            except Exception as e:  # noqa: BLE001
                errors.append(str(e))
        return {"found": len(pending), "processed": processed, "errors": errors}


if __name__ == "__main__":
    import json
    from datetime import datetime

    extractor = FundamentalsExtractor()
    extractor.repo.ensure_schema()
    result = extractor.extract_pending_from_db()
    logger.info(
        json.dumps(
            {
                "status": "ok",
                "found": result.get("found"),
                "processed": result.get("processed"),
                "errors": result.get("errors"),
                "timestamp": datetime.now().isoformat(),
            },
            default=str,
        )
    )
