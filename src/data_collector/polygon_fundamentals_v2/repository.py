from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional, List

from src.data_collector.polygon_fundamentals.db_pool import get_connection_pool
from src.utils.logger import get_logger


logger = get_logger(__name__)


RAW_STAGING_DDL = """
    CREATE TABLE IF NOT EXISTS raw_fundamental_json (
        id BIGSERIAL PRIMARY KEY,
        ticker_id INTEGER NOT NULL,
        period_end DATE,
        timeframe TEXT,
        fiscal_period TEXT,
        fiscal_year TEXT,
        filing_date DATE,
        source TEXT,
        payload_json JSONB NOT NULL,
        response_hash TEXT UNIQUE,
        ingested_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_raw_fund_json_ticker_end ON raw_fundamental_json(ticker_id, period_end DESC);
    CREATE INDEX IF NOT EXISTS idx_raw_fund_json_filing ON raw_fundamental_json(filing_date);
"""

FACTS_V2_DDL = """
    CREATE TABLE IF NOT EXISTS public.fundamental_facts_v2 (
        id BIGSERIAL PRIMARY KEY,
        ticker_id INTEGER NOT NULL,
        "date" DATE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        filing_date DATE NULL,
        fiscal_period VARCHAR(10) NULL,
        fiscal_year VARCHAR(10) NULL,
        timeframe VARCHAR(20) NULL,
        cik VARCHAR(20) NULL,
        company_name VARCHAR(255) NULL,
        source_filing_url TEXT NULL,
        source_filing_file_url TEXT NULL,
        acceptance_datetime TIMESTAMP NULL,
        sic_code VARCHAR(10) NULL,
        sic_description VARCHAR(255) NULL,
        revenues NUMERIC(15, 2) NULL,
        cost_of_revenue NUMERIC(15, 2) NULL,
        gross_profit NUMERIC(15, 2) NULL,
        operating_expenses NUMERIC(15, 2) NULL,
        selling_general_and_administrative_expenses NUMERIC(15, 2) NULL,
        research_and_development NUMERIC(15, 2) NULL,
        operating_income_loss NUMERIC(15, 2) NULL,
        nonoperating_income_loss NUMERIC(15, 2) NULL,
        income_loss_from_continuing_operations_before_tax NUMERIC(15, 2) NULL,
        income_tax_expense_benefit NUMERIC(15, 2) NULL,
        income_loss_from_continuing_operations_after_tax NUMERIC(15, 2) NULL,
        net_income_loss NUMERIC(15, 2) NULL,
        net_income_loss_attributable_to_parent NUMERIC(15, 2) NULL,
        basic_earnings_per_share NUMERIC(10, 4) NULL,
        diluted_earnings_per_share NUMERIC(10, 4) NULL,
        basic_average_shares NUMERIC(15, 2) NULL,
        diluted_average_shares NUMERIC(15, 2) NULL,
        assets NUMERIC(15, 2) NULL,
        current_assets NUMERIC(15, 2) NULL,
        noncurrent_assets NUMERIC(15, 2) NULL,
        inventory NUMERIC(15, 2) NULL,
        other_current_assets NUMERIC(15, 2) NULL,
        fixed_assets NUMERIC(15, 2) NULL,
        other_noncurrent_assets NUMERIC(15, 2) NULL,
        liabilities NUMERIC(15, 2) NULL,
        current_liabilities NUMERIC(15, 2) NULL,
        noncurrent_liabilities NUMERIC(15, 2) NULL,
        accounts_payable NUMERIC(15, 2) NULL,
        other_current_liabilities NUMERIC(15, 2) NULL,
        long_term_debt NUMERIC(15, 2) NULL,
        other_noncurrent_liabilities NUMERIC(15, 2) NULL,
        equity NUMERIC(15, 2) NULL,
        equity_attributable_to_parent NUMERIC(15, 2) NULL,
        net_cash_flow_from_operating_activities NUMERIC(15, 2) NULL,
        net_cash_flow_from_investing_activities NUMERIC(15, 2) NULL,
        net_cash_flow_from_financing_activities NUMERIC(15, 2) NULL,
        net_cash_flow NUMERIC(15, 2) NULL,
        net_cash_flow_continuing NUMERIC(15, 2) NULL,
        net_cash_flow_from_operating_activities_continuing NUMERIC(15, 2) NULL,
        net_cash_flow_from_investing_activities_continuing NUMERIC(15, 2) NULL,
        net_cash_flow_from_financing_activities_continuing NUMERIC(15, 2) NULL,
        comprehensive_income_loss NUMERIC(15, 2) NULL,
        comprehensive_income_loss_attributable_to_parent NUMERIC(15, 2) NULL,
        other_comprehensive_income_loss NUMERIC(15, 2) NULL,
        other_comprehensive_income_loss_attributable_to_parent NUMERIC(15, 2) NULL,
        data_quality_score NUMERIC(5, 4) NULL,
        missing_data_count INT DEFAULT 0 NULL,
        direct_report_fields_count INT DEFAULT 0 NULL,
        imputed_fields_count INT DEFAULT 0 NULL,
        derived_fields_count INT DEFAULT 0 NULL,
        total_fields_count INT DEFAULT 0 NULL,
        data_completeness_percentage NUMERIC(5, 2) NULL,
        completeness_score NUMERIC(5, 4) GENERATED ALWAYS AS (
            CASE WHEN total_fields_count = 0 THEN 0.0
                 ELSE (total_fields_count - missing_data_count)::NUMERIC / total_fields_count::NUMERIC
            END
        ) STORED,
        data_source_confidence NUMERIC(3, 2) NULL,
        CONSTRAINT fundamental_facts_v2_unique UNIQUE (ticker_id, date),
        CONSTRAINT fk_facts_v2_ticker_id FOREIGN KEY (ticker_id) REFERENCES public.tickers(id) ON DELETE CASCADE
    );
"""


class FundamentalsRepository:
    """DB access layer. Only SQL and transactions live here."""

    def __init__(self) -> None:
        self.pool = get_connection_pool()

    def ensure_schema(self) -> None:
        with self.pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(RAW_STAGING_DDL)
                cur.execute(FACTS_V2_DDL)
                conn.commit()
        logger.info("Ensured fundamentals V2 staging and facts schema")

    def get_ticker_id(self, ticker: str) -> Optional[int]:
        with self.pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM tickers WHERE ticker=%s", (ticker,))
                row = cur.fetchone()
                return int(row["id"]) if row and "id" in row else None

    def get_ticker_symbol(self, ticker_id: int) -> Optional[str]:
        with self.pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT ticker FROM tickers WHERE id=%s", (ticker_id,))
                row = cur.fetchone()
                return str(row["ticker"]) if row and "ticker" in row else None

    def set_ticker_financials(self, *, ticker_id: int, has_financials: bool) -> bool:
        """
        Update the `has_financials` flag for a given ticker id.
        Returns True if a row was updated, False otherwise. Handles errors with rollback and logging.
        """
        with self.pool.get_connection() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                            UPDATE public.tickers
                            SET has_financials = %s, last_updated = CURRENT_TIMESTAMP
                            WHERE id = %s
                        """,
                        (has_financials, ticker_id),
                    )
                    updated: int = getattr(cur, "rowcount", 0) or 0
                conn.commit()
                if updated == 0:
                    logger.warning(
                        f"No ticker updated for id={ticker_id} when setting has_financials={has_financials}"
                    )
                else:
                    logger.info(
                        f"Updated has_financials={has_financials} for ticker_id={ticker_id}"
                    )
                return updated > 0
            except Exception as e:  # noqa: BLE001
                try:
                    conn.rollback()
                except Exception:  # noqa: BLE001
                    pass
                logger.error(
                    f"Failed to update has_financials for ticker_id={ticker_id} to {has_financials}: {e}"
                )
                return False

    def fetch_pending_raw(self) -> List[Dict[str, Any]]:
        """
        Return ALL raw payload rows that are missing in facts or have fresher ingestion/filing than facts.updated_at/filing_date.
        """
        sql = """
            SELECT r.ticker_id,
                r.period_end,
                r.filing_date AS raw_filing_date,
                r.ingested_at,
                r.payload_json
            FROM raw_fundamental_json r
            LEFT JOIN public.fundamental_facts_v2 f
                ON f.ticker_id = r.ticker_id AND f.date = r.period_end
            WHERE f.id IS NULL
                OR f.updated_at IS NULL
                OR f.updated_at < r.ingested_at
                OR (r.filing_date IS NOT NULL AND (f.filing_date IS NULL OR f.filing_date < r.filing_date))
            ORDER BY r.ingested_at DESC
        """
        with self.pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall() or []
                logger.info(f"📊 Found {len(rows)} pending raw fundamental records to process")
                return rows

    def upsert_raw_payload(
        self,
        *,
        ticker_id: int,
        period_end: Optional[str],
        timeframe: Optional[str],
        fiscal_period: Optional[str],
        fiscal_year: Optional[str],
        filing_date: Optional[str],
        source: Optional[str],
        payload: Dict[str, Any],
    ) -> None:
        response_hash = self._hash_payload(payload)
        with self.pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                        INSERT INTO raw_fundamental_json (
                        ticker_id, period_end, timeframe, fiscal_period, fiscal_year, filing_date, source, payload_json, response_hash
                        ) VALUES (
                        %(ticker_id)s, %(period_end)s, %(timeframe)s, %(fiscal_period)s, %(fiscal_year)s, %(filing_date)s, %(source)s, %(payload)s, %(response_hash)s
                        )
                        ON CONFLICT (response_hash) DO UPDATE SET
                        payload_json = EXCLUDED.payload_json,
                        filing_date = EXCLUDED.filing_date;
                    """,
                    {
                        "ticker_id": ticker_id,
                        "period_end": period_end,
                        "timeframe": timeframe,
                        "fiscal_period": fiscal_period,
                        "fiscal_year": fiscal_year,
                        "filing_date": filing_date,
                        "source": source,
                        "payload": json.dumps(payload, default=str),
                        "response_hash": response_hash,
                    },
                )
                conn.commit()

    @staticmethod
    def _hash_payload(payload: Dict[str, Any]) -> str:
        data = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(data).hexdigest()


