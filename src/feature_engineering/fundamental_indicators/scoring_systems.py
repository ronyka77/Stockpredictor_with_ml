"""
Scoring Systems Calculator

This module implements advanced financial health and bankruptcy prediction scores
including Altman Z-Score, Piotroski F-Score, and other financial health metrics.
"""

from typing import Dict, List, Optional, Any
from datetime import date

from src.utils.logger import get_logger
from src.feature_engineering.fundamental_indicators.base import BaseFundamentalCalculator, FundamentalCalculationResult, FundamentalCalculatorRegistry
from src.data_collector.polygon_fundamentals.data_models import FundamentalDataResponse
from src.feature_engineering.data_loader import StockDataLoader

logger = get_logger(__name__)

@FundamentalCalculatorRegistry.register('scoring_systems')
class ScoringSystemsCalculator(BaseFundamentalCalculator):
    """Calculator for financial health and bankruptcy prediction scores"""
    
    def __init__(self, fundamental_config: Optional[Any] = None):
        """
        Initialize the scoring systems calculator
        
        Args:
            fundamental_config: Configuration for fundamental calculations
        """
        super().__init__(fundamental_config)
        self.data_loader = None
    
    def get_required_fields(self) -> Dict[str, List[str]]:
        """Get required fields for scoring calculations"""
        return {
            'income_statement': ['revenues', 'net_income_loss', 'operating_income_loss', 'cost_of_revenue', 'gross_profit'],
            'balance_sheet': ['assets', 'current_assets', 'current_liabilities', 'liabilities', 'equity', 'long_term_debt_noncurrent'],
            'cash_flow': ['net_cash_flow_from_operating_activities']
        }
    
    def get_market_data_from_loader(self, ticker: str) -> Dict[str, Optional[float]]:
        """
        Get market cap and employee data from the data loader
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with market_cap and total_employees
        """
        market_data = {
            'market_cap': None,
            'total_employees': None,
            'weighted_shares_outstanding': None
        }
        
        try:
            # Initialize data loader if not already done
            if self.data_loader is None:
                self.data_loader = StockDataLoader()
            
            # Get ticker metadata
            metadata = self.data_loader.get_ticker_metadata(ticker)
            
            if metadata:
                # Extract market cap (should be in USD)
                market_cap = metadata.get('market_cap')
                if market_cap is not None:
                    market_data['market_cap'] = float(market_cap)
                    logger.info(f"Retrieved market cap for {ticker}: ${market_cap:,.0f}")
                
                # Extract total employees
                total_employees = metadata.get('total_employees')
                if total_employees is not None:
                    market_data['total_employees'] = float(total_employees)
                    logger.info(f"Retrieved employee count for {ticker}: {total_employees:,.0f}")
                
                # Extract weighted shares outstanding
                shares = metadata.get('weighted_shares_outstanding')
                if shares is not None:
                    market_data['weighted_shares_outstanding'] = float(shares)
                    logger.info(f"Retrieved shares outstanding for {ticker}: {shares:,.0f}")
                
                logger.info(f"Successfully retrieved market data for {ticker}")
            else:
                logger.warning(f"No metadata found for ticker {ticker}")
                
        except Exception as e:
            logger.error(f"Error retrieving market data for {ticker}: {str(e)}")
            self.calculation_errors.append(f"Failed to retrieve market data: {str(e)}")
        
        return market_data
    
    def calculate(self, data: FundamentalDataResponse, pre_calculated_ratios: Optional[Any] = None) -> FundamentalCalculationResult:
        """Calculate all available scoring systems"""
        self.reset_quality_tracking()
        
        # Get ticker from the first available statement
        ticker = "UNKNOWN"
        if data.income_statements and len(data.income_statements) > 0:
            ticker = data.income_statements[0].ticker or "UNKNOWN"
        elif data.balance_sheets and len(data.balance_sheets) > 0:
            ticker = data.balance_sheets[0].ticker or "UNKNOWN"
        elif data.cash_flow_statements and len(data.cash_flow_statements) > 0:
            ticker = data.cash_flow_statements[0].ticker or "UNKNOWN"
        elif data.company_details:
            ticker = data.company_details.ticker or "UNKNOWN"
        
        if not self.validate_data(data):
            return FundamentalCalculationResult(
                ticker=ticker,
                date=date.today(),
                values={},
                quality_score=0.0,
                missing_data_count=self.missing_data_count,
                calculation_errors=self.calculation_errors
            )
        
        statements = self.get_latest_statements(data)
        periods = self.get_statements_by_period(data, periods=4)
        
        # Get market data from data loader
        market_data = self.get_market_data_from_loader(ticker)
        
        scores = {}
        
        # Bankruptcy Risk Scores
        scores.update(self._calculate_altman_z_score(statements, market_data))
        scores.update(self._calculate_ohlson_o_score(statements, market_data))
        # scores.update(self._calculate_zmijewski_score(statements))  # TODO: Missing interest expense
        
        # Piotroski F-Score and Individual Components
        piotroski_results = self._calculate_piotroski_f_score(statements, periods, market_data)
        scores.update(piotroski_results)
        
        # Quality and Financial Health Metrics
        scores.update(self._calculate_financial_leverage(statements, market_data))
        scores.update(self._calculate_working_capital_ratio(statements))
        scores.update(self._calculate_market_based_ratios(statements, market_data, pre_calculated_ratios))
        # scores.update(self._calculate_earnings_quality_score(statements, periods))  # TODO: Missing depreciation
        # scores.update(self._calculate_cash_conversion_ratio(statements, periods))  # TODO: Missing working capital details
        # scores.update(self._calculate_accruals_ratio(statements))  # TODO: Missing depreciation
        
        # Earnings Metrics (require more historical data)
        # scores.update(self._calculate_earnings_persistence(periods))  # TODO: Need more periods
        # scores.update(self._calculate_earnings_predictability(periods))  # TODO: Need more periods
        
        # Composite Scores
        scores.update(self._calculate_financial_health_composite(scores))
        scores.update(self._calculate_quality_composite(scores))
        scores.update(self._calculate_bankruptcy_risk_composite(scores))
        
        result = FundamentalCalculationResult(
            ticker=ticker,
            date=statements['income_statement'].filing_date if statements['income_statement'] else date.today(),
            values=scores,
            quality_score=self.calculate_quality_score(),
            missing_data_count=self.missing_data_count,
            calculation_errors=self.calculation_errors
        )
        
        return result
    
    def _calculate_altman_z_score(self, statements: Dict[str, Any], market_data: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
        """
        Calculate Altman Z-Score for bankruptcy prediction
        Formula: Z = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E
        Where: A=Working Capital/Total Assets, B=Retained Earnings/Total Assets,
                C=EBIT/Total Assets, D=Market Value Equity/Total Liabilities, E=Sales/Total Assets
        """
        scores = {}
        
        income_stmt = statements.get('income_statement')
        balance_sheet = statements.get('balance_sheet')
        
        if not income_stmt or not balance_sheet:
            return scores
        
        # Get required values
        current_assets = self.get_financial_value(balance_sheet.current_assets)
        current_liabilities = self.get_financial_value(balance_sheet.current_liabilities)
        total_assets = self.get_financial_value(balance_sheet.assets)
        total_liabilities = self.get_financial_value(balance_sheet.liabilities)
        operating_income = self.get_financial_value(income_stmt.operating_income_loss)
        revenues = self.get_financial_value(income_stmt.revenues)
        equity = self.get_financial_value(balance_sheet.equity)
        
        # Get market cap from data loader
        market_cap = market_data.get('market_cap')
        
        if not all([current_assets, current_liabilities, total_assets, total_liabilities]):
            return scores
        
        # Calculate components
        working_capital = current_assets - current_liabilities
        
        # Altman Z-Score components
        wc_ta = self.safe_divide(working_capital, total_assets)  # A
        ebit_ta = self.safe_divide(operating_income, total_assets) if operating_income else 0  # C
        
        # D component: Market Value Equity / Total Liabilities
        if market_cap and total_liabilities:
            # Use market cap as market value of equity
            mve_tl = self.safe_divide(market_cap, total_liabilities)
            logger.info(f"Using market cap for Altman Z-Score: Market Value/Total Liabilities = {mve_tl:.4f}")
        elif equity:
            # Fallback to book value of equity
            mve_tl = self.safe_divide(equity, total_liabilities)
            logger.info("Using book value of equity for Altman Z-Score (market cap not available)")
        else:
            mve_tl = 0
        
        s_ta = self.safe_divide(revenues, total_assets) if revenues else 0  # E
        
        # Calculate Z-Score (modified version without retained earnings component for now)
        if all(x is not None for x in [wc_ta, ebit_ta, mve_tl, s_ta]):
            # Modified formula without retained earnings component (will be corrected when data available)
            z_score = (1.2 * wc_ta + 3.3 * ebit_ta + 0.6 * mve_tl + 1.0 * s_ta)
            scores['altman_z_score'] = z_score
            
            # Log the calculation for debugging
            logger.info(f"Altman Z-Score calculation: WC/TA={wc_ta:.4f}, EBIT/TA={ebit_ta:.4f}, MVE/TL={mve_tl:.4f}, S/TA={s_ta:.4f}, Z-Score={z_score:.4f}")
        
        return scores
    
    def _calculate_ohlson_o_score(self, statements: Dict[str, Any], market_data: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
        """
        Calculate Ohlson O-Score for bankruptcy prediction
        
        O-Score = -1.32 - 0.407*log(TA) + 6.03*TL/TA - 1.43*WC/TA + 0.0757*CL/CA 
                    - 2.37*NI/TA - 1.83*FFO/TL + 0.285*INTWO - 1.72*OENEG - 0.521*CHIN
        
        Where:
        - TA = Total Assets
        - TL = Total Liabilities  
        - WC = Working Capital (Current Assets - Current Liabilities)
        - CA = Current Assets
        - CL = Current Liabilities
        - NI = Net Income
        - FFO = Funds from Operations (approx. Net Income + Depreciation)
        - INTWO = 1 if Net Income was negative for the last two years, 0 otherwise
        - OENEG = 1 if Total Liabilities > Total Assets, 0 otherwise
        - CHIN = (NI_t - NI_t-1) / (|NI_t| + |NI_t-1|)
        """
        import math
        
        scores = {}
        
        # Get current period statements
        balance_sheet = statements.get('balance_sheet')
        income_statement = statements.get('income_statement')
        cash_flow = statements.get('cash_flow')
        
        if not balance_sheet or not income_statement:
            logger.info("Insufficient data for Ohlson O-Score calculation")
            return scores
        
        try:
            # Extract financial values
            total_assets = self.get_financial_value(balance_sheet.assets)
            total_liabilities = self.get_financial_value(balance_sheet.liabilities)
            current_assets = self.get_financial_value(balance_sheet.current_assets)
            current_liabilities = self.get_financial_value(balance_sheet.current_liabilities)
            net_income = self.get_financial_value(income_statement.net_income_loss)
            
            # Check if we have minimum required data
            if not all([total_assets, total_liabilities, current_assets, current_liabilities, net_income]):
                logger.info("Missing required fields for Ohlson O-Score")
                return scores
            
            # Calculate intermediate ratios
            tl_ta = self.safe_divide(total_liabilities, total_assets)  # TL/TA
            working_capital = current_assets - current_liabilities
            wc_ta = self.safe_divide(working_capital, total_assets)    # WC/TA
            cl_ca = self.safe_divide(current_liabilities, current_assets)  # CL/CA
            ni_ta = self.safe_divide(net_income, total_assets)        # NI/TA
            
            # Funds from Operations approximation (Net Income + Depreciation)
            depreciation = 0
            if cash_flow:
                depreciation = self.get_financial_value(cash_flow.depreciation_depletion_and_amortization) or 0
            
            funds_from_operations = net_income + depreciation
            ffo_tl = self.safe_divide(funds_from_operations, total_liabilities)  # FFO/TL
            
            # OENEG: 1 if Total Liabilities > Total Assets (technically impossible but can indicate data issues)
            oeneg = 1 if total_liabilities > total_assets else 0
            
            # For INTWO and CHIN, we would need historical data
            # For now, set conservative defaults
            intwo = 0  # Assume not negative for two consecutive years
            chin = 0   # Assume no significant change in net income
            
            # Calculate Ohlson O-Score
            # O-Score = -1.32 - 0.407*log(TA) + 6.03*TL/TA - 1.43*WC/TA + 0.0757*CL/CA 
            #           - 2.37*NI/TA - 1.83*FFO/TL + 0.285*INTWO - 1.72*OENEG - 0.521*CHIN
            
            if all([tl_ta is not None, wc_ta is not None, cl_ca is not None, 
                    ni_ta is not None, ffo_tl is not None]):
                
                log_ta = math.log(total_assets) if total_assets > 0 else 0
                
                ohlson_o_score = (
                    -1.32 
                    - 0.407 * log_ta
                    + 6.03 * tl_ta
                    - 1.43 * wc_ta
                    + 0.0757 * cl_ca
                    - 2.37 * ni_ta
                    - 1.83 * ffo_tl
                    + 0.285 * intwo
                    - 1.72 * oeneg
                    - 0.521 * chin
                )
                
                scores['ohlson_o_score'] = ohlson_o_score
                
                logger.info(f"Ohlson O-Score calculated: {ohlson_o_score:.4f}")
                logger.info(f"  Components: TL/TA={tl_ta:.4f}, WC/TA={wc_ta:.4f}, NI/TA={ni_ta:.4f}")
            
        except Exception as e:
            logger.error(f"Error calculating Ohlson O-Score: {e}")
            
        return scores
    
    def _calculate_piotroski_f_score(self, statements: Dict[str, Any], periods: List[Dict[str, Any]], market_data: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
        """
        Calculate Piotroski F-Score (0-9 scale) and individual components
        """
        scores = {}
        
        income_stmt = statements.get('income_statement')
        balance_sheet = statements.get('balance_sheet')
        cash_flow = statements.get('cash_flow')
        
        if not income_stmt or not balance_sheet:
            return scores
        
        f_score = 0
        
        # Get current values
        net_income = self.get_financial_value(income_stmt.net_income_loss)
        operating_cf = self.get_financial_value(cash_flow.net_cash_flow_from_operating_activities) if cash_flow else None
        total_assets = self.get_financial_value(balance_sheet.assets)
        current_assets = self.get_financial_value(balance_sheet.current_assets)
        current_liabilities = self.get_financial_value(balance_sheet.current_liabilities)
        # total_liabilities = self.get_financial_value(balance_sheet.liabilities)
        revenues = self.get_financial_value(income_stmt.revenues)
        gross_profit = self.get_financial_value(income_stmt.gross_profit)
        
        # Get shares outstanding from market data
        shares_outstanding = market_data.get('weighted_shares_outstanding')
        
        # 1. Positive ROA (Net Income)
        roa_positive = net_income and net_income > 0
        if roa_positive:
            f_score += 1
        scores['piotroski_roa_positive'] = roa_positive
        
        # 2. Positive operating cash flow
        cfo_positive = operating_cf and operating_cf > 0
        if cfo_positive:
            f_score += 1
        scores['piotroski_cfo_positive'] = cfo_positive
        
        # 3. ROA improvement (year-over-year)
        roa_improved = False
        if len(periods) >= 2 and total_assets:
            current_roa = self.safe_divide(net_income, total_assets) if net_income else None
            
            prev_period = periods[1]
            prev_income = prev_period.get('income_statement')
            prev_balance = prev_period.get('balance_sheet')
            
            if prev_income and prev_balance:
                prev_net_income = self.get_financial_value(prev_income.net_income_loss)
                prev_assets = self.get_financial_value(prev_balance.assets)
                prev_roa = self.safe_divide(prev_net_income, prev_assets) if prev_net_income else None
                
                if current_roa and prev_roa and current_roa > prev_roa:
                    roa_improved = True
                    f_score += 1
        
        scores['piotroski_roa_improved'] = roa_improved
        
        # 4. CFO vs ROA (Quality of earnings)
        cfo_vs_roa = False
        if operating_cf and net_income and operating_cf > net_income:
            cfo_vs_roa = True
            f_score += 1
        scores['piotroski_cfo_vs_roa'] = cfo_vs_roa
        
        # 5. Long-term debt decreased
        debt_decreased = False
        if len(periods) >= 2:
            current_debt = self.get_financial_value(balance_sheet.long_term_debt_noncurrent) or 0
            
            prev_period = periods[1]
            prev_balance = prev_period.get('balance_sheet')
            
            if prev_balance:
                prev_debt = self.get_financial_value(prev_balance.long_term_debt_noncurrent) or 0
                if current_debt < prev_debt:
                    debt_decreased = True
                    f_score += 1
        
        scores['piotroski_debt_decreased'] = debt_decreased
        
        # 6. Current ratio improved
        current_ratio_improved = False
        if len(periods) >= 2 and current_assets and current_liabilities:
            current_ratio = self.safe_divide(current_assets, current_liabilities)
            
            prev_period = periods[1]
            prev_balance = prev_period.get('balance_sheet')
            
            if prev_balance:
                prev_current_assets = self.get_financial_value(prev_balance.current_assets)
                prev_current_liabilities = self.get_financial_value(prev_balance.current_liabilities)
                prev_current_ratio = self.safe_divide(prev_current_assets, prev_current_liabilities) if prev_current_assets and prev_current_liabilities else None
                
                if current_ratio and prev_current_ratio and current_ratio > prev_current_ratio:
                    current_ratio_improved = True
                    f_score += 1
        
        scores['piotroski_current_ratio_improved'] = current_ratio_improved
        
        # 7. Shares outstanding decreased (now using data from loader)
        shares_decreased = False
        if len(periods) >= 2 and shares_outstanding:
            # Get previous period shares from market data if available
            # For now, mark as not available since we don't have historical shares data
            # TODO: Enhance to get historical shares data from multiple periods
            shares_decreased = False  # Will be enhanced when historical market data is available
        
        scores['piotroski_shares_outstanding'] = shares_decreased
        
        # 8. Gross margin improved
        gross_margin_improved = False
        if len(periods) >= 2 and gross_profit and revenues:
            current_gross_margin = self.safe_divide(gross_profit, revenues)
            
            prev_period = periods[1]
            prev_income = prev_period.get('income_statement')
            
            if prev_income:
                prev_gross_profit = self.get_financial_value(prev_income.gross_profit)
                prev_revenues = self.get_financial_value(prev_income.revenues)
                prev_gross_margin = self.safe_divide(prev_gross_profit, prev_revenues) if prev_gross_profit and prev_revenues else None
                
                if current_gross_margin and prev_gross_margin and current_gross_margin > prev_gross_margin:
                    gross_margin_improved = True
                    f_score += 1
        
        scores['piotroski_gross_margin_improved'] = gross_margin_improved
        
        # 9. Asset turnover improved
        asset_turnover_improved = False
        if len(periods) >= 2 and revenues and total_assets:
            current_asset_turnover = self.safe_divide(revenues, total_assets)
            
            prev_period = periods[1]
            prev_income = prev_period.get('income_statement')
            prev_balance = prev_period.get('balance_sheet')
            
            if prev_income and prev_balance:
                prev_revenues = self.get_financial_value(prev_income.revenues)
                prev_assets = self.get_financial_value(prev_balance.assets)
                prev_asset_turnover = self.safe_divide(prev_revenues, prev_assets) if prev_revenues and prev_assets else None
                
                if current_asset_turnover and prev_asset_turnover and current_asset_turnover > prev_asset_turnover:
                    asset_turnover_improved = True
                    f_score += 1
        
        scores['piotroski_asset_turnover_improved'] = asset_turnover_improved
        
        scores['piotroski_f_score'] = f_score
        return scores
    
    def _calculate_financial_leverage(self, statements: Dict[str, Any], market_data: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
        """Calculate financial leverage ratios"""
        scores = {}
        
        balance_sheet = statements.get('balance_sheet')
        if not balance_sheet:
            return scores
        
        total_assets = self.get_financial_value(balance_sheet.assets)
        total_liabilities = self.get_financial_value(balance_sheet.liabilities)
        equity = self.get_financial_value(balance_sheet.equity)
        long_term_debt = self.get_financial_value(balance_sheet.long_term_debt_noncurrent) or 0
        market_cap = market_data.get('market_cap')
        
        # Traditional Financial Leverage = Total Assets / Total Equity
        if total_assets and equity:
            financial_leverage = self.safe_divide(total_assets, equity)
            scores['financial_leverage'] = financial_leverage
        
        # Market-Based Financial Leverage = (Market Cap + Total Debt) / Market Cap
        if market_cap and total_liabilities:
            market_based_leverage = self.safe_divide(market_cap + total_liabilities, market_cap)
            scores['market_based_financial_leverage'] = market_based_leverage
            logger.info(f"Market-based leverage: (Market Cap + Debt) / Market Cap = {market_based_leverage:.4f}")
        
        # Debt-to-Market-Cap Ratio
        if market_cap and long_term_debt:
            debt_to_market_cap = self.safe_divide(long_term_debt, market_cap)
            scores['debt_to_market_cap_ratio'] = debt_to_market_cap
        
        return scores

    def _calculate_working_capital_ratio(self, statements: Dict[str, Any]) -> Dict[str, Optional[float]]:
        """Calculate working capital ratio"""
        scores = {}
        
        balance_sheet = statements.get('balance_sheet')
        if not balance_sheet:
            return scores
        
        current_assets = self.get_financial_value(balance_sheet.current_assets)
        current_liabilities = self.get_financial_value(balance_sheet.current_liabilities)
        
        if current_assets and current_liabilities:
            # Working Capital Ratio = Current Assets / Current Liabilities
            working_capital_ratio = self.safe_divide(current_assets, current_liabilities)
            scores['working_capital_ratio'] = working_capital_ratio
        
        return scores

    def _calculate_market_based_ratios(self, statements: Dict[str, Any], market_data: Dict[str, Optional[float]], pre_calculated_ratios: Optional[Any] = None) -> Dict[str, Optional[float]]:
        """
        Use pre-calculated market-based financial ratios from the ratios calculator
        
        This method now uses ratios already calculated by the ratios calculator
        to avoid duplicate calculations and ensure consistency.
        
        Args:
            statements: Dictionary containing financial statements
            market_data: Dictionary containing market data (market_cap, etc.)
            pre_calculated_ratios: FundamentalCalculationResult from ratios calculator
            
        Returns:
            Dictionary containing market-based ratios from pre-calculated values
        """
        scores = {}
        
        # First priority: Use pre-calculated ratios from the ratios calculator
        if pre_calculated_ratios and hasattr(pre_calculated_ratios, 'values'):
            ratios_values = pre_calculated_ratios.values
            
            # Map ratio calculator fields to scoring system field names
            ratio_mapping = {
                'price_to_earnings_ratio': ratios_values.get('pe_ratio'),
                'price_to_book_ratio': ratios_values.get('pb_ratio'),
                'market_to_book_ratio': ratios_values.get('pb_ratio'),  # Same as P/B
                'price_to_sales_ratio': ratios_values.get('ps_ratio'),
                'ev_ebitda_ratio': ratios_values.get('ev_ebitda'),
                'peg_ratio': ratios_values.get('peg_ratio'),
                'roe_from_ratios': ratios_values.get('roe'),
                'roa_from_ratios': ratios_values.get('roa')
            }
            
            # Add non-null ratios to scores
            for key, value in ratio_mapping.items():
                if value is not None:
                    scores[key] = float(value)
            
            logger.info(f"Using pre-calculated market ratios: P/E={ratios_values.get('pe_ratio')}, P/B={ratios_values.get('pb_ratio')}, P/S={ratios_values.get('ps_ratio')}")
            return scores
        
        # Fallback: Try database lookup (legacy behavior)
        income_statement = statements.get('income_statement')
        if not income_statement or not hasattr(income_statement, 'ticker'):
            logger.info("No ticker available for market ratios lookup")
            return self._calculate_basic_market_ratios(statements, market_data)
        
        ticker = income_statement.ticker
        filing_date = income_statement.filing_date
        
        if not ticker or not filing_date:
            logger.info("Missing ticker or filing date for market ratios lookup")
            return self._calculate_basic_market_ratios(statements, market_data)
        
        try:
            # Initialize data loader to fetch ratios from database
            from src.feature_engineering.data_loader import StockDataLoader
            
            # Query to get the latest fundamental ratios for this ticker around the filing date
            query = """
            SELECT pe_ratio, pb_ratio, ps_ratio, ev_ebitda, peg_ratio, roe, roa
            FROM fundamental_ratios 
            WHERE ticker = %s 
            AND date <= %s
            ORDER BY date DESC 
            LIMIT 1
            """
            
            data_loader = StockDataLoader()
            results = data_loader.execute_query(query, [ticker, filing_date])
            
            if results:
                row = results[0]
                
                # Map database fields to our scoring system field names
                ratio_mapping = {
                    'price_to_earnings_ratio': row[0],  # pe_ratio
                    'price_to_book_ratio': row[1],      # pb_ratio
                    'market_to_book_ratio': row[1],     # pb_ratio (alternative name)
                    'price_to_sales_ratio': row[2],     # ps_ratio
                    'ev_ebitda_ratio': row[3],          # ev_ebitda
                    'peg_ratio': row[4],                # peg_ratio
                    'roe_from_db': row[5],              # roe (for comparison)
                    'roa_from_db': row[6]               # roa (for comparison)
                }
                
                # Only add non-null ratios to scores
                for key, value in ratio_mapping.items():
                    if value is not None:
                        scores[key] = float(value)
                
                logger.info(f"Retrieved market ratios from database for {ticker}: P/E={row[0]}, P/B={row[1]}, P/S={row[2]}")
                
            else:
                logger.warning(f"No fundamental ratios found in database for {ticker} on or before {filing_date}")
                # Final fallback: calculate basic ratios
                scores.update(self._calculate_basic_market_ratios(statements, market_data))
            
            data_loader.close()
            
        except Exception as e:
            logger.error(f"Error fetching market ratios from database: {e}")
            self.calculation_errors.append(f"Market ratios database error: {str(e)}")
            
            # Final fallback: calculate basic ratios
            scores.update(self._calculate_basic_market_ratios(statements, market_data))
        
        return scores

    def _calculate_basic_market_ratios(self, statements: Dict[str, Any], market_data: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
        """
        Fallback method to calculate basic market ratios when database lookup fails
        
        Args:
            statements: Dictionary containing financial statements
            market_data: Dictionary containing market data
            
        Returns:
            Dictionary containing calculated basic market ratios
        """
        scores = {}
        
        balance_sheet = statements.get('balance_sheet')
        income_statement = statements.get('income_statement')
        
        if not balance_sheet or not income_statement:
            return scores
        
        try:
            # Extract financial values
            net_income = self.get_financial_value(income_statement.net_income_loss)
            revenues = self.get_financial_value(income_statement.revenues)
            equity = self.get_financial_value(balance_sheet.equity)
            
            # Get market data
            market_cap = market_data.get('market_cap')
            
            # Calculate basic ratios as fallback
            if market_cap and net_income and net_income > 0:
                pe_ratio = self.safe_divide(market_cap, net_income)
                scores['price_to_earnings_ratio'] = pe_ratio
            
            if market_cap and equity and equity > 0:
                pb_ratio = self.safe_divide(market_cap, equity)
                scores['price_to_book_ratio'] = pb_ratio
                scores['market_to_book_ratio'] = pb_ratio
            
            if market_cap and revenues and revenues > 0:
                ps_ratio = self.safe_divide(market_cap, revenues)
                scores['price_to_sales_ratio'] = ps_ratio
            
        except Exception as e:
            logger.error(f"Error calculating fallback market ratios: {e}")
        
        return scores
    
    def _calculate_financial_health_composite(self, scores: Dict[str, Any]) -> Dict[str, Optional[float]]:
        """Calculate composite financial health score"""
        composite_scores = {}
        
        # Components for financial health composite
        components = []
        
        # Add Altman Z-Score (normalized)
        if 'altman_z_score' in scores and scores['altman_z_score'] is not None:
            # Normalize Altman Z-Score (>3.0 = healthy, <1.8 = distressed)
            z_score = scores['altman_z_score']
            normalized_z = min(1.0, max(0.0, (z_score - 1.8) / (3.0 - 1.8))) if z_score else 0.0
            components.append(normalized_z)
        
        # Add Ohlson O-Score (inverted and normalized - lower O-Score = better health)
        if 'ohlson_o_score' in scores and scores['ohlson_o_score'] is not None:
            # Normalize Ohlson O-Score (< -1.32 = healthy, > 0 = distressed)
            o_score = scores['ohlson_o_score']
            # Invert: lower O-Score = better health
            normalized_o = min(1.0, max(0.0, (-1.32 - o_score) / 1.32)) if o_score is not None else 0.0
            components.append(normalized_o)
        
        # Add Piotroski F-Score (already 0-9 scale, normalize to 0-1)
        if 'piotroski_f_score' in scores and scores['piotroski_f_score'] is not None:
            normalized_piotroski = scores['piotroski_f_score'] / 9.0
            components.append(normalized_piotroski)
        
        # Add Working Capital Ratio (normalize around 2.0 as healthy)
        if 'working_capital_ratio' in scores and scores['working_capital_ratio'] is not None:
            wc_ratio = scores['working_capital_ratio']
            normalized_wc = min(1.0, max(0.0, wc_ratio / 2.0)) if wc_ratio else 0.0
            components.append(normalized_wc)
        
        # Add Market-to-Book ratio consideration (reasonable P/B around 1-3)
        if 'market_to_book_ratio' in scores and scores['market_to_book_ratio'] is not None:
            mb_ratio = scores['market_to_book_ratio']
            # Normalize around 1-3 range (too low or too high both indicate issues)
            if mb_ratio <= 1:
                normalized_mb = mb_ratio  # Below 1 = undervalued but concerning
            elif mb_ratio <= 3:
                normalized_mb = 1.0  # Sweet spot
            else:
                normalized_mb = max(0.0, 1.0 - (mb_ratio - 3) / 5)  # Above 3 = potentially overvalued
            components.append(normalized_mb * 0.5)  # Lower weight for market metrics
        
        if components:
            financial_health_composite = sum(components) / len(components)
            composite_scores['financial_health_composite'] = financial_health_composite
        
        return composite_scores
    
    def _calculate_quality_composite(self, scores: Dict[str, Any]) -> Dict[str, Optional[float]]:
        """Calculate composite earnings quality score"""
        composite_scores = {}
        
        # Components for quality composite
        components = []
        
        # Add Piotroski CFO vs ROA
        if 'piotroski_cfo_vs_roa' in scores:
            components.append(1.0 if scores['piotroski_cfo_vs_roa'] else 0.0)
        
        # Add Piotroski ROA positive
        if 'piotroski_roa_positive' in scores:
            components.append(1.0 if scores['piotroski_roa_positive'] else 0.0)
        
        # Add Piotroski CFO positive
        if 'piotroski_cfo_positive' in scores:
            components.append(1.0 if scores['piotroski_cfo_positive'] else 0.0)
        
        # Add Piotroski gross margin improved
        if 'piotroski_gross_margin_improved' in scores:
            components.append(1.0 if scores['piotroski_gross_margin_improved'] else 0.0)
        
        # Add reasonable P/E ratio (earnings quality indicator)
        if 'price_to_earnings_ratio' in scores and scores['price_to_earnings_ratio'] is not None:
            pe_ratio = scores['price_to_earnings_ratio']
            # Reasonable P/E range 10-25 indicates quality earnings
            if 10 <= pe_ratio <= 25:
                pe_quality = 1.0
            elif pe_ratio < 10:
                pe_quality = pe_ratio / 10.0  # Low P/E might indicate issues
            else:
                pe_quality = max(0.0, 1.0 - (pe_ratio - 25) / 50)  # High P/E might indicate overvaluation
            components.append(pe_quality * 0.5)  # Lower weight for market-based quality
        
        if components:
            quality_composite = sum(components) / len(components)
            composite_scores['quality_composite'] = quality_composite
        
        return composite_scores
    
    def _calculate_bankruptcy_risk_composite(self, scores: Dict[str, Any]) -> Dict[str, Optional[float]]:
        """Calculate composite bankruptcy risk score"""
        composite_scores = {}
        
        # Components for bankruptcy risk composite
        components = []
        
        # Add Altman Z-Score (inverted - higher Z-Score = lower bankruptcy risk)
        if 'altman_z_score' in scores and scores['altman_z_score'] is not None:
            z_score = scores['altman_z_score']
            # Invert and normalize: higher Z-Score = lower risk
            normalized_risk = max(0.0, min(1.0, (3.0 - z_score) / (3.0 - 1.8))) if z_score else 1.0
            components.append(normalized_risk)
        
        # Add Ohlson O-Score (higher O-Score = higher bankruptcy risk)
        if 'ohlson_o_score' in scores and scores['ohlson_o_score'] is not None:
            o_score = scores['ohlson_o_score']
            # Normalize: higher O-Score = higher risk
            normalized_o_risk = min(1.0, max(0.0, (o_score + 1.32) / 2.64)) if o_score is not None else 0.5
            components.append(normalized_o_risk)
        
        # Add financial leverage (higher leverage = higher risk)
        if 'financial_leverage' in scores and scores['financial_leverage'] is not None:
            leverage = scores['financial_leverage']
            # Normalize around 2.0 as moderate risk
            normalized_leverage_risk = min(1.0, max(0.0, (leverage - 1.0) / 3.0)) if leverage else 0.0
            components.append(normalized_leverage_risk)
        
        # Add working capital ratio (lower ratio = higher risk)
        if 'working_capital_ratio' in scores and scores['working_capital_ratio'] is not None:
            wc_ratio = scores['working_capital_ratio']
            # Invert: lower working capital ratio = higher risk
            normalized_wc_risk = max(0.0, min(1.0, (2.0 - wc_ratio) / 2.0)) if wc_ratio else 1.0
            components.append(normalized_wc_risk)
        
        # Add market-based leverage risk
        if 'market_based_financial_leverage' in scores and scores['market_based_financial_leverage'] is not None:
            market_leverage = scores['market_based_financial_leverage']
            # Higher market-based leverage = higher risk
            normalized_market_leverage_risk = min(1.0, max(0.0, (market_leverage - 1.0) / 2.0))
            components.append(normalized_market_leverage_risk * 0.7)  # Lower weight for market metrics
        
        if components:
            bankruptcy_risk_composite = sum(components) / len(components)
            composite_scores['bankruptcy_risk_composite'] = bankruptcy_risk_composite
        
        return composite_scores