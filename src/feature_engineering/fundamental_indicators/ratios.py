"""
Financial Ratios Calculator

This module implements calculation of core financial ratios for database storage.
Only calculates the specific ratios defined in the database schema.
"""

from typing import Dict, List, Optional, Any
from datetime import date

from src.utils.logger import get_logger
from src.feature_engineering.fundamental_indicators.base import BaseFundamentalCalculator, FundamentalCalculationResult, FundamentalCalculatorRegistry
from src.data_collector.polygon_fundamentals.data_models import FundamentalDataResponse
from src.feature_engineering.data_loader import StockDataLoader

logger = get_logger(__name__)

@FundamentalCalculatorRegistry.register('ratios')
class FundamentalRatiosCalculator(BaseFundamentalCalculator):
    """Calculator for fundamental financial ratios - database schema focused"""
    
    def get_required_fields(self) -> Dict[str, List[str]]:
        """Get required fields for ratio calculations"""
        return {
            'income_statement': [
                'revenues', 'net_income_loss', 'gross_profit', 'operating_income_loss'
            ],
            'balance_sheet': [
                'assets', 'equity', 'current_assets', 'current_liabilities', 'liabilities'
            ],
            'cash_flow': []  # Optional for basic ratios
        }
    
    def _fetch_market_data(self, ticker: str) -> Dict[str, Optional[float]]:
        """
        Fetch market data from database using existing StockDataLoader
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with market data fields
        """
        market_data = {
            'current_stock_price': None,
            'market_cap': None,
            'shares_outstanding': None
        }
        
        try:
            # Use existing StockDataLoader to fetch market data
            with StockDataLoader() as data_loader:
                # Get ticker metadata (includes market_cap and shares_outstanding)
                metadata = data_loader.get_ticker_metadata(ticker)
                
                if metadata:
                    market_data['market_cap'] = metadata.get('market_cap')
                    market_data['shares_outstanding'] = metadata.get('weighted_shares_outstanding')
                    
                    self.logger.info(f"Retrieved metadata for {ticker}: market_cap={market_data['market_cap']}, shares={market_data['shares_outstanding']}")
                
                # Get latest stock price from historical data
                # Load recent data (last 5 days to ensure we get the latest available)
                end_date = date.today()
                start_date = '2024-01-01'
                
                price_data = data_loader.load_stock_data(ticker, start_date, end_date)
                
                if not price_data.empty:
                    # Get the most recent closing price
                    latest_price = price_data['close'].iloc[-1]
                    market_data['current_stock_price'] = float(latest_price)
                    
                    latest_date = price_data.index[-1].strftime('%Y-%m-%d')
                    self.logger.info(f"Retrieved latest price for {ticker}: ${latest_price:.2f} on {latest_date}")
                else:
                    self.logger.warning(f"No recent price data found for {ticker}")
                    
        except Exception as e:
            self.logger.error(f"Error fetching market data for {ticker}: {e}")
        
        return market_data
    
    def _validate_market_data(self, market_data: Dict[str, Optional[float]], ticker: str) -> bool:
        """
        Validate that market data was successfully retrieved
        
        Args:
            market_data: Dictionary with market data
            ticker: Stock ticker symbol
            
        Returns:
            True if sufficient market data is available
        """
        has_price = market_data.get('current_stock_price') is not None
        has_market_cap = market_data.get('market_cap') is not None
        has_shares = market_data.get('shares_outstanding') is not None
        
        self.logger.info(f"Market data validation for {ticker}: price={has_price}, market_cap={has_market_cap}, shares={has_shares}")
        
        if not any([has_price, has_market_cap, has_shares]):
            self.logger.warning(f"No market data available for {ticker}")
            return False
        
        if not has_price:
            self.logger.warning(f"Missing current stock price for {ticker}")
        if not has_market_cap:
            self.logger.warning(f"Missing market cap for {ticker}")
        if not has_shares:
            self.logger.warning(f"Missing shares outstanding for {ticker}")
            
        return True
    
    def calculate(self, data: FundamentalDataResponse) -> FundamentalCalculationResult:
        """
        Calculate financial ratios for database storage
        
        Args:
            data: FundamentalDataResponse containing financial statements
            
        Returns:
            FundamentalCalculationResult with calculated ratios
        """
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
            self.logger.warning(f"Ratios validation failed for {ticker}: {self.calculation_errors}")
            return FundamentalCalculationResult(
                ticker=ticker,
                date=date.today(),
                values={},
                quality_score=0.0,
                missing_data_count=self.missing_data_count,
                calculation_errors=self.calculation_errors
            )
        
        # Get latest statements
        statements = self.get_latest_statements(data)
        income_stmt = statements['income_statement']
        balance_sheet = statements['balance_sheet']
        
        # Fetch market data for market-based ratios
        market_data = self._fetch_market_data(ticker)
        
        # Validate market data (for logging purposes)
        self._validate_market_data(market_data, ticker)
        
        # Calculate only the required ratios
        ratios = {}
        
        # Market-based ratios (will be None without market data)
        ratios.update(self._calculate_market_ratios(income_stmt, balance_sheet, market_data, ticker, data))
        
        # Profitability ratios
        ratios.update(self._calculate_profitability_ratios(income_stmt, balance_sheet))
        
        # Liquidity ratios
        ratios.update(self._calculate_liquidity_ratios(balance_sheet))
        
        # Leverage ratios
        ratios.update(self._calculate_leverage_ratios(income_stmt, balance_sheet))
        
        # Apply outlier capping to all ratios
        for ratio_name, value in ratios.items():
            ratios[ratio_name] = self.apply_outlier_capping(value, ratio_name)
        
        result = FundamentalCalculationResult(
            ticker=ticker,
            date=income_stmt.filing_date if income_stmt else date.today(),
            values=ratios,
            quality_score=self.calculate_quality_score(),
            missing_data_count=self.missing_data_count,
            calculation_errors=self.calculation_errors
        )
        
        self.log_calculation_summary(result)
        return result
    
    def _calculate_market_ratios(self, income_stmt: Any, balance_sheet: Any, market_data: Dict[str, Optional[float]], ticker: str, data: FundamentalDataResponse) -> Dict[str, Optional[float]]:
        """
        Calculate market-based ratios using available market data
        
        Args:
            income_stmt: Income statement data
            balance_sheet: Balance sheet data  
            market_data: Dictionary with current_stock_price, market_cap, shares_outstanding
            ticker: Stock ticker symbol
            data: FundamentalDataResponse containing historical income statements
        """
        ratios = {}
        
        # Extract market data
        current_stock_price = market_data.get('current_stock_price')
        market_cap = market_data.get('market_cap')
        market_shares_outstanding = market_data.get('shares_outstanding')
        
        # Get fundamental data for calculations
        net_income = self.get_financial_value(income_stmt.net_income_loss) if income_stmt else None
        revenues = self.get_financial_value(income_stmt.revenues) if income_stmt else None
        operating_income = self.get_financial_value(income_stmt.operating_income_loss) if income_stmt else None
        equity = self.get_financial_value(balance_sheet.equity) if balance_sheet else None
        shares_outstanding = self.get_financial_value(income_stmt.weighted_average_shares_outstanding) if income_stmt else None
        total_liabilities = self.get_financial_value(balance_sheet.liabilities) if balance_sheet else None
        
        # Use market shares outstanding if available, otherwise use fundamental data
        effective_shares = market_shares_outstanding or shares_outstanding
        
        # PE Ratio = Current Price / EPS
        if current_stock_price and net_income and effective_shares:
            eps = self.safe_divide(net_income, effective_shares)
            if eps and eps > 0:
                ratios['pe_ratio'] = self.safe_divide(current_stock_price, eps)
            else:
                ratios['pe_ratio'] = None
        else:
            ratios['pe_ratio'] = None
        
        # PB Ratio = Current Price / Book Value per Share  
        if current_stock_price and equity and effective_shares:
            book_value_per_share = self.safe_divide(equity, effective_shares)
            if book_value_per_share and book_value_per_share > 0:
                ratios['pb_ratio'] = self.safe_divide(current_stock_price, book_value_per_share)
            else:
                ratios['pb_ratio'] = None
        else:
            ratios['pb_ratio'] = None
        
        # PS Ratio = Market Cap / Revenue (TTM)
        if market_cap and revenues and revenues > 0:
            ratios['ps_ratio'] = self.safe_divide(market_cap, revenues)
        else:
            ratios['ps_ratio'] = None
        
        # EV/EBITDA = Enterprise Value / EBITDA
        if market_cap and operating_income and total_liabilities:
            # Simplified EV calculation (missing cash and depreciation data)
            # enterprise_value = market_cap + total_liabilities - cash_and_equivalents
            # For now, use market_cap + total_liabilities as approximation
            enterprise_value = market_cap + total_liabilities
            
            # EBITDA approximation using operating income (missing depreciation/amortization)
            # ebitda = operating_income + depreciation_amortization
            # For now, use operating income as lower bound
            ebitda_approx = operating_income
            
            if ebitda_approx and ebitda_approx > 0:
                ratios['ev_ebitda'] = self.safe_divide(enterprise_value, ebitda_approx)
                # Add warning about approximation
                self.calculation_errors.append("EV/EBITDA calculated without cash and D&A data")
            else:
                ratios['ev_ebitda'] = None
        else:
            ratios['ev_ebitda'] = None
        
        # PEG Ratio = PE Ratio / Earnings Growth Rate
        # Note: Requires historical earnings data to calculate growth rate
        pe_ratio = ratios.get('pe_ratio')
        if pe_ratio:
            # Calculate earnings growth rate from historical data
            earnings_growth_rate = self._calculate_earnings_growth_rate(data, 4)
            
            if earnings_growth_rate and earnings_growth_rate > 0:
                # Convert growth rate to percentage for PEG calculation
                growth_rate_percentage = earnings_growth_rate * 100
                ratios['peg_ratio'] = self.safe_divide(pe_ratio, growth_rate_percentage)
                # self.logger.info(f"Calculated PEG ratio for {ticker}: {ratios['peg_ratio']:.2f}")
            else:
                ratios['peg_ratio'] = None
                self.calculation_errors.append("PEG ratio: earnings growth rate not available")
        else:
            ratios['peg_ratio'] = None
        
        return ratios
    
    def _calculate_profitability_ratios(self, income_stmt: Any, balance_sheet: Any) -> Dict[str, Optional[float]]:
        """Calculate profitability ratios"""
        ratios = {}
        
        if not income_stmt:
            return ratios
        
        # Get required values
        net_income = self.get_financial_value(income_stmt.net_income_loss)
        revenues = self.get_financial_value(income_stmt.revenues)
        gross_profit = self.get_financial_value(income_stmt.gross_profit)
        operating_income = self.get_financial_value(income_stmt.operating_income_loss)
        
        # Balance sheet values
        equity = self.get_financial_value(balance_sheet.equity) if balance_sheet else None
        assets = self.get_financial_value(balance_sheet.assets) if balance_sheet else None
        
        # Return on Equity (ROE) - Net Income / Shareholders' Equity
        if net_income and equity:
            ratios['roe'] = self.safe_divide(net_income, equity)
        
        # Return on Assets (ROA) - Net Income / Total Assets
        if net_income and assets:
            ratios['roa'] = self.safe_divide(net_income, assets)
        
        # Return on Investment (ROI) - using total assets as investment base
        if net_income and assets:
            ratios['roi'] = self.safe_divide(net_income, assets)
        
        # Gross Margin (Gross Profit / Revenue)
        if gross_profit and revenues:
            ratios['gross_margin'] = self.safe_divide(gross_profit, revenues)
        
        # Operating Margin (Operating Income / Revenue)
        if operating_income and revenues:
            ratios['operating_margin'] = self.safe_divide(operating_income, revenues)
        
        # Net Margin (Net Income / Revenue)
        if net_income and revenues:
            ratios['net_margin'] = self.safe_divide(net_income, revenues)
        
        return ratios
    
    def _calculate_liquidity_ratios(self, balance_sheet: Any) -> Dict[str, Optional[float]]:
        """Calculate liquidity ratios"""
        ratios = {}
        
        if not balance_sheet:
            return ratios
        
        # Get required values
        current_assets = self.get_financial_value(balance_sheet.current_assets)
        current_liabilities = self.get_financial_value(balance_sheet.current_liabilities)
        cash = self.get_financial_value(balance_sheet.cash_and_cash_equivalents_at_carrying_value)
        inventory = self.get_financial_value(balance_sheet.inventory_net)
        
        # Current Ratio (Current Assets / Current Liabilities)
        if current_assets and current_liabilities:
            ratios['current_ratio'] = self.safe_divide(current_assets, current_liabilities)
        
        # Quick Ratio ((Current Assets - Inventory) / Current Liabilities)
        if current_assets and current_liabilities:
            quick_assets = current_assets
            if inventory:
                quick_assets = current_assets - inventory
            ratios['quick_ratio'] = self.safe_divide(quick_assets, current_liabilities)
        
        # Cash Ratio (Cash / Current Liabilities)
        if cash and current_liabilities:
            ratios['cash_ratio'] = self.safe_divide(cash, current_liabilities)
        
        return ratios
    
    def _calculate_leverage_ratios(self, income_stmt: Any, balance_sheet: Any) -> Dict[str, Optional[float]]:
        """Calculate leverage ratios"""
        ratios = {}
        
        if not balance_sheet:
            return ratios
        
        # Get required values
        total_liabilities = self.get_financial_value(balance_sheet.liabilities)
        equity = self.get_financial_value(balance_sheet.equity)
        assets = self.get_financial_value(balance_sheet.assets)
        
        # Income statement values
        interest_expense = self.get_financial_value(income_stmt.interest_expense) if income_stmt else None
        operating_income = self.get_financial_value(income_stmt.operating_income_loss) if income_stmt else None
        
        # Debt-to-Equity Ratio (Total Liabilities / Shareholders' Equity)
        if total_liabilities and equity:
            ratios['debt_to_equity'] = self.safe_divide(total_liabilities, equity)
        
        # Debt-to-Assets Ratio (Total Liabilities / Total Assets)
        if total_liabilities and assets:
            ratios['debt_to_assets'] = self.safe_divide(total_liabilities, assets)
        
        # Interest Coverage Ratio (Operating Income / Interest Expense)
        if operating_income and interest_expense and interest_expense > 0:
            ratios['interest_coverage'] = self.safe_divide(operating_income, interest_expense)
        
        return ratios

    def _calculate_earnings_growth_rate(self, data: FundamentalDataResponse, periods: int = 4) -> Optional[float]:
        """
        Calculate earnings growth rate using historical fundamental data
        
        Args:
            data: FundamentalDataResponse containing historical income statements
            periods: Number of periods to look back for growth calculation
            
        Returns:
            Annualized earnings growth rate as decimal (e.g., 0.15 for 15%)
        """
        try:
            if not data.income_statements or len(data.income_statements) < 2:
                self.logger.info("Insufficient income statement data for growth calculation")
                return None
            
            # Get historical net income values (sorted by date, most recent first)
            earnings_data = []
            for stmt in data.income_statements[:periods]:
                net_income = self.get_financial_value(stmt.net_income_loss)
                if net_income is not None:
                    earnings_data.append({
                        'net_income': net_income,
                        'filing_date': stmt.filing_date,
                        'fiscal_period': stmt.fiscal_period,
                        'fiscal_year': stmt.fiscal_year
                    })
            
            if len(earnings_data) < 2:
                self.logger.info("Insufficient valid earnings data for growth calculation")
                return None
            
            # Sort by filing date (oldest first for CAGR calculation)
            earnings_data.sort(key=lambda x: x['filing_date'])
            
            beginning_earnings = earnings_data[0]['net_income']
            ending_earnings = earnings_data[-1]['net_income']
            
            # Calculate time period in years
            start_date = earnings_data[0]['filing_date']
            end_date = earnings_data[-1]['filing_date']
            time_diff = end_date - start_date
            years = time_diff.days / 365.25  # Account for leap years
            
            if years <= 0:
                self.logger.info("Invalid time period for growth calculation")
                return None
            
            # Handle negative earnings scenarios
            if beginning_earnings <= 0:
                if ending_earnings > 0:
                    # Turned profitable - use simple approach
                    self.logger.info("Company turned profitable - using simplified growth calculation")
                    return 1.0  # 100% growth (turned profitable)
                else:
                    # Both periods negative - can't calculate meaningful growth
                    self.logger.info("Negative earnings in both periods - cannot calculate growth")
                    return None
            
            # Calculate CAGR: (Ending Value / Beginning Value)^(1/years) - 1
            if ending_earnings <= 0:
                # Turned unprofitable
                self.logger.info("Company turned unprofitable")
                return -1.0  # -100% growth (turned unprofitable)
            
            cagr = (ending_earnings / beginning_earnings) ** (1 / years) - 1
            
            # Sanity check - cap extreme growth rates
            if abs(cagr) > 10.0:  # More than 1000% growth
                self.logger.warning(f"Extreme earnings growth rate calculated: {cagr:.2%}")
                return None
            
            # self.logger.info(f"Calculated earnings CAGR: {cagr:.2%} over {years:.1f} years")
            # self.logger.info(f"Earnings growth details: ${beginning_earnings/1e9:.1f}B -> ${ending_earnings/1e9:.1f}B")
            return cagr
            
        except Exception as e:
            self.logger.error(f"Error calculating earnings growth: {e}")
            return None

