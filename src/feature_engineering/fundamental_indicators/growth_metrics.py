"""
Growth Metrics Calculator

This module implements calculation of growth rates and efficiency metrics
for database storage. Only calculates the specific metrics defined in the database schema.
"""

from typing import Dict, List, Optional, Any
from datetime import date

from src.utils.logger import get_logger
from src.feature_engineering.fundamental_indicators.base import BaseFundamentalCalculator, FundamentalCalculationResult, FundamentalCalculatorRegistry
from src.data_collector.polygon_fundamentals.data_models import FundamentalDataResponse

logger = get_logger(__name__)

@FundamentalCalculatorRegistry.register('growth_metrics')
class GrowthMetricsCalculator(BaseFundamentalCalculator):
    """Calculator for growth rates and efficiency metrics - database schema focused"""
    
    def get_required_fields(self) -> Dict[str, List[str]]:
        """Get required fields for growth calculations"""
        return {
            'income_statement': [
                'revenues', 'net_income_loss', 'cost_of_revenue'
            ],
            'balance_sheet': [
                'assets', 'equity', 'current_assets', 'current_liabilities', 'inventory_net'
            ],
            'cash_flow': []  # Optional for basic growth metrics
        }
    
    def calculate(self, data: FundamentalDataResponse) -> FundamentalCalculationResult:
        """
        Calculate growth metrics for database storage
        
        Args:
            data: FundamentalDataResponse containing financial statements
            
        Returns:
            FundamentalCalculationResult with calculated growth metrics
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
            self.logger.warning(f"Growth metrics validation failed for {ticker}: {self.calculation_errors}")
            return FundamentalCalculationResult(
                ticker=ticker,
                date=date.today(),
                values={},
                quality_score=0.0,
                missing_data_count=self.missing_data_count,
                calculation_errors=self.calculation_errors
            )
        
        # Get multiple periods for growth calculations
        periods = self.get_statements_by_period(data, periods=12)  # Get up to 12 quarters for 3-year calculations
        
        if len(periods) < 2:
            self.calculation_errors.append("Insufficient historical data for growth calculations")
            return FundamentalCalculationResult(
                ticker=ticker,
                date=date.today(),
                values={},
                quality_score=0.0,
                missing_data_count=self.missing_data_count,
                calculation_errors=self.calculation_errors
            )
        
        # Calculate all growth metrics
        growth_metrics = {}
        
        # Revenue Growth Metrics
        growth_metrics.update(self._calculate_revenue_growth(periods))
        
        # Earnings Growth Metrics  
        growth_metrics.update(self._calculate_earnings_growth(periods))
        
        # Book Value Growth Metrics
        growth_metrics.update(self._calculate_book_value_growth(periods))
        
        # Asset Growth Metrics
        growth_metrics.update(self._calculate_asset_growth(periods))
        
        # Efficiency Metrics
        growth_metrics.update(self._calculate_efficiency_metrics(periods))
        
        # Apply outlier capping to all metrics
        for metric_name, value in growth_metrics.items():
            growth_metrics[metric_name] = self.apply_outlier_capping(value, metric_name)
        
        result = FundamentalCalculationResult(
            ticker=ticker,
            date=periods[0]['date'] if periods else date.today(),
            values=growth_metrics,
            quality_score=self.calculate_quality_score(),
            missing_data_count=self.missing_data_count,
            calculation_errors=self.calculation_errors
        )
        
        self.log_calculation_summary(result)
        return result
    
    def _calculate_revenue_growth(self, periods: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
        """Calculate revenue growth rates (1Y and 3Y)"""
        growth_metrics = {}
        
        # Extract revenue data from periods
        revenues = []
        for period in periods:
            income_stmt = period.get('income_statement')
            if income_stmt:
                revenue = self.get_financial_value(income_stmt.revenues)
                if revenue is not None:
                    revenues.append({
                        'value': revenue,
                        'date': period['date']
                    })
        
        if len(revenues) < 2:
            self.logger.debug("Insufficient revenue data for growth calculations")
            growth_metrics['revenue_growth_1y'] = None
            growth_metrics['revenue_growth_3y'] = None
            return growth_metrics
        
        # Sort by date (oldest first)
        revenues.sort(key=lambda x: x['date'])
        
        # 1-Year Revenue Growth (4 quarters)
        if len(revenues) >= 4:
            beginning_revenue = revenues[0]['value']
            ending_revenue = revenues[3]['value']
            years = (revenues[3]['date'] - revenues[0]['date']).days / 365.25
            
            if beginning_revenue > 0 and years > 0:
                growth_metrics['revenue_growth_1y'] = self.calculate_cagr(beginning_revenue, ending_revenue, years)
            else:
                growth_metrics['revenue_growth_1y'] = None
        else:
            growth_metrics['revenue_growth_1y'] = None
        
        # 3-Year Revenue Growth (12 quarters)
        if len(revenues) >= 12:
            beginning_revenue = revenues[0]['value']
            ending_revenue = revenues[11]['value']
            years = (revenues[11]['date'] - revenues[0]['date']).days / 365.25
            
            if beginning_revenue > 0 and years > 0:
                growth_metrics['revenue_growth_3y'] = self.calculate_cagr(beginning_revenue, ending_revenue, years)
            else:
                growth_metrics['revenue_growth_3y'] = None
        else:
            growth_metrics['revenue_growth_3y'] = None
        
        return growth_metrics
    
    def _calculate_earnings_growth(self, periods: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
        """Calculate earnings growth rates (1Y and 3Y)"""
        growth_metrics = {}
        
        # Extract earnings data from periods
        earnings = []
        for period in periods:
            income_stmt = period.get('income_statement')
            if income_stmt:
                net_income = self.get_financial_value(income_stmt.net_income_loss)
                if net_income is not None:
                    earnings.append({
                        'value': net_income,
                        'date': period['date']
                    })
        
        if len(earnings) < 2:
            self.logger.debug("Insufficient earnings data for growth calculations")
            growth_metrics['earnings_growth_1y'] = None
            growth_metrics['earnings_growth_3y'] = None
            return growth_metrics
        
        # Sort by date (oldest first)
        earnings.sort(key=lambda x: x['date'])
        
        # 1-Year Earnings Growth (4 quarters)
        if len(earnings) >= 4:
            beginning_earnings = earnings[0]['value']
            ending_earnings = earnings[3]['value']
            years = (earnings[3]['date'] - earnings[0]['date']).days / 365.25
            
            growth_metrics['earnings_growth_1y'] = self._calculate_earnings_cagr(beginning_earnings, ending_earnings, years)
        else:
            growth_metrics['earnings_growth_1y'] = None
        
        # 3-Year Earnings Growth (12 quarters)
        if len(earnings) >= 12:
            beginning_earnings = earnings[0]['value']
            ending_earnings = earnings[11]['value']
            years = (earnings[11]['date'] - earnings[0]['date']).days / 365.25
            
            growth_metrics['earnings_growth_3y'] = self._calculate_earnings_cagr(beginning_earnings, ending_earnings, years)
        else:
            growth_metrics['earnings_growth_3y'] = None
        
        return growth_metrics
    
    def _calculate_book_value_growth(self, periods: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
        """Calculate book value (equity) growth rates (1Y and 3Y)"""
        growth_metrics = {}
        
        # Extract equity data from periods
        book_values = []
        for period in periods:
            balance_sheet = period.get('balance_sheet')
            if balance_sheet:
                equity = self.get_financial_value(balance_sheet.equity)
                if equity is not None:
                    book_values.append({
                        'value': equity,
                        'date': period['date']
                    })
        
        if len(book_values) < 2:
            self.logger.debug("Insufficient book value data for growth calculations")
            growth_metrics['book_value_growth_1y'] = None
            growth_metrics['book_value_growth_3y'] = None
            return growth_metrics
        
        # Sort by date (oldest first)
        book_values.sort(key=lambda x: x['date'])
        
        # 1-Year Book Value Growth (4 quarters)
        if len(book_values) >= 4:
            beginning_bv = book_values[0]['value']
            ending_bv = book_values[3]['value']
            years = (book_values[3]['date'] - book_values[0]['date']).days / 365.25
            
            if beginning_bv > 0 and years > 0:
                growth_metrics['book_value_growth_1y'] = self.calculate_cagr(beginning_bv, ending_bv, years)
            else:
                growth_metrics['book_value_growth_1y'] = None
        else:
            growth_metrics['book_value_growth_1y'] = None
        
        # 3-Year Book Value Growth (12 quarters)
        if len(book_values) >= 12:
            beginning_bv = book_values[0]['value']
            ending_bv = book_values[11]['value']
            years = (book_values[11]['date'] - book_values[0]['date']).days / 365.25
            
            if beginning_bv > 0 and years > 0:
                growth_metrics['book_value_growth_3y'] = self.calculate_cagr(beginning_bv, ending_bv, years)
            else:
                growth_metrics['book_value_growth_3y'] = None
        else:
            growth_metrics['book_value_growth_3y'] = None
        
        return growth_metrics
    
    def _calculate_asset_growth(self, periods: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
        """Calculate asset growth rates (1Y and 3Y)"""
        growth_metrics = {}
        
        # Extract asset data from periods
        assets = []
        for period in periods:
            balance_sheet = period.get('balance_sheet')
            if balance_sheet:
                total_assets = self.get_financial_value(balance_sheet.assets)
                if total_assets is not None:
                    assets.append({
                        'value': total_assets,
                        'date': period['date']
                    })
        
        if len(assets) < 2:
            self.logger.debug("Insufficient asset data for growth calculations")
            growth_metrics['asset_growth_1y'] = None
            growth_metrics['asset_growth_3y'] = None
            return growth_metrics
        
        # Sort by date (oldest first)
        assets.sort(key=lambda x: x['date'])
        
        # 1-Year Asset Growth (4 quarters)
        if len(assets) >= 4:
            beginning_assets = assets[0]['value']
            ending_assets = assets[3]['value']
            years = (assets[3]['date'] - assets[0]['date']).days / 365.25
            
            if beginning_assets > 0 and years > 0:
                growth_metrics['asset_growth_1y'] = self.calculate_cagr(beginning_assets, ending_assets, years)
            else:
                growth_metrics['asset_growth_1y'] = None
        else:
            growth_metrics['asset_growth_1y'] = None
        
        # 3-Year Asset Growth (12 quarters)
        if len(assets) >= 12:
            beginning_assets = assets[0]['value']
            ending_assets = assets[11]['value']
            years = (assets[11]['date'] - assets[0]['date']).days / 365.25
            
            if beginning_assets > 0 and years > 0:
                growth_metrics['asset_growth_3y'] = self.calculate_cagr(beginning_assets, ending_assets, years)
            else:
                growth_metrics['asset_growth_3y'] = None
        else:
            growth_metrics['asset_growth_3y'] = None
        
        return growth_metrics
    
    def _calculate_efficiency_metrics(self, periods: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
        """Calculate efficiency and turnover metrics"""
        efficiency_metrics = {}
        
        if len(periods) < 2:
            # Return None for all efficiency metrics
            efficiency_metrics.update({
                'asset_turnover': None,
                'inventory_turnover': None,
                'working_capital_turnover': None
            })
            return efficiency_metrics
        
        # Get latest period data
        latest_period = periods[0]  # Most recent
        previous_period = periods[1] if len(periods) > 1 else None
        
        latest_income = latest_period.get('income_statement')
        latest_balance = latest_period.get('balance_sheet')
        previous_balance = previous_period.get('balance_sheet') if previous_period else None
        
        # Asset Turnover = Revenue / Average Total Assets
        if latest_income and latest_balance and previous_balance:
            revenue = self.get_financial_value(latest_income.revenues)
            current_assets = self.get_financial_value(latest_balance.assets)
            previous_assets = self.get_financial_value(previous_balance.assets)
            
            if revenue and current_assets and previous_assets:
                average_assets = (current_assets + previous_assets) / 2
                efficiency_metrics['asset_turnover'] = self.safe_divide(revenue, average_assets)
            else:
                efficiency_metrics['asset_turnover'] = None
        else:
            efficiency_metrics['asset_turnover'] = None
        
        # Inventory Turnover = COGS / Average Inventory
        if latest_income and latest_balance and previous_balance:
            cogs = self.get_financial_value(latest_income.cost_of_revenue)
            current_inventory = self.get_financial_value(latest_balance.inventory_net)
            previous_inventory = self.get_financial_value(previous_balance.inventory_net)
            
            if cogs and current_inventory and previous_inventory:
                average_inventory = (current_inventory + previous_inventory) / 2
                efficiency_metrics['inventory_turnover'] = self.safe_divide(cogs, average_inventory)
            else:
                efficiency_metrics['inventory_turnover'] = None
        else:
            efficiency_metrics['inventory_turnover'] = None
        
        # Working Capital Turnover = Revenue / Average Working Capital
        if latest_income and latest_balance and previous_balance:
            revenue = self.get_financial_value(latest_income.revenues)
            
            # Current working capital
            current_ca = self.get_financial_value(latest_balance.current_assets)
            current_cl = self.get_financial_value(latest_balance.current_liabilities)
            current_wc = (current_ca - current_cl) if (current_ca and current_cl) else None
            
            # Previous working capital
            previous_ca = self.get_financial_value(previous_balance.current_assets)
            previous_cl = self.get_financial_value(previous_balance.current_liabilities)
            previous_wc = (previous_ca - previous_cl) if (previous_ca and previous_cl) else None
            
            if revenue and current_wc and previous_wc:
                average_wc = (current_wc + previous_wc) / 2
                if average_wc != 0:
                    efficiency_metrics['working_capital_turnover'] = self.safe_divide(revenue, average_wc)
                else:
                    efficiency_metrics['working_capital_turnover'] = None
            else:
                efficiency_metrics['working_capital_turnover'] = None
        else:
            efficiency_metrics['working_capital_turnover'] = None
        
        return efficiency_metrics
    
    def _calculate_earnings_cagr(self, beginning_earnings: float, ending_earnings: float, years: float) -> Optional[float]:
        """
        Calculate earnings CAGR with special handling for negative earnings
        
        Args:
            beginning_earnings: Starting earnings value
            ending_earnings: Ending earnings value
            years: Time period in years
            
        Returns:
            CAGR as decimal or None if cannot calculate
        """
        if years <= 0:
            return None
        
        # Handle negative earnings scenarios
        if beginning_earnings <= 0:
            if ending_earnings > 0:
                # Turned profitable - return 100% growth
                self.logger.debug("Company turned profitable during period")
                return 1.0
            else:
                # Both periods negative - cannot calculate meaningful CAGR
                self.logger.debug("Negative earnings in both periods")
                return None
        
        if ending_earnings <= 0:
            # Turned unprofitable - return -100% growth
            self.logger.debug("Company turned unprofitable during period")
            return -1.0
        
        # Standard CAGR calculation for positive earnings
        return self.calculate_cagr(beginning_earnings, ending_earnings, years)

 