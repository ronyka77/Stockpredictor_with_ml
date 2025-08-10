"""
Sector Analysis Calculator

This module implements comprehensive sector analysis including GICS classification,
percentile rankings, relative ratios, and peer comparisons across sector, industry, and market levels.
Only calculates the specific metrics defined in the database schema.
"""

from typing import Dict, List, Optional, Any
from datetime import date
import statistics

from src.utils.logger import get_logger
from src.feature_engineering.fundamental_indicators.base import BaseFundamentalCalculator, FundamentalCalculationResult, FundamentalCalculatorRegistry
from src.data_collector.polygon_fundamentals.data_models import FundamentalDataResponse

logger = get_logger(__name__)

@FundamentalCalculatorRegistry.register('sector_analysis')
class SectorAnalysisCalculator(BaseFundamentalCalculator):
    """Calculator for comprehensive sector analysis - database schema focused"""
    
    def get_required_fields(self) -> Dict[str, List[str]]:
        """Get required fields for sector analysis - now uses external data"""
        return {
            'income_statement': [],
            'balance_sheet': [],
            'cash_flow': [],
            'company_details': []
        }
    
    def calculate(self, data: FundamentalDataResponse) -> FundamentalCalculationResult:
        """
        Calculate comprehensive sector analysis metrics for database storage
        
        Args:
            data: FundamentalDataResponse containing financial statements
            
        Returns:
            FundamentalCalculationResult with sector analysis metrics
        """
        self.reset_quality_tracking()
        
        # Get ticker from statements
        ticker = "UNKNOWN"
        if data.income_statements and len(data.income_statements) > 0:
            ticker = data.income_statements[0].ticker or "UNKNOWN"
        elif data.balance_sheets and len(data.balance_sheets) > 0:
            ticker = data.balance_sheets[0].ticker or "UNKNOWN"
        elif data.cash_flow_statements and len(data.cash_flow_statements) > 0:
            ticker = data.cash_flow_statements[0].ticker or "UNKNOWN"
        elif data.company_details:
            ticker = data.company_details.ticker or "UNKNOWN"
        
        # Calculate all sector analysis metrics
        sector_metrics = {}
        
        # Fetch external data and peer data together
        peer_data_result = self._fetch_peer_data(ticker)
        
        if not peer_data_result or not peer_data_result.get('external_data'):
            self.logger.error(f"External data required for sector analysis of {ticker}")
            return FundamentalCalculationResult(
                ticker=ticker,
                date=date.today(),
                values={},
                quality_score=0.0,
                missing_data_count=1,
                calculation_errors=["External data required for sector analysis"]
            )
        
        external_data = peer_data_result['external_data']
        peer_data = peer_data_result['peer_data']
        
        # GICS Classification (4 fields) - from external SIC code
        sic_code = external_data.get('sic_code')
        sic_description = external_data.get('sic_description')
        sector_metrics.update(self._get_gics_classification(sic_code, sic_description))
        
        # Get company ratios from external source
        company_ratios = self._get_company_ratios_from_external(external_data)
        
        # Get peer data and calculate comparative metrics
        gics_sector = sector_metrics.get('gics_sector')
        
        if gics_sector and sic_code and peer_data:
            # Calculate all comparative metrics
            sector_metrics.update(self._calculate_sector_percentiles(company_ratios, peer_data['sector']))
            sector_metrics.update(self._calculate_industry_percentiles(company_ratios, peer_data['industry']))
            sector_metrics.update(self._calculate_market_percentiles(company_ratios, peer_data['market']))
            
            sector_metrics.update(self._calculate_relative_ratios(company_ratios, peer_data['sector']))
            
            sector_metrics.update(self._calculate_sector_medians(peer_data['sector']))
            sector_metrics.update(self._calculate_industry_medians(peer_data['industry']))
            sector_metrics.update(self._calculate_market_medians(peer_data['market']))
            
            sector_metrics.update(self._calculate_rankings_and_counts(company_ratios, peer_data))
        else:
            # If no sector classification, set all comparative metrics to None
            sector_metrics.update(self._get_empty_comparative_metrics())
        
        result = FundamentalCalculationResult(
            ticker=ticker,
            date=date.today(),
            values=sector_metrics,
            quality_score=self.calculate_quality_score(),
            missing_data_count=self.missing_data_count,
            calculation_errors=self.calculation_errors
        )
        
        self.log_calculation_summary(result)
        return result
    
    def _get_gics_classification(self, sic_code: Optional[str], sic_description: Optional[str] = None) -> Dict[str, Optional[str]]:
        """Get complete GICS classification hierarchy from external SIC code"""
        classification = {
            'gics_sector': None,
            'gics_industry_group': None,
            'gics_industry': None,
            'gics_sub_industry': None
        }
        
        if not sic_code:
            self.logger.info("No SIC code available for GICS classification")
            return classification
        
        try:
            sic_int = int(sic_code)
            
            # Map SIC to GICS hierarchy
            gics_mapping = self._map_sic_to_gics_hierarchy(sic_int, sic_description)
            classification.update(gics_mapping)
            
            self.logger.info(f"GICS classification for SIC {sic_code}: {gics_mapping['gics_sector']}")
            
        except (ValueError, TypeError):
            self.logger.warning(f"Invalid SIC code format: {sic_code}")
            self.calculation_errors.append(f"Invalid SIC code: {sic_code}")
        
        return classification
    
    def _map_sic_to_gics_hierarchy(self, sic_code: int, sic_description: Optional[str] = None) -> Dict[str, Optional[str]]:
        """Map SIC code to complete GICS hierarchy"""
        
        # Technology Sector (3570-3579, 3600-3699, 7370-7379)
        if (3570 <= sic_code <= 3579 or 3600 <= sic_code <= 3699 or 7370 <= sic_code <= 7379):
            return {
                'gics_sector': 'Information Technology',
                'gics_industry_group': 'Technology Hardware & Equipment' if sic_code < 7000 else 'Software & Services',
                'gics_industry': 'Technology Hardware, Storage & Peripherals' if sic_code < 7000 else 'Software',
                'gics_sub_industry': 'Computer Hardware' if sic_code < 7000 else 'Application Software'
            }
        
        # Healthcare Sector (2830-2836, 3840-3851, 8000-8099)
        elif (2830 <= sic_code <= 2836 or 3840 <= sic_code <= 3851 or 8000 <= sic_code <= 8099):
            return {
                'gics_sector': 'Health Care',
                'gics_industry_group': 'Pharmaceuticals, Biotechnology & Life Sciences' if sic_code < 8000 else 'Health Care Equipment & Services',
                'gics_industry': 'Pharmaceuticals' if sic_code < 8000 else 'Health Care Equipment & Supplies',
                'gics_sub_industry': 'Pharmaceuticals' if sic_code < 8000 else 'Health Care Equipment'
            }
        
        # Financial Sector (6000-6799)
        elif 6000 <= sic_code <= 6799:
            return {
                'gics_sector': 'Financials',
                'gics_industry_group': 'Banks' if sic_code < 6200 else 'Diversified Financials',
                'gics_industry': 'Banks' if sic_code < 6200 else 'Capital Markets',
                'gics_sub_industry': 'Regional Banks' if sic_code < 6200 else 'Investment Banking & Brokerage'
            }
        
        # Consumer Discretionary (5000-5999, 7000-7299)
        elif (5000 <= sic_code <= 5999 or 7000 <= sic_code <= 7299):
            return {
                'gics_sector': 'Consumer Discretionary',
                'gics_industry_group': 'Retailing' if sic_code < 7000 else 'Consumer Services',
                'gics_industry': 'Specialty Retail' if sic_code < 7000 else 'Hotels, Restaurants & Leisure',
                'gics_sub_industry': 'Specialty Stores' if sic_code < 7000 else 'Restaurants'
            }
        
        # Consumer Staples (2000-2099, 5400-5499)
        elif (2000 <= sic_code <= 2099 or 5400 <= sic_code <= 5499):
            return {
                'gics_sector': 'Consumer Staples',
                'gics_industry_group': 'Food, Beverage & Tobacco',
                'gics_industry': 'Food Products' if sic_code < 5000 else 'Food & Staples Retailing',
                'gics_sub_industry': 'Packaged Foods & Meats' if sic_code < 5000 else 'Food Retail'
            }
        
        # Energy Sector (1300-1399, 2900-2999)
        elif (1300 <= sic_code <= 1399 or 2900 <= sic_code <= 2999):
            return {
                'gics_sector': 'Energy',
                'gics_industry_group': 'Energy',
                'gics_industry': 'Oil, Gas & Consumable Fuels',
                'gics_sub_industry': 'Oil & Gas Exploration & Production' if sic_code < 2900 else 'Oil & Gas Refining & Marketing'
            }
        
        # Materials Sector (1000-1299, 2800-2829)
        elif (1000 <= sic_code <= 1299 or 2800 <= sic_code <= 2829):
            return {
                'gics_sector': 'Materials',
                'gics_industry_group': 'Materials',
                'gics_industry': 'Metals & Mining' if sic_code < 2800 else 'Chemicals',
                'gics_sub_industry': 'Steel' if sic_code < 2800 else 'Commodity Chemicals'
            }
        
        # Industrials Sector (1400-1799, 3000-3569, 3580-3599, 3700-3799, 4000-4999)
        elif (1400 <= sic_code <= 1799 or 3000 <= sic_code <= 3569 or 3580 <= sic_code <= 3599 or 
                3700 <= sic_code <= 3799 or 4000 <= sic_code <= 4999):
            return {
                'gics_sector': 'Industrials',
                'gics_industry_group': 'Capital Goods' if sic_code < 4000 else 'Transportation',
                'gics_industry': 'Machinery' if sic_code < 4000 else 'Transportation Infrastructure',
                'gics_sub_industry': 'Industrial Machinery' if sic_code < 4000 else 'Railroads'
            }
        
        # Utilities Sector (4900-4999)
        elif 4900 <= sic_code <= 4999:
            return {
                'gics_sector': 'Utilities',
                'gics_industry_group': 'Utilities',
                'gics_industry': 'Electric Utilities',
                'gics_sub_industry': 'Electric Utilities'
            }
        
        # Real Estate Sector (6500-6599)
        elif 6500 <= sic_code <= 6599:
            return {
                'gics_sector': 'Real Estate',
                'gics_industry_group': 'Real Estate',
                'gics_industry': 'Real Estate Management & Development',
                'gics_sub_industry': 'Real Estate Development'
            }
        
        # Communication Services (4800-4899)
        elif 4800 <= sic_code <= 4899:
            return {
                'gics_sector': 'Communication Services',
                'gics_industry_group': 'Telecommunication Services',
                'gics_industry': 'Diversified Telecommunication Services',
                'gics_sub_industry': 'Integrated Telecommunication Services'
            }
        
        # Default classification
        else:
            return {
                'gics_sector': 'Industrials',  # Default sector
                'gics_industry_group': 'Capital Goods',
                'gics_industry': 'Machinery',
                'gics_sub_industry': 'Industrial Machinery'
            }
    
    def _get_company_ratios_from_external(self, external_data: Dict[str, Any]) -> Dict[str, Optional[float]]:
        """Get company ratios from external data source"""
        ratios = {}
        
        # Extract ratios from external data
        ratios['pe_ratio'] = external_data.get('pe_ratio')
        ratios['pb_ratio'] = external_data.get('pb_ratio')
        ratios['ps_ratio'] = external_data.get('ps_ratio')
        ratios['roe'] = external_data.get('roe')
        ratios['roa'] = external_data.get('roa')
        ratios['debt_to_equity'] = external_data.get('debt_to_equity')
        
        return ratios
    
    def _get_external_data(self, data_loader: Any, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get external data for the ticker from database (ratios, SIC code, etc.)
        
        Args:
            data_loader: StockDataLoader instance
            ticker: Company ticker
            
        Returns:
            Dictionary with external data or None if not found
        """
        try:
            # Query to get ticker metadata and latest fundamental ratios
            query = """
            SELECT DISTINCT ON (fr.ticker)
                t.ticker,
                t.sic_code,
                t.sic_description,
                fr.pe_ratio,
                fr.pb_ratio,
                fr.ps_ratio,
                fr.roe,
                fr.roa,
                fr.debt_to_equity
            FROM tickers t
            LEFT JOIN fundamental_ratios fr ON t.ticker = fr.ticker
            WHERE t.ticker = %s
            AND t.active = true
            ORDER BY fr.ticker, fr.date DESC
            """
            
            results = data_loader.execute_query(query, [ticker])
            
            if not results:
                self.logger.warning(f"No external data found for ticker {ticker}")
                return None
            
            row = results[0]
            external_data = {
                'ticker': row[0],
                'sic_code': row[1],
                'sic_description': row[2],
                'pe_ratio': float(row[3]) if row[3] is not None else None,
                'pb_ratio': float(row[4]) if row[4] is not None else None,
                'ps_ratio': float(row[5]) if row[5] is not None else None,
                'roe': float(row[6]) if row[6] is not None else None,
                'roa': float(row[7]) if row[7] is not None else None,
                'debt_to_equity': float(row[8]) if row[8] is not None else None
            }
            
            self.logger.info(f"Retrieved external data for {ticker}: SIC={external_data['sic_code']}")
            return external_data
            
        except Exception as e:
            self.logger.error(f"Error fetching external data for {ticker}: {str(e)}")
            return None
    
    def _fetch_peer_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Fetch external data and peer data from database for sector analysis
        
        Args:
            ticker: Current company ticker
            
        Returns:
            Dictionary containing external_data and peer_data, or None if failed
        """
        try:
            from src.feature_engineering.data_loader import StockDataLoader
            
            self.logger.info(f"Fetching external data and peer data for {ticker}")
            
            with StockDataLoader() as data_loader:
                # First, get external data for the ticker (ratios, SIC code, etc.)
                external_data = self._get_external_data(data_loader, ticker)
                
                if not external_data or not external_data.get('sic_code'):
                    self.logger.warning(f"No external data or SIC code found for {ticker}")
                    return None
                
                sic_code = external_data['sic_code']
                
                # Initialize peer data structure
                peer_data = {
                    'sector': [],
                    'industry': [],
                    'market': []
                }
                
                # Fetch sector peers (same SIC code)
                sector_peers = self._get_sector_peers(data_loader, sic_code, ticker)
                peer_data['sector'] = sector_peers
                
                # For SIC-based grouping, industry peers are the same as sector peers
                peer_data['industry'] = sector_peers
                
                # Fetch market peers (all companies with fundamental data)
                market_peers = self._get_market_peers(data_loader, ticker)
                peer_data['market'] = market_peers
                
                self.logger.info(f"Retrieved data for {ticker}: "
                                f"external_data={bool(external_data)}, "
                                f"sector={len(peer_data['sector'])}, "
                                f"industry={len(peer_data['industry'])}, "
                                f"market={len(peer_data['market'])}")
                
                return {
                    'external_data': external_data,
                    'peer_data': peer_data
                }
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {ticker}: {str(e)}")
            self.calculation_errors.append(f"Data fetch error: {str(e)}")
            return None
    
    def _get_sector_peers(self, data_loader: Any, sic_code: str, exclude_ticker: str) -> List[Dict[str, Any]]:
        """
        Get peer companies in the same SIC sector
        
        Args:
            data_loader: StockDataLoader instance
            sic_code: SIC code for sector grouping
            exclude_ticker: Ticker to exclude from results
            
        Returns:
            List of peer company fundamental ratios
        """
        try:
            # Query for companies with same SIC code and their latest fundamental ratios
            query = """
            SELECT DISTINCT ON (fr.ticker)
                fr.ticker,
                fr.pe_ratio,
                fr.pb_ratio, 
                fr.ps_ratio,
                fr.roe,
                fr.roa,
                fr.debt_to_equity,
                fr.current_ratio,
                fr.quick_ratio,
                fr.gross_margin,
                fr.operating_margin,
                fr.net_margin,
                t.market_cap,
                t.sic_description
            FROM fundamental_ratios fr
            JOIN tickers t ON fr.ticker = t.ticker
            WHERE t.sic_code = %s
            AND fr.ticker != %s
            AND t.active = true
            ORDER BY fr.ticker, fr.date DESC
            """
            
            results = data_loader.execute_query(query, [sic_code, exclude_ticker])
            
            # Convert to list of dictionaries
            peer_data = []
            for row in results:
                peer_dict = {
                    'ticker': row[0],
                    'pe_ratio': float(row[1]) if row[1] is not None else None,
                    'pb_ratio': float(row[2]) if row[2] is not None else None,
                    'ps_ratio': float(row[3]) if row[3] is not None else None,
                    'roe': float(row[4]) if row[4] is not None else None,
                    'roa': float(row[5]) if row[5] is not None else None,
                    'debt_to_equity': float(row[6]) if row[6] is not None else None,
                    'current_ratio': float(row[7]) if row[7] is not None else None,
                    'quick_ratio': float(row[8]) if row[8] is not None else None,
                    'gross_margin': float(row[9]) if row[9] is not None else None,
                    'operating_margin': float(row[10]) if row[10] is not None else None,
                    'net_margin': float(row[11]) if row[11] is not None else None,
                    'market_cap': float(row[12]) if row[12] is not None else None,
                    'sic_description': row[13]
                }
                peer_data.append(peer_dict)
            
            self.logger.info(f"Found {len(peer_data)} sector peers for SIC {sic_code}")
            return peer_data
            
        except Exception as e:
            self.logger.error(f"Error fetching sector peers for SIC {sic_code}: {e}")
            return []
    
    def _get_market_peers(self, data_loader: Any, exclude_ticker: str) -> List[Dict[str, Any]]:
        """
        Get market-wide peer companies for comparison
        
        Args:
            data_loader: StockDataLoader instance
            exclude_ticker: Ticker to exclude from results
            
        Returns:
            List of market peer fundamental ratios
        """
        try:
            # Query for all companies with fundamental data (sample for performance)
            query = """
            SELECT DISTINCT ON (fr.ticker)
                fr.ticker,
                fr.pe_ratio,
                fr.pb_ratio,
                fr.roe,
                t.market_cap
            FROM fundamental_ratios fr
            JOIN tickers t ON fr.ticker = t.ticker
            WHERE fr.ticker != %s
            AND t.active = true
            AND t.is_sp500 = true  -- Focus on S&P 500 for market comparison
            ORDER BY fr.ticker, fr.date DESC
            LIMIT 500  -- Limit for performance
            """
            
            results = data_loader.execute_query(query, [exclude_ticker])
            
            # Convert to list of dictionaries
            peer_data = []
            for row in results:
                peer_dict = {
                    'ticker': row[0],
                    'pe_ratio': float(row[1]) if row[1] is not None else None,
                    'pb_ratio': float(row[2]) if row[2] is not None else None,
                    'roe': float(row[3]) if row[3] is not None else None,
                    'market_cap': float(row[4]) if row[4] is not None else None
                }
                peer_data.append(peer_dict)
            
            self.logger.info(f"Found {len(peer_data)} market peers")
            return peer_data
            
        except Exception as e:
            self.logger.error(f"Error fetching market peers: {e}")
            return []
    
    def _calculate_sector_percentiles(self, company_ratios: Dict[str, Optional[float]], sector_data: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
        """Calculate percentile rankings within sector"""
        percentiles = {}
        
        # Key ratios for sector percentiles
        ratio_fields = ['pe_ratio', 'pb_ratio', 'ps_ratio', 'roe', 'roa', 'debt_to_equity']
        
        for ratio_name in ratio_fields:
            company_value = company_ratios.get(ratio_name)
            if company_value is None:
                percentiles[f"{ratio_name}_sector_percentile"] = None
                continue
            
            # Extract sector values
            sector_values = [peer.get(ratio_name) for peer in sector_data if peer.get(ratio_name) is not None]
            
            if len(sector_values) >= 3:  # Minimum data points
                percentile = self._calculate_percentile(company_value, sector_values)
                percentiles[f"{ratio_name}_sector_percentile"] = percentile
            else:
                percentiles[f"{ratio_name}_sector_percentile"] = None
        
        return percentiles
    
    def _calculate_industry_percentiles(self, company_ratios: Dict[str, Optional[float]], industry_data: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
        """Calculate percentile rankings within industry"""
        percentiles = {}
        
        # Key ratios for industry percentiles
        ratio_fields = ['pe_ratio', 'pb_ratio', 'roe']
        
        for ratio_name in ratio_fields:
            company_value = company_ratios.get(ratio_name)
            if company_value is None:
                percentiles[f"{ratio_name}_industry_percentile"] = None
                continue
            
            # Extract industry values
            industry_values = [peer.get(ratio_name) for peer in industry_data if peer.get(ratio_name) is not None]
            
            if len(industry_values) >= 3:  # Minimum data points
                percentile = self._calculate_percentile(company_value, industry_values)
                percentiles[f"{ratio_name}_industry_percentile"] = percentile
            else:
                percentiles[f"{ratio_name}_industry_percentile"] = None
        
        return percentiles
    
    def _calculate_market_percentiles(self, company_ratios: Dict[str, Optional[float]], market_data: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
        """Calculate percentile rankings within entire market"""
        percentiles = {}
        
        # Key ratios for market percentiles
        ratio_fields = ['pe_ratio', 'pb_ratio', 'roe']
        
        for ratio_name in ratio_fields:
            company_value = company_ratios.get(ratio_name)
            if company_value is None:
                percentiles[f"{ratio_name}_market_percentile"] = None
                continue
            
            # Extract market values
            market_values = [peer.get(ratio_name) for peer in market_data if peer.get(ratio_name) is not None]
            
            if len(market_values) >= 10:  # Minimum data points for market
                percentile = self._calculate_percentile(company_value, market_values)
                percentiles[f"{ratio_name}_market_percentile"] = percentile
            else:
                percentiles[f"{ratio_name}_market_percentile"] = None
        
        return percentiles
    
    def _calculate_relative_ratios(self, company_ratios: Dict[str, Optional[float]], sector_data: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
        """Calculate ratios relative to sector median"""
        relative_ratios = {}
        
        # Key ratios for relative comparison
        ratio_fields = ['pe_ratio', 'pb_ratio', 'ps_ratio', 'roe', 'roa']
        
        for ratio_name in ratio_fields:
            company_value = company_ratios.get(ratio_name)
            if company_value is None:
                relative_ratios[f"{ratio_name}_relative_to_sector"] = None
                continue
            
            # Calculate sector median
            sector_values = [peer.get(ratio_name) for peer in sector_data if peer.get(ratio_name) is not None]
            
            if len(sector_values) >= 3:
                sector_median = statistics.median(sector_values)
                if sector_median != 0:
                    relative_ratio = company_value / sector_median
                    relative_ratios[f"{ratio_name}_relative_to_sector"] = relative_ratio
                else:
                    relative_ratios[f"{ratio_name}_relative_to_sector"] = None
            else:
                relative_ratios[f"{ratio_name}_relative_to_sector"] = None
        
        return relative_ratios
    
    def _calculate_sector_medians(self, sector_data: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
        """Calculate sector median values"""
        medians = {}
        
        ratio_fields = ['pe_ratio', 'pb_ratio', 'ps_ratio', 'roe', 'roa']
        
        for ratio_name in ratio_fields:
            sector_values = [peer.get(ratio_name) for peer in sector_data if peer.get(ratio_name) is not None]
            
            if len(sector_values) >= 3:
                medians[f"sector_median_{ratio_name}"] = statistics.median(sector_values)
            else:
                medians[f"sector_median_{ratio_name}"] = None
        
        return medians
    
    def _calculate_industry_medians(self, industry_data: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
        """Calculate industry median values"""
        medians = {}
        
        ratio_fields = ['pe_ratio', 'pb_ratio', 'roe']
        
        for ratio_name in ratio_fields:
            industry_values = [peer.get(ratio_name) for peer in industry_data if peer.get(ratio_name) is not None]
            
            if len(industry_values) >= 3:
                medians[f"industry_median_{ratio_name}"] = statistics.median(industry_values)
            else:
                medians[f"industry_median_{ratio_name}"] = None
        
        return medians
    
    def _calculate_market_medians(self, market_data: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
        """Calculate market median values"""
        medians = {}
        
        ratio_fields = ['pe_ratio', 'pb_ratio', 'roe']
        
        for ratio_name in ratio_fields:
            market_values = [peer.get(ratio_name) for peer in market_data if peer.get(ratio_name) is not None]
            
            if len(market_values) >= 10:
                medians[f"market_median_{ratio_name}"] = statistics.median(market_values)
            else:
                medians[f"market_median_{ratio_name}"] = None
        
        return medians
    
    def _calculate_rankings_and_counts(self, company_ratios: Dict[str, Optional[float]], peer_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Optional[int]]:
        """Calculate rankings and company counts"""
        rankings = {}
        
        # Use ROE for ranking calculations (most reliable fundamental metric)
        company_roe = company_ratios.get('roe')
        
        # Sector ranking
        sector_data = peer_data['sector']
        if company_roe is not None and len(sector_data) >= 3:
            sector_roe_values = [peer.get('roe') for peer in sector_data if peer.get('roe') is not None]
            if sector_roe_values:
                sector_rank = sum(1 for roe in sector_roe_values if roe < company_roe) + 1
                rankings['sector_rank'] = sector_rank
                rankings['sector_total_companies'] = len(sector_roe_values)
            else:
                rankings['sector_rank'] = None
                rankings['sector_total_companies'] = None
        else:
            rankings['sector_rank'] = None
            rankings['sector_total_companies'] = None
        
        # Industry ranking
        industry_data = peer_data['industry']
        if company_roe is not None and len(industry_data) >= 3:
            industry_roe_values = [peer.get('roe') for peer in industry_data if peer.get('roe') is not None]
            if industry_roe_values:
                industry_rank = sum(1 for roe in industry_roe_values if roe < company_roe) + 1
                rankings['industry_rank'] = industry_rank
                rankings['industry_total_companies'] = len(industry_roe_values)
            else:
                rankings['industry_rank'] = None
                rankings['industry_total_companies'] = None
        else:
            rankings['industry_rank'] = None
            rankings['industry_total_companies'] = None
        
        # Market ranking
        market_data = peer_data['market']
        if company_roe is not None and len(market_data) >= 10:
            market_roe_values = [peer.get('roe') for peer in market_data if peer.get('roe') is not None]
            if market_roe_values:
                market_rank = sum(1 for roe in market_roe_values if roe < company_roe) + 1
                rankings['market_rank'] = market_rank
                rankings['market_total_companies'] = len(market_roe_values)
            else:
                rankings['market_rank'] = None
                rankings['market_total_companies'] = None
        else:
            rankings['market_rank'] = None
            rankings['market_total_companies'] = None
        
        return rankings
    
    def _calculate_percentile(self, value: float, peer_values: List[float]) -> float:
        """Calculate percentile ranking of value within peer group"""
        if not peer_values:
            return 0.0
        
        peer_values_sorted = sorted(peer_values)
        rank = sum(1 for v in peer_values_sorted if v < value)
        percentile = (rank / len(peer_values_sorted)) * 100
        
        return round(percentile, 2)
    
    def _get_empty_comparative_metrics(self) -> Dict[str, Optional[float]]:
        """Return all comparative metrics set to None when no sector classification"""
        return {
            # Sector percentiles
            'pe_sector_percentile': None,
            'pb_sector_percentile': None,
            'ps_sector_percentile': None,
            'roe_sector_percentile': None,
            'roa_sector_percentile': None,
            'debt_to_equity_sector_percentile': None,
            
            # Industry percentiles
            'pe_industry_percentile': None,
            'pb_industry_percentile': None,
            'roe_industry_percentile': None,
            
            # Market percentiles
            'pe_market_percentile': None,
            'pb_market_percentile': None,
            'roe_market_percentile': None,
            
            # Relative ratios
            'pe_relative_to_sector': None,
            'pb_relative_to_sector': None,
            'ps_relative_to_sector': None,
            'roe_relative_to_sector': None,
            'roa_relative_to_sector': None,
            
            # Sector medians
            'sector_median_pe': None,
            'sector_median_pb': None,
            'sector_median_ps': None,
            'sector_median_roe': None,
            'sector_median_roa': None,
            
            # Industry medians
            'industry_median_pe': None,
            'industry_median_pb': None,
            'industry_median_roe': None,
            
            # Market medians
            'market_median_pe': None,
            'market_median_pb': None,
            'market_median_roe': None,
            
            # Rankings and counts
            'sector_rank': None,
            'industry_rank': None,
            'market_rank': None,
            'sector_total_companies': None,
            'industry_total_companies': None,
            'market_total_companies': None
        }

 