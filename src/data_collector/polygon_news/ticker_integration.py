"""
Integration with existing ticker_manager for news collection prioritization
Provides intelligent ticker selection based on market cap and trading volume
"""

from typing import List, Dict, Optional, Tuple, Any

from src.utils.logger import get_polygon_logger
from src.data_collector.ticker_manager import TickerManager
logger = get_polygon_logger(__name__)


class NewsTickerIntegration:
    """
    Integration layer between news collection and existing ticker management
    Provides prioritized ticker lists for efficient news collection
    """
    
    def __init__(self, ticker_manager: Optional[TickerManager] = None):
        """
        Initialize ticker integration
        
        Args:
            ticker_manager: Optional existing TickerManager instance
        """
        self.ticker_manager = ticker_manager
        self.logger = get_polygon_logger(self.__class__.__name__)
        
        # Fallback ticker lists if ticker_manager is unavailable
        self.fallback_major_tickers = [
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'VEA', 'VWO',
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BRK.B', 'V', 'MA',
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'DHR', 'ABT',
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'KMI', 'OKE'
        ]
        
        self.fallback_growth_tickers = [
            'TSLA', 'NVDA', 'AMD', 'CRM', 'SNOW', 'PLTR', 'ROKU', 'SQ',
            'SHOP', 'TWLO', 'ZM', 'DOCU', 'OKTA', 'CRWD', 'NET', 'DDOG'
        ]
        
        self.fallback_value_tickers = [
            'BRK.B', 'JPM', 'BAC', 'WFC', 'XOM', 'CVX', 'JNJ', 'PG',
            'KO', 'PEP', 'WMT', 'HD', 'VZ', 'T', 'IBM', 'INTC'
        ]
    
    def get_prioritized_tickers(self, 
                                max_tickers: int = 100,
                                include_etfs: bool = True,
                                market_cap_min: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Get prioritized list of tickers for news collection
        
        Args:
            max_tickers: Maximum number of tickers to return
            include_etfs: Whether to include ETFs
            market_cap_min: Minimum market cap filter (in billions)
            
        Returns:
            List of ticker dictionaries with priority information
        """
        if self.ticker_manager:
            try:
                return self._get_tickers_from_manager(max_tickers, include_etfs, market_cap_min)
            except Exception as e:
                self.logger.warning(f"Failed to get tickers from manager: {e}")
                return self._get_fallback_tickers(max_tickers, include_etfs)
        else:
            self.logger.info("Using fallback ticker list")
            return self._get_fallback_tickers(max_tickers, include_etfs)
    
    def _get_tickers_from_manager(self, 
                                    max_tickers: int,
                                    include_etfs: bool,
                                    market_cap_min: Optional[float]) -> List[Dict[str, Any]]:
        """Get tickers from the ticker manager with prioritization"""
        try:
            # Get active tickers from manager
            active_tickers = self.ticker_manager.get_active_tickers()
            
            prioritized_tickers = []
            
            for ticker_info in active_tickers:
                ticker = ticker_info.get('ticker', '')
                market_cap = ticker_info.get('market_cap', 0)
                avg_volume = ticker_info.get('avg_volume', 0)
                sector = ticker_info.get('sector', '')
                
                # Apply market cap filter
                if market_cap_min and market_cap < market_cap_min * 1e9:
                    continue
                
                # Skip ETFs if not requested
                if not include_etfs and self._is_etf(ticker):
                    continue
                
                # Calculate priority score
                priority_score = self._calculate_priority_score(
                    market_cap, avg_volume, sector, ticker
                )
                
                prioritized_tickers.append({
                    'ticker': ticker,
                    'priority_score': priority_score,
                    'market_cap': market_cap,
                    'avg_volume': avg_volume,
                    'sector': sector,
                    'is_major': ticker in self.fallback_major_tickers,
                    'category': self._categorize_ticker(ticker, sector, market_cap)
                })
            
            # Sort by priority score (highest first)
            prioritized_tickers.sort(key=lambda x: x['priority_score'], reverse=True)
            
            return prioritized_tickers[:max_tickers]
            
        except Exception as e:
            self.logger.error(f"Error getting tickers from manager: {e}")
            raise
    
    def _get_fallback_tickers(self, max_tickers: int, include_etfs: bool) -> List[Dict[str, Any]]:
        """Get fallback ticker list when ticker manager is unavailable"""
        fallback_tickers = []
        
        # Major tickers (highest priority)
        for i, ticker in enumerate(self.fallback_major_tickers):
            if not include_etfs and self._is_etf(ticker):
                continue
                
            fallback_tickers.append({
                'ticker': ticker,
                'priority_score': 100 - i,  # Decreasing priority
                'market_cap': None,
                'avg_volume': None,
                'sector': 'Unknown',
                'is_major': True,
                'category': 'major'
            })
        
        # Add growth tickers if we need more
        if len(fallback_tickers) < max_tickers:
            for i, ticker in enumerate(self.fallback_growth_tickers):
                if len(fallback_tickers) >= max_tickers:
                    break
                    
                if ticker not in [t['ticker'] for t in fallback_tickers]:
                    fallback_tickers.append({
                        'ticker': ticker,
                        'priority_score': 50 - i,
                        'market_cap': None,
                        'avg_volume': None,
                        'sector': 'Technology',
                        'is_major': False,
                        'category': 'growth'
                    })
        
        # Add value tickers if we still need more
        if len(fallback_tickers) < max_tickers:
            for i, ticker in enumerate(self.fallback_value_tickers):
                if len(fallback_tickers) >= max_tickers:
                    break
                    
                if ticker not in [t['ticker'] for t in fallback_tickers]:
                    fallback_tickers.append({
                        'ticker': ticker,
                        'priority_score': 25 - i,
                        'market_cap': None,
                        'avg_volume': None,
                        'sector': 'Financial',
                        'is_major': False,
                        'category': 'value'
                    })
        
        return fallback_tickers[:max_tickers]
    
    def _calculate_priority_score(self, 
                                    market_cap: float,
                                    avg_volume: float,
                                    sector: str,
                                    ticker: str) -> float:
        """
        Calculate priority score for news collection
        Higher score = higher priority for news collection
        """
        score = 0.0
        
        # Market cap component (0-40 points)
        if market_cap:
            if market_cap >= 500e9:  # $500B+
                score += 40
            elif market_cap >= 100e9:  # $100B+
                score += 35
            elif market_cap >= 50e9:   # $50B+
                score += 30
            elif market_cap >= 10e9:   # $10B+
                score += 25
            elif market_cap >= 1e9:    # $1B+
                score += 15
            else:
                score += 5
        
        # Volume component (0-30 points)
        if avg_volume:
            if avg_volume >= 50e6:     # 50M+ shares
                score += 30
            elif avg_volume >= 20e6:   # 20M+ shares
                score += 25
            elif avg_volume >= 10e6:   # 10M+ shares
                score += 20
            elif avg_volume >= 5e6:    # 5M+ shares
                score += 15
            elif avg_volume >= 1e6:    # 1M+ shares
                score += 10
            else:
                score += 5
        
        # Sector bonus (0-20 points)
        high_news_sectors = ['Technology', 'Healthcare', 'Financial', 'Energy', 'Consumer']
        if any(sector_name in sector for sector_name in high_news_sectors):
            score += 20
        else:
            score += 10
        
        # Major ticker bonus (0-10 points)
        if ticker in self.fallback_major_tickers:
            score += 10
        
        return score
    
    def _categorize_ticker(self, ticker: str, sector: str, market_cap: Optional[float]) -> str:
        """Categorize ticker for news collection strategy"""
        if ticker in self.fallback_major_tickers:
            return 'major'
        elif ticker in self.fallback_growth_tickers:
            return 'growth'
        elif ticker in self.fallback_value_tickers:
            return 'value'
        elif self._is_etf(ticker):
            return 'etf'
        elif market_cap and market_cap >= 100e9:
            return 'large_cap'
        elif market_cap and market_cap >= 10e9:
            return 'mid_cap'
        else:
            return 'small_cap'
    
    def _is_etf(self, ticker: str) -> bool:
        """Check if ticker is likely an ETF"""
        etf_indicators = ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'VEA', 'VWO', 'XLF', 'XLK', 'XLE']
        return ticker in etf_indicators or len(ticker) <= 3
    
    def get_tickers_by_category(self, category: str, max_tickers: int = 50) -> List[str]:
        """
        Get tickers by specific category
        
        Args:
            category: Category name ('major', 'growth', 'value', 'etf')
            max_tickers: Maximum number of tickers to return
            
        Returns:
            List of ticker symbols
        """
        all_tickers = self.get_prioritized_tickers(max_tickers * 2)
        
        category_tickers = [
            ticker_info['ticker'] 
            for ticker_info in all_tickers 
            if ticker_info['category'] == category
        ]
        
        return category_tickers[:max_tickers]
    
    def get_major_tickers(self, max_tickers: int = 20) -> List[str]:
        """Get list of major tickers for priority news collection"""
        return self.get_tickers_by_category('major', max_tickers)
    
    def get_growth_tickers(self, max_tickers: int = 30) -> List[str]:
        """Get list of growth tickers for news collection"""
        return self.get_tickers_by_category('growth', max_tickers)
    
    def get_sector_tickers(self, sector: str, max_tickers: int = 20) -> List[str]:
        """
        Get tickers from specific sector
        
        Args:
            sector: Sector name to filter by
            max_tickers: Maximum number of tickers to return
            
        Returns:
            List of ticker symbols from the sector
        """
        all_tickers = self.get_prioritized_tickers(200)
        
        sector_tickers = [
            ticker_info['ticker']
            for ticker_info in all_tickers
            if sector.lower() in ticker_info.get('sector', '').lower()
        ]
        
        return sector_tickers[:max_tickers]
    
    def get_collection_strategy(self, total_budget: int = 1000) -> Dict[str, List[str]]:
        """
        Get optimized collection strategy based on available request budget
        
        Args:
            total_budget: Total number of API requests available
            
        Returns:
            Dictionary with ticker allocation by priority
        """
        strategy = {
            'high_priority': [],
            'medium_priority': [],
            'low_priority': []
        }
        
        # Allocate budget: 50% high, 30% medium, 20% low priority
        high_budget = int(total_budget * 0.5)
        medium_budget = int(total_budget * 0.3)
        
        all_tickers = self.get_prioritized_tickers(total_budget)
        
        # High priority: top scoring tickers
        strategy['high_priority'] = [
            ticker['ticker'] for ticker in all_tickers[:high_budget]
        ]
        
        # Medium priority: next tier
        strategy['medium_priority'] = [
            ticker['ticker'] for ticker in all_tickers[high_budget:high_budget + medium_budget]
        ]
        
        # Low priority: remaining
        strategy['low_priority'] = [
            ticker['ticker'] for ticker in all_tickers[high_budget + medium_budget:total_budget]
        ]
        
        self.logger.info(f"Collection strategy: {len(strategy['high_priority'])} high, "
                        f"{len(strategy['medium_priority'])} medium, "
                        f"{len(strategy['low_priority'])} low priority tickers")
        
        return strategy
    
    def validate_ticker_list(self, tickers: List[str]) -> Tuple[List[str], List[str]]:
        """
        Validate ticker list and return valid/invalid tickers
        
        Args:
            tickers: List of ticker symbols to validate
            
        Returns:
            Tuple of (valid_tickers, invalid_tickers)
        """
        valid_tickers = []
        invalid_tickers = []
        
        for ticker in tickers:
            # Basic validation
            if not ticker or not isinstance(ticker, str):
                invalid_tickers.append(ticker)
                continue
            
            # Clean ticker symbol
            clean_ticker = ticker.strip().upper()
            
            # Basic format validation
            if len(clean_ticker) < 1 or len(clean_ticker) > 10:
                invalid_tickers.append(ticker)
                continue
            
            # Check for invalid characters
            if not clean_ticker.replace('.', '').replace('-', '').isalnum():
                invalid_tickers.append(ticker)
                continue
            
            valid_tickers.append(clean_ticker)
        
        return valid_tickers, invalid_tickers
    
    def get_ticker_info(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific ticker"""
        prioritized_tickers = self.get_prioritized_tickers(1000)
        
        for ticker_info in prioritized_tickers:
            if ticker_info['ticker'] == ticker.upper():
                return ticker_info
        
        # Return basic info if not found in prioritized list
        return {
            'ticker': ticker.upper(),
            'priority_score': 0,
            'market_cap': None,
            'avg_volume': None,
            'sector': 'Unknown',
            'is_major': ticker.upper() in self.fallback_major_tickers,
            'category': 'unknown'
        } 