"""
Data handling component for QuantLab Professional.
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import streamlit as st
from utils.logger import get_logger

logger = get_logger(__name__)


@st.cache_data(ttl=3600)
def _cached_download_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Cached function for downloading single ticker stock data."""
    return yf.download(
        ticker, 
        start=start_date, 
        end=end_date, 
        auto_adjust=False, 
        progress=False,
        threads=True
    )


@st.cache_data(ttl=3600)
def _cached_download_multiple_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Cached function for downloading multiple ticker stock data."""
    return yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        group_by='ticker',
        threads=True
    )


class DataHandler:
    """
    Professional data handling with caching, validation, and preprocessing.
    """
    
    def __init__(self):
        self.cache = {}
        logger.info("DataHandler initialized")
    
    def load_data(self, ticker: str, days: int = 1825) -> Optional[pd.DataFrame]:
        """
        Load stock data with professional caching and error handling.
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days of historical data
        
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            logger.info(f"Loading data for {ticker} from {start_date.date()} to {end_date.date()}")
            
            # Use cached download function
            data = _cached_download_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            if data.empty:
                logger.warning(f"No data returned for {ticker}")
                return None
            
            # Clean data
            data = self._clean_data(data)
            
            logger.info(f"Successfully loaded {len(data)} data points for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data for {ticker}: {e}")
            return None
    
    def load_multiple_tickers(self, tickers: List[str], days: int = 1825) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple tickers efficiently.
        
        Args:
            tickers: List of ticker symbols
            days: Number of days of historical data
        
        Returns:
            Dictionary mapping tickers to their data
        """
        results = {}
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            logger.info(f"Loading data for {len(tickers)} tickers")
            
            # Use cached batch download
            data = _cached_download_multiple_data(tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            if isinstance(data.columns, pd.MultiIndex):
                # Multi-ticker data
                for ticker in tickers:
                    try:
                        ticker_data = data[ticker].dropna()
                        if not ticker_data.empty:
                            results[ticker] = self._clean_data(ticker_data)
                    except KeyError:
                        logger.warning(f"No data found for {ticker}")
                        continue
            else:
                # Single ticker data
                if len(tickers) == 1:
                    results[tickers[0]] = self._clean_data(data)
            
            logger.info(f"Successfully loaded data for {len(results)} tickers")
            return results
            
        except Exception as e:
            logger.error(f"Error loading multiple tickers: {e}")
            return {}
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate financial data.
        
        Args:
            data: Raw financial data
        
        Returns:
            Cleaned DataFrame
        """
        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Ensure datetime index
        data.index = pd.to_datetime(data.index, errors='coerce')
        
        # Remove duplicates and sort
        data = data[~data.index.duplicated()].sort_index()
        
        # Fill missing values with forward fill (then backward fill for leading NaNs)
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Remove rows with invalid prices (negative or zero)
        data = data[(data['Close'] > 0) & (data['High'] > 0) & 
                   (data['Low'] > 0) & (data['Open'] > 0)]
        
        # Validate OHLC relationships
        data = data[
            (data['High'] >= data['Low']) & 
            (data['High'] >= data['Open']) & 
            (data['High'] >= data['Close']) &
            (data['Low'] <= data['Open']) & 
            (data['Low'] <= data['Close'])
        ]
        
        logger.debug(f"Cleaned data: {len(data)} valid rows")
        return data
    
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw data with technical indicators and features.
        
        Args:
            data: Clean OHLCV data
        
        Returns:
            DataFrame with additional features
        """
        try:
            processed_data = data.copy()
            
            # Basic returns
            processed_data['Return'] = processed_data['Close'].pct_change()
            processed_data['LogReturn'] = np.log(processed_data['Close'] / processed_data['Close'].shift(1))
            
            # Moving averages
            for window in [5, 10, 20, 50, 200]:
                processed_data[f'SMA_{window}'] = processed_data['Close'].rolling(window).mean()
                processed_data[f'EMA_{window}'] = processed_data['Close'].ewm(span=window).mean()
            
            # Technical indicators
            processed_data = self._add_technical_indicators(processed_data)
            
            # Volatility measures
            processed_data = self._add_volatility_measures(processed_data)
            
            # Volume indicators (if available)
            if 'Volume' in processed_data.columns:
                processed_data = self._add_volume_indicators(processed_data)
            
            logger.info(f"Processed data with {len(processed_data.columns)} features")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataset."""
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = data['Close'].ewm(span=12).mean()
        ema_26 = data['Close'].ewm(span=26).mean()
        data['MACD'] = ema_12 - ema_26
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        
        # Bollinger Bands
        sma_20 = data['Close'].rolling(20).mean()
        std_20 = data['Close'].rolling(20).std()
        data['BB_Upper'] = sma_20 + (std_20 * 2)
        data['BB_Lower'] = sma_20 - (std_20 * 2)
        data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
        data['BB_Position'] = (data['Close'] - data['BB_Lower']) / data['BB_Width']
        
        # Stochastic Oscillator
        lowest_low = data['Low'].rolling(14).min()
        highest_high = data['High'].rolling(14).max()
        data['Stoch_K'] = 100 * (data['Close'] - lowest_low) / (highest_high - lowest_low)
        data['Stoch_D'] = data['Stoch_K'].rolling(3).mean()
        
        # Average True Range (ATR)
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data['ATR'] = true_range.rolling(14).mean()
        
        return data
    
    def _add_volatility_measures(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility measures."""
        returns = data['Return']
        
        # Rolling volatility (different windows)
        for window in [5, 10, 20, 60]:
            data[f'Vol_{window}'] = returns.rolling(window).std() * np.sqrt(252)
        
        # Parkinson volatility (uses high-low range)
        data['Parkinson_Vol'] = np.sqrt(
            252 * 0.25 * np.log(data['High'] / data['Low']) ** 2
        ).rolling(20).mean()
        
        # Garman-Klass volatility
        ln_hl = np.log(data['High'] / data['Low'])
        ln_co = np.log(data['Close'] / data['Open'])
        data['GK_Vol'] = np.sqrt(
            252 * (0.5 * ln_hl ** 2 - (2 * np.log(2) - 1) * ln_co ** 2)
        ).rolling(20).mean()
        
        return data
    
    def _add_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        # Volume moving averages
        data['Volume_SMA_20'] = data['Volume'].rolling(20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA_20']
        
        # On-Balance Volume (OBV)
        data['OBV'] = (np.sign(data['Return']) * data['Volume']).cumsum()
        
        # Volume Price Trend (VPT)
        data['VPT'] = (data['Return'] * data['Volume']).cumsum()
        
        # Chaikin Money Flow
        mf_multiplier = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
        mf_volume = mf_multiplier * data['Volume']
        data['CMF'] = mf_volume.rolling(20).sum() / data['Volume'].rolling(20).sum()
        
        return data
    
    def calculate_summary_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate summary metrics for the dataset."""
        try:
            latest_close = float(data['Close'].iloc[-1])
            prev_close = float(data['Close'].iloc[-2])
            change = latest_close - prev_close
            change_pct = (change / prev_close) * 100
            
            return {
                'latest_close': latest_close,
                'change': change,
                'change_pct': change_pct,
                'start_date': data.index[0].strftime('%Y-%m-%d'),
                'end_date': data.index[-1].strftime('%Y-%m-%d'),
                'total_days': len(data),
                'avg_volume': data['Volume'].mean() if 'Volume' in data.columns else 0,
                'volatility_20d': data['Return'].rolling(20).std().iloc[-1] * np.sqrt(252) * 100,
                'max_drawdown': self._calculate_max_drawdown(data['Close'])
            }
        except Exception as e:
            logger.error(f"Error calculating summary metrics: {e}")
            return {}
    
    def calculate_statistical_summary(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive statistical summary."""
        try:
            # Select numeric columns for analysis
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            # Calculate statistics
            stats = data[numeric_cols].describe()
            
            # Add additional statistics
            stats.loc['skewness'] = data[numeric_cols].skew()
            stats.loc['kurtosis'] = data[numeric_cols].kurtosis()
            
            # Add financial metrics for price columns
            price_cols = ['Open', 'High', 'Low', 'Close']
            for col in price_cols:
                if col in data.columns:
                    returns = data[col].pct_change().dropna()
                    if len(returns) > 0:
                        stats.loc['sharpe_ratio', col] = (
                            returns.mean() / returns.std() * np.sqrt(252)
                        ) if returns.std() > 0 else 0
            
            return stats.round(4)
            
        except Exception as e:
            logger.error(f"Error calculating statistical summary: {e}")
            return pd.DataFrame()
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        try:
            cumulative = prices / prices.iloc[0]
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min() * 100
        except:
            return 0.0
    
    def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate data quality and return quality metrics.
        
        Args:
            data: Financial data to validate
        
        Returns:
            Dictionary with quality metrics
        """
        quality_checks = {}
        
        try:
            # Check for sufficient data points
            quality_checks['sufficient_data'] = len(data) >= 252  # At least 1 year
            
            # Check for missing values
            quality_checks['no_missing_values'] = not data.isnull().any().any()
            
            # Check for data continuity (no large gaps)
            date_diffs = data.index.to_series().diff().dt.days
            quality_checks['continuous_data'] = (date_diffs <= 7).all()  # Max 7-day gaps
            
            # Check for outliers in returns
            if 'Return' in data.columns:
                returns = data['Return'].dropna()
                z_scores = np.abs((returns - returns.mean()) / returns.std())
                quality_checks['no_extreme_outliers'] = (z_scores < 5).all()
            
            # Check OHLC consistency
            quality_checks['ohlc_consistent'] = (
                (data['High'] >= data['Low']).all() and
                (data['High'] >= data['Open']).all() and
                (data['High'] >= data['Close']).all() and
                (data['Low'] <= data['Open']).all() and
                (data['Low'] <= data['Close']).all()
            )
            
            logger.info(f"Data quality validation completed: {sum(quality_checks.values())}/{len(quality_checks)} checks passed")
            
        except Exception as e:
            logger.error(f"Error in data quality validation: {e}")
            quality_checks['validation_error'] = True
        
        return quality_checks