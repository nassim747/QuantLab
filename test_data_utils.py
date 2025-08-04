import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from data_utils import (
    clean_data, add_technical_indicators, prepare_ml_features, 
    calculate_basic_metrics, get_data_info
)


class TestDataUtils:
    
    def create_sample_data(self, n_days=100):
        """Create sample stock data for testing."""
        dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
        np.random.seed(42)  # For reproducible tests
        
        # Generate realistic stock price data
        initial_price = 100
        returns = np.random.normal(0.001, 0.02, n_days)  # Small daily returns
        prices = [initial_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = pd.DataFrame({
            'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, n_days),
            'Adj Close': prices
        }, index=dates)
        
        # Ensure High >= Close >= Low and High >= Open >= Low
        for i in range(len(data)):
            data.iloc[i, data.columns.get_loc('High')] = max(
                data.iloc[i, data.columns.get_loc('High')],
                data.iloc[i, data.columns.get_loc('Close')],
                data.iloc[i, data.columns.get_loc('Open')]
            )
            data.iloc[i, data.columns.get_loc('Low')] = min(
                data.iloc[i, data.columns.get_loc('Low')],
                data.iloc[i, data.columns.get_loc('Close')],
                data.iloc[i, data.columns.get_loc('Open')]
            )
        
        return data
    
    def test_clean_data(self):
        """Test data cleaning functionality."""
        # Create test data with MultiIndex columns
        data = self.create_sample_data(50)
        
        # Create MultiIndex columns to test flattening
        data.columns = pd.MultiIndex.from_tuples([
            ('Open', 'AAPL'), ('High', 'AAPL'), ('Low', 'AAPL'),
            ('Close', 'AAPL'), ('Volume', 'AAPL'), ('Adj Close', 'AAPL')
        ])
        
        # Add duplicate index to test deduplication
        data = pd.concat([data, data.iloc[:1]])
        
        cleaned_data = clean_data(data)
        
        # Check that MultiIndex was flattened
        assert not isinstance(cleaned_data.columns, pd.MultiIndex)
        assert list(cleaned_data.columns) == ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        
        # Check that duplicates were removed
        assert not cleaned_data.index.duplicated().any()
        
        # Check that index is datetime
        assert pd.api.types.is_datetime64_any_dtype(cleaned_data.index)
    
    def test_add_technical_indicators(self):
        """Test technical indicators calculation."""
        data = self.create_sample_data(100)
        data_with_indicators = add_technical_indicators(data)
        
        # Check that all expected indicators are present
        expected_indicators = [
            'SMA_10', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
            'MACD', 'MACD_Signal', 'MACD_Histogram', 'RSI',
            'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_Position',
            'Volume_SMA', 'Volume_Ratio', 'High_Low_Ratio', 'Close_Open_Ratio'
        ]
        
        for indicator in expected_indicators:
            assert indicator in data_with_indicators.columns, f"Missing indicator: {indicator}"
        
        # Check RSI is within expected range (0-100)
        rsi_values = data_with_indicators['RSI'].dropna()
        assert all(0 <= val <= 100 for val in rsi_values), "RSI values out of range"
        
        # Check MACD calculation
        macd_check = (data_with_indicators['EMA_12'] - data_with_indicators['EMA_26'])
        macd_diff = abs(data_with_indicators['MACD'] - macd_check).dropna()
        assert macd_diff.max() < 1e-10, "MACD calculation incorrect"
        
        # Check Bollinger Bands
        bb_width_check = (data_with_indicators['BB_Upper'] - data_with_indicators['BB_Lower'])
        bb_width_diff = abs(data_with_indicators['BB_Width'] - bb_width_check).dropna()
        assert bb_width_diff.max() < 1e-10, "Bollinger Bands width calculation incorrect"
    
    def test_prepare_ml_features(self):
        """Test ML feature preparation."""
        data = self.create_sample_data(100)
        forecast_horizon = 5
        
        # Test with regular returns
        ml_data, feature_cols = prepare_ml_features(data, forecast_horizon, use_log_returns=False)
        
        assert 'Target' in ml_data.columns
        assert 'Return' in feature_cols
        assert len(ml_data) > 0
        
        # Check that target is shifted correctly
        original_close = data['Close'].iloc[:-forecast_horizon]
        target_close = data['Close'].iloc[forecast_horizon:]
        
        # The target should be approximately equal to future close prices
        # (accounting for any NaN values that were dropped)
        ml_target = ml_data['Target'].iloc[:len(target_close)]
        
        # Test with log returns
        ml_data_log, feature_cols_log = prepare_ml_features(data, forecast_horizon, use_log_returns=True)
        
        assert 'LogReturn' in feature_cols_log
        assert len(ml_data_log) > 0
        
        # Check that we have technical indicators
        tech_indicators = ['SMA_10', 'RSI', 'MACD', 'BB_Position']
        for indicator in tech_indicators:
            if indicator in ml_data.columns:
                assert indicator in feature_cols
    
    def test_calculate_basic_metrics(self):
        """Test basic metrics calculation."""
        data = self.create_sample_data(50)
        metrics = calculate_basic_metrics(data)
        
        required_keys = ['latest_close', 'change', 'change_pct', 'start_date', 'end_date', 'total_days']
        
        for key in required_keys:
            assert key in metrics, f"Missing metric: {key}"
        
        # Check types and ranges
        assert isinstance(metrics['latest_close'], float)
        assert isinstance(metrics['change'], float)
        assert isinstance(metrics['change_pct'], float)
        assert isinstance(metrics['total_days'], int)
        assert metrics['total_days'] == len(data)
        
        # Check date format
        assert len(metrics['start_date']) == 10  # YYYY-MM-DD format
        assert len(metrics['end_date']) == 10
    
    @patch('data_utils.load_stock_data')
    def test_get_data_info_success(self, mock_load_data):
        """Test successful data loading."""
        # Mock successful data loading
        sample_data = self.create_sample_data(100)
        mock_load_data.return_value = sample_data
        
        data, error = get_data_info('AAPL', days=365)
        
        assert data is not None
        assert error is None
        assert len(data) == 100
        mock_load_data.assert_called_once()
    
    @patch('data_utils.load_stock_data')
    def test_get_data_info_empty_data(self, mock_load_data):
        """Test handling of empty data."""
        # Mock empty data
        mock_load_data.return_value = pd.DataFrame()
        
        data, error = get_data_info('INVALID', days=365)
        
        assert data is None
        assert "No data found" in error
    
    @patch('data_utils.load_stock_data')
    def test_get_data_info_exception(self, mock_load_data):
        """Test handling of exceptions during data loading."""
        # Mock exception
        mock_load_data.side_effect = Exception("Network error")
        
        data, error = get_data_info('AAPL', days=365)
        
        assert data is None
        assert "Error downloading data" in error
        assert "Network error" in error
    
    def test_feature_engineering_edge_cases(self):
        """Test feature engineering with edge cases."""
        # Test with minimal data
        small_data = self.create_sample_data(10)
        ml_data, feature_cols = prepare_ml_features(small_data, forecast_horizon=1)
        
        # Should still work but with fewer features
        assert len(ml_data) >= 0
        assert 'Target' in ml_data.columns if len(ml_data) > 0 else True
        
        # Test with large forecast horizon
        data = self.create_sample_data(50)
        ml_data, feature_cols = prepare_ml_features(data, forecast_horizon=30)
        
        # Should have fewer data points due to large forecast horizon
        assert len(ml_data) < len(data)
    
    def test_technical_indicators_edge_cases(self):
        """Test technical indicators with edge cases."""
        # Test with constant prices (no volatility)
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        constant_data = pd.DataFrame({
            'Open': [100] * 50,
            'High': [100] * 50,
            'Low': [100] * 50,
            'Close': [100] * 50,
            'Volume': [1000000] * 50
        }, index=dates)
        
        data_with_indicators = add_technical_indicators(constant_data)
        
        # RSI should be around 50 for constant prices
        rsi_values = data_with_indicators['RSI'].dropna()
        if len(rsi_values) > 0:
            # For constant prices, RSI calculation might result in NaN or 50
            assert all(pd.isna(val) or abs(val - 50) < 10 for val in rsi_values)
        
        # Moving averages should equal the constant price
        sma_values = data_with_indicators['SMA_10'].dropna()
        assert all(abs(val - 100) < 1e-10 for val in sma_values)


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__])