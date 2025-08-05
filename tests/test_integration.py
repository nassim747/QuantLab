"""
Integration tests for QuantLab Professional.
Tests the complete pipeline from data loading to portfolio optimization.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Import components to test
from components.data_handler import DataHandler
from ml_models import MLModelTrainer
from backtesting.advanced_backtester import AdvancedBacktester, PositionSizing
from portfolio.optimization import PortfolioOptimizer
from data_utils import prepare_ml_features


class TestIntegrationPipeline:
    """Test the complete QuantLab pipeline integration."""
    
    @pytest.fixture
    def sample_stock_data(self):
        """Create realistic sample stock data for testing."""
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        n_days = len(dates)
        
        # Generate realistic price movements
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns
        
        # Create price series with realistic properties
        prices = [100.0]  # Starting price
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLCV data with realistic relationships
        data = pd.DataFrame(index=dates[:len(prices)])
        data['Close'] = prices
        
        # Generate OHLC with proper relationships
        for i, price in enumerate(prices):
            volatility = 0.02
            daily_range = price * volatility * np.random.uniform(0.5, 2.0)
            
            high = price + daily_range * np.random.uniform(0.0, 1.0)
            low = price - daily_range * np.random.uniform(0.0, 1.0)
            open_price = low + (high - low) * np.random.uniform(0.0, 1.0)
            
            data.loc[data.index[i], 'Open'] = open_price
            data.loc[data.index[i], 'High'] = max(high, price, open_price)
            data.loc[data.index[i], 'Low'] = min(low, price, open_price)
            data.loc[data.index[i], 'Volume'] = np.random.randint(1000000, 10000000)
        
        return data
    
    @pytest.fixture
    def multi_asset_data(self):
        """Create multi-asset data for portfolio optimization tests."""
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        
        np.random.seed(42)
        
        # Create correlated returns (realistic for stocks)
        base_returns = np.random.normal(0.0005, 0.015, len(dates))
        
        data = {}
        for i, ticker in enumerate(tickers):
            # Add some correlation and individual noise
            correlation_factor = 0.3 + 0.4 * np.random.random()
            individual_returns = (
                base_returns * correlation_factor + 
                np.random.normal(0, 0.01, len(dates)) * (1 - correlation_factor)
            )
            
            # Generate price series
            prices = [100.0 * (1 + i * 0.1)]  # Different starting prices
            for ret in individual_returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            data[ticker] = pd.DataFrame({
                'Close': prices,
                'Return': individual_returns
            }, index=dates[:len(prices)])
        
        return data
    
    def test_complete_data_pipeline(self, sample_stock_data):
        """Test complete data processing pipeline."""
        data_handler = DataHandler()
        
        # Test data processing
        processed_data = data_handler.process_data(sample_stock_data)
        
        # Verify technical indicators were added
        expected_indicators = ['SMA_20', 'RSI', 'MACD', 'BB_Upper', 'Vol_20']
        for indicator in expected_indicators:
            assert indicator in processed_data.columns, f"Missing indicator: {indicator}"
        
        # Test data quality validation
        quality = data_handler.validate_data_quality(processed_data)
        assert quality['sufficient_data'], "Should have sufficient data"
        assert quality['ohlc_consistent'], "OHLC data should be consistent"
        
        # Test summary metrics calculation
        metrics = data_handler.calculate_summary_metrics(processed_data)
        assert 'latest_close' in metrics
        assert 'volatility_20d' in metrics
        assert metrics['total_days'] == len(processed_data)
    
    def test_ml_pipeline_integration(self, sample_stock_data):
        """Test ML pipeline with realistic data."""
        # Prepare features (using the fixed return-based approach)
        ml_data, feature_cols = prepare_ml_features(
            sample_stock_data, 
            forecast_horizon=5, 
            use_log_returns=True
        )
        
        assert len(feature_cols) > 10, "Should have multiple features"
        assert 'Target' in ml_data.columns, "Should have target variable"
        
        # Test that target is actually returns, not prices
        target_values = ml_data['Target'].dropna()
        assert abs(target_values.mean()) < 0.1, "Target should be centered around 0 (returns)"
        assert target_values.std() < 1.0, "Target volatility should be reasonable"
        
        # Test ML training
        trainer = MLModelTrainer(use_scaling=True)
        
        # Split data
        split_idx = int(len(ml_data) * 0.8)
        X_train = ml_data[feature_cols].iloc[:split_idx]
        X_test = ml_data[feature_cols].iloc[split_idx:]
        y_train = ml_data['Target'].iloc[:split_idx]
        y_test = ml_data['Target'].iloc[split_idx:]
        
        # Remove NaN values
        train_mask = ~(X_train.isnull().any(axis=1) | y_train.isnull())
        test_mask = ~(X_test.isnull().any(axis=1) | y_test.isnull())
        
        X_train_clean = X_train[train_mask]
        y_train_clean = y_train[train_mask]
        X_test_clean = X_test[test_mask]
        y_test_clean = y_test[test_mask]
        
        if len(X_train_clean) < 50:
            pytest.skip("Not enough clean training data")
        
        # Train model
        model = trainer.train_model(
            X_train_clean, y_train_clean, 
            "Random Forest Regressor", 
            n_estimators=50  # Faster for testing
        )
        
        assert trainer.is_trained, "Model should be marked as trained"
        
        # Evaluate model
        metrics, predictions = trainer.evaluate_model(X_test_clean, y_test_clean)
        
        # Check that metrics are reasonable
        assert 'rmse' in metrics and metrics['rmse'] > 0
        assert 'hit_rate' in metrics and 0 <= metrics['hit_rate'] <= 1
        assert len(predictions) == len(y_test_clean)
    
    def test_backtesting_integration(self, sample_stock_data):
        """Test advanced backtesting integration."""
        # Create simple momentum signals for testing
        data = sample_stock_data.copy()
        data['SMA_10'] = data['Close'].rolling(10).mean()
        data['SMA_20'] = data['Close'].rolling(20).mean()
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        signals[data['SMA_10'] > data['SMA_20']] = 1
        signals[data['SMA_10'] < data['SMA_20']] = -1
        
        # Test backtester
        backtester = AdvancedBacktester(
            initial_capital=10000,
            transaction_cost=0.001,
            slippage=0.0005,
            position_sizing=PositionSizing.VOLATILITY_ADJUSTED,
            max_position_pct=0.2,
            stop_loss_pct=0.05
        )
        
        # Run backtest
        results = backtester.run_backtest(data, signals, ticker="TEST")
        
        # Verify results structure
        assert 'equity_curve' in results
        assert 'trades' in results
        assert 'metrics' in results
        
        # Check metrics
        metrics = results['metrics']
        expected_metrics = [
            'total_return', 'sharpe_ratio', 'max_drawdown', 
            'win_rate', 'total_trades'
        ]
        for metric in expected_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
        
        # Verify equity curve makes sense
        equity_curve = results['equity_curve']
        assert len(equity_curve) > 0, "Should have equity curve data"
        assert equity_curve.iloc[0] > 0, "Should start with positive capital"
        
        # Verify realistic performance bounds
        total_return = metrics['total_return']
        assert -0.9 < total_return < 10.0, f"Unrealistic total return: {total_return}"
    
    def test_portfolio_optimization_integration(self, multi_asset_data):
        """Test portfolio optimization with multiple assets."""
        # Create returns data for optimization
        returns_data = pd.DataFrame()
        for ticker, data in multi_asset_data.items():
            returns_data[ticker] = data['Return']
        
        # Remove any NaN values
        returns_data = returns_data.dropna()
        
        if len(returns_data) < 252:  # Need at least 1 year of data
            pytest.skip("Not enough data for portfolio optimization")
        
        # Test portfolio optimizer
        optimizer = PortfolioOptimizer(returns_data, risk_free_rate=0.02)
        
        # Test different optimization strategies
        max_sharpe_result = optimizer.max_sharpe_optimization()
        assert abs(max_sharpe_result.weights.sum() - 1.0) < 1e-6, "Weights should sum to 1"
        assert all(w >= -1e-6 for w in max_sharpe_result.weights), "Weights should be non-negative"
        
        min_var_result = optimizer.min_variance_optimization()
        assert abs(min_var_result.weights.sum() - 1.0) < 1e-6, "Weights should sum to 1"
        
        risk_parity_result = optimizer.risk_parity_optimization()
        assert abs(risk_parity_result.weights.sum() - 1.0) < 1e-6, "Weights should sum to 1"
        
        # Test efficient frontier generation
        frontier_data = optimizer.efficient_frontier(n_portfolios=50)
        assert len(frontier_data) > 0, "Should generate frontier portfolios"
        assert 'Return' in frontier_data.columns
        assert 'Volatility' in frontier_data.columns
        assert 'Sharpe' in frontier_data.columns
        
        # Verify frontier makes sense (higher return generally means higher risk)
        assert frontier_data['Volatility'].min() >= 0, "Volatility should be non-negative"
        assert frontier_data['Return'].std() > 0, "Should have return variation"
    
    def test_error_handling_integration(self, sample_stock_data):
        """Test error handling throughout the pipeline."""
        # Test with corrupted data
        corrupted_data = sample_stock_data.copy()
        corrupted_data.iloc[10:20, :] = np.nan  # Insert NaN values
        corrupted_data.iloc[50:60, 0] = -100  # Insert negative prices
        
        data_handler = DataHandler()
        
        # Should handle corrupted data gracefully
        processed_data = data_handler.process_data(corrupted_data)
        assert len(processed_data) > 0, "Should still return some processed data"
        
        # Test quality validation with bad data
        quality = data_handler.validate_data_quality(corrupted_data)
        assert not quality['no_missing_values'], "Should detect missing values"
        
        # Test ML with insufficient data
        small_data = sample_stock_data.iloc[:30].copy()  # Very small dataset
        ml_data, features = prepare_ml_features(small_data)
        
        trainer = MLModelTrainer()
        
        # Should handle small datasets gracefully
        if len(ml_data.dropna()) < 10:
            with pytest.raises((ValueError, IndexError)):
                trainer.train_model(
                    ml_data[features].dropna(), 
                    ml_data['Target'].dropna(), 
                    "Random Forest Regressor"
                )
    
    def test_performance_benchmarks(self, sample_stock_data):
        """Test performance requirements."""
        import time
        
        data_handler = DataHandler()
        
        # Test data processing speed
        start_time = time.time()
        processed_data = data_handler.process_data(sample_stock_data)
        processing_time = time.time() - start_time
        
        assert processing_time < 5.0, f"Data processing too slow: {processing_time}s"
        
        # Test ML training speed
        ml_data, features = prepare_ml_features(processed_data)
        clean_data = ml_data.dropna()
        
        if len(clean_data) > 100:
            trainer = MLModelTrainer()
            
            X = clean_data[features]
            y = clean_data['Target']
            
            start_time = time.time()
            trainer.train_model(X, y, "Linear Regression")
            training_time = time.time() - start_time
            
            assert training_time < 10.0, f"ML training too slow: {training_time}s"
    
    def test_financial_scenario_edge_cases(self, sample_stock_data):
        """Test with realistic financial edge cases."""
        # Test market crash scenario
        crash_data = sample_stock_data.copy()
        crash_start = len(crash_data) // 2
        crash_end = crash_start + 20
        
        # Simulate 30% drop over 20 days
        for i in range(crash_start, min(crash_end, len(crash_data))):
            if i > 0:
                crash_data.iloc[i, crash_data.columns.get_loc('Close')] = (
                    crash_data.iloc[i-1, crash_data.columns.get_loc('Close')] * 0.985
                )
        
        # Test backtester with crash scenario
        signals = pd.Series(1, index=crash_data.index)  # Always long
        
        backtester = AdvancedBacktester(
            initial_capital=10000,
            stop_loss_pct=0.05  # 5% stop loss should help
        )
        
        results = backtester.run_backtest(crash_data, signals, ticker="CRASH_TEST")
        
        # Should have some protection from stop losses
        assert results['metrics']['max_drawdown'] > -0.5, "Stop losses should limit drawdown"
        assert len(results['trades']) > 0, "Should have executed some trades"
        
        # Test with high volatility period
        volatile_data = sample_stock_data.copy()
        volatile_data['Close'] *= (1 + np.random.normal(0, 0.05, len(volatile_data)))
        
        data_handler = DataHandler()
        quality = data_handler.validate_data_quality(volatile_data)
        
        # Should still pass basic quality checks despite volatility
        assert quality['sufficient_data'], "Should handle volatile data"


class TestMarketRegimeDetection:
    """Test market regime detection and adaptation."""
    
    def test_bull_bear_market_detection(self):
        """Test detection of different market regimes."""
        # Create bull market data (trending up)
        dates = pd.date_range('2020-01-01', '2022-12-31', freq='D')
        prices = [100]
        
        for i in range(1, len(dates)):
            # Bull market: positive drift with normal volatility
            daily_return = np.random.normal(0.001, 0.015)  # 0.1% daily drift
            prices.append(prices[-1] * (1 + daily_return))
        
        bull_data = pd.DataFrame({
            'Close': prices,
            'Return': pd.Series(prices).pct_change()
        }, index=dates)
        
        # Test that our algorithms detect uptrend
        sma_20 = bull_data['Close'].rolling(20).mean()
        sma_50 = bull_data['Close'].rolling(50).mean()
        
        # In bull market, shorter MA should be above longer MA most of the time
        uptrend_days = (sma_20 > sma_50).sum()
        total_valid_days = len(sma_50.dropna())
        uptrend_ratio = uptrend_days / total_valid_days
        
        assert uptrend_ratio > 0.6, f"Should detect bull market trend: {uptrend_ratio}"
    
    def test_sideways_market_handling(self):
        """Test handling of sideways/range-bound markets."""
        # Create sideways market (oscillating around mean)
        dates = pd.date_range('2020-01-01', '2022-12-31', freq='D')
        base_price = 100
        prices = []
        
        for i in range(len(dates)):
            # Oscillate around base price with no trend
            cycle_component = 10 * np.sin(2 * np.pi * i / 252)  # Annual cycle
            noise = np.random.normal(0, 2)
            prices.append(base_price + cycle_component + noise)
        
        sideways_data = pd.DataFrame({
            'Close': prices,
            'Return': pd.Series(prices).pct_change()
        }, index=dates)
        
        # Test mean reversion properties
        returns = sideways_data['Return'].dropna()
        
        # In sideways market, returns should have minimal autocorrelation
        autocorr = returns.autocorr()
        assert abs(autocorr) < 0.2, f"Sideways market should have low autocorrelation: {autocorr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])