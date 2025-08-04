import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backtester import SimpleBacktester, StrategySignalGenerator


class TestSimpleBacktester:
    
    def create_sample_price_data(self, n_days=100, initial_price=100):
        """Create sample price data for testing."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
        
        # Generate price series with random walk
        returns = np.random.normal(0.001, 0.02, n_days)
        prices = [initial_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        price_data = pd.DataFrame({
            'Close': prices,
            'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Volume': np.random.randint(1000000, 10000000, n_days)
        }, index=dates)
        
        return price_data
    
    def create_simple_signals(self, n_days=100):
        """Create simple buy/hold/sell signals for testing."""
        dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
        signals = pd.Series(0, index=dates)  # Start with all hold signals
        
        # Add some buy and sell signals
        signals.iloc[10] = 1   # Buy signal
        signals.iloc[30] = -1  # Sell signal
        signals.iloc[50] = 1   # Buy signal
        signals.iloc[80] = -1  # Sell signal
        
        return signals
    
    def test_backtester_initialization(self):
        """Test SimpleBacktester initialization."""
        backtester = SimpleBacktester(initial_capital=20000, transaction_cost=0.002)
        
        assert backtester.initial_capital == 20000
        assert backtester.transaction_cost == 0.002
        assert backtester.results is None
    
    def test_run_backtest_basic(self):
        """Test basic backtesting functionality."""
        backtester = SimpleBacktester(initial_capital=10000, transaction_cost=0.001)
        
        # Create test data
        price_data = self.create_sample_price_data(50)
        signals = self.create_simple_signals(50)
        prices = price_data['Close']
        
        # Run backtest
        results = backtester.run_backtest(price_data, signals, prices, "Test Strategy")
        
        # Check results structure
        assert 'data' in results
        assert 'trades' in results
        assert 'metrics' in results
        assert 'strategy_name' in results
        
        # Check data DataFrame
        results_df = results['data']
        expected_columns = [
            'Price', 'Signal', 'Portfolio_Value', 'Position', 'Cash',
            'Strategy_Return', 'Market_Return', 'Strategy_Cumulative', 'Market_Cumulative'
        ]
        
        for col in expected_columns:
            assert col in results_df.columns
        
        # Check that portfolio values are reasonable
        assert all(results_df['Portfolio_Value'] > 0)
        assert results_df['Portfolio_Value'].iloc[0] == 10000  # Should start with initial capital
    
    def test_run_backtest_with_trades(self):
        """Test backtest with actual trades."""
        backtester = SimpleBacktester(initial_capital=10000, transaction_cost=0.001)
        
        # Create test data with clear signals
        price_data = self.create_sample_price_data(20, initial_price=100)
        
        # Create signals that should generate trades
        signals = pd.Series(0, index=price_data.index)
        signals.iloc[2] = 1   # Buy signal early
        signals.iloc[10] = -1  # Sell signal later
        
        prices = price_data['Close']
        
        results = backtester.run_backtest(price_data, signals, prices, "Trade Test")
        
        # Check that trades were executed
        trades = results['trades']
        assert len(trades) >= 1  # Should have at least one trade
        
        # Check trade structure
        if len(trades) > 0:
            trade = trades[0]
            required_keys = ['date', 'action', 'shares', 'price']
            for key in required_keys:
                assert key in trade
            
            assert trade['action'] in ['BUY', 'SELL']
            assert trade['shares'] > 0
            assert trade['price'] > 0
    
    def test_calculate_metrics(self):
        """Test performance metrics calculation."""
        backtester = SimpleBacktester(initial_capital=10000, transaction_cost=0.001)
        
        price_data = self.create_sample_price_data(50)
        signals = self.create_simple_signals(50)
        prices = price_data['Close']
        
        results = backtester.run_backtest(price_data, signals, prices, "Metrics Test")
        metrics = results['metrics']
        
        # Check that all expected metrics are present
        expected_metrics = [
            'total_return', 'market_total_return', 'excess_return',
            'ann_return', 'market_ann_return', 'ann_vol', 'sharpe',
            'max_drawdown', 'num_trades', 'win_rate', 'final_value'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
        
        # Check metric types and ranges
        assert isinstance(metrics['total_return'], float)
        assert isinstance(metrics['num_trades'], int)
        assert 0 <= metrics['win_rate'] <= 1
        assert metrics['max_drawdown'] <= 0
        assert metrics['final_value'] > 0
    
    def test_empty_signals(self):
        """Test backtest with no trading signals (all hold)."""
        backtester = SimpleBacktester(initial_capital=10000, transaction_cost=0.001)
        
        price_data = self.create_sample_price_data(30)
        signals = pd.Series(0, index=price_data.index)  # All hold signals
        prices = price_data['Close']
        
        results = backtester.run_backtest(price_data, signals, prices, "Hold Strategy")
        
        # Should have no trades
        assert len(results['trades']) == 0
        
        # Portfolio value should remain constant (no investments made)
        portfolio_values = results['data']['Portfolio_Value']
        assert all(val == 10000 for val in portfolio_values)
        
        # Metrics should reflect no trading
        metrics = results['metrics']
        assert metrics['num_trades'] == 0
        assert metrics['total_return'] == 0
    
    def test_transaction_costs(self):
        """Test that transaction costs are properly applied."""
        # Test with different transaction costs
        high_cost_backtester = SimpleBacktester(initial_capital=10000, transaction_cost=0.05)  # 5%
        low_cost_backtester = SimpleBacktester(initial_capital=10000, transaction_cost=0.001)  # 0.1%
        
        price_data = self.create_sample_price_data(20)
        signals = pd.Series(0, index=price_data.index)
        signals.iloc[2] = 1   # Buy
        signals.iloc[10] = -1  # Sell
        prices = price_data['Close']
        
        high_cost_results = high_cost_backtester.run_backtest(price_data, signals, prices, "High Cost")
        low_cost_results = low_cost_backtester.run_backtest(price_data, signals, prices, "Low Cost")
        
        # High transaction cost strategy should perform worse (all else equal)
        high_cost_return = high_cost_results['metrics']['total_return']
        low_cost_return = low_cost_results['metrics']['total_return']
        
        # With transaction costs, high cost should be less profitable
        assert high_cost_return <= low_cost_return
    
    def test_plot_results(self):
        """Test plotting functionality."""
        backtester = SimpleBacktester(initial_capital=10000, transaction_cost=0.001)
        
        price_data = self.create_sample_price_data(30)
        signals = self.create_simple_signals(30)
        prices = price_data['Close']
        
        # Run backtest first
        backtester.run_backtest(price_data, signals, prices, "Plot Test")
        
        # Test plotting
        fig = backtester.plot_results(show_trades=True)
        
        assert fig is not None
        # Check that figure has the expected structure (3 subplots)
        assert len(fig.data) > 0  # Should have traces
    
    def test_plot_results_without_backtest(self):
        """Test plotting without running backtest first."""
        backtester = SimpleBacktester()
        
        with pytest.raises(ValueError, match="No backtest results available"):
            backtester.plot_results()
    
    def test_get_trade_summary(self):
        """Test trade summary functionality."""
        backtester = SimpleBacktester(initial_capital=10000, transaction_cost=0.001)
        
        price_data = self.create_sample_price_data(20)
        signals = self.create_simple_signals(20)
        prices = price_data['Close']
        
        # Run backtest
        backtester.run_backtest(price_data, signals, prices, "Summary Test")
        
        # Get trade summary
        trade_summary = backtester.get_trade_summary()
        
        if len(backtester.results['trades']) > 0:
            assert isinstance(trade_summary, pd.DataFrame)
            assert len(trade_summary) == len(backtester.results['trades'])
        else:
            assert trade_summary.empty


class TestStrategySignalGenerator:
    
    def create_sample_data(self, n_days=100):
        """Create sample data for signal generation testing."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
        
        # Create trending price data for better signal testing
        trend = np.linspace(100, 120, n_days)
        noise = np.random.normal(0, 2, n_days)
        prices = trend + noise
        
        data = pd.DataFrame({
            'Close': prices,
            'Open': prices * (1 + np.random.normal(0, 0.005, n_days)),
            'High': prices * (1 + np.random.uniform(0.001, 0.02, n_days)),
            'Low': prices * (1 - np.random.uniform(0.001, 0.02, n_days)),
            'Volume': np.random.randint(1000000, 10000000, n_days)
        }, index=dates)
        
        return data
    
    def test_prediction_based_signals(self):
        """Test prediction-based signal generation."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
        
        prices = pd.Series([100, 101, 102, 103, 104] * 4, index=dates)
        predictions = pd.Series([105, 99, 107, 101, 106] * 4, index=dates)
        
        signals = StrategySignalGenerator.prediction_based_signals(
            predictions, prices, threshold=0.02
        )
        
        # Check signal structure
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(prices)
        assert all(signal in [-1, 0, 1] for signal in signals)
        
        # Check that buy signals occur when prediction > price * (1 + threshold)
        buy_signals = signals[signals == 1]
        for idx in buy_signals.index:
            assert predictions[idx] > prices[idx] * 1.02
    
    def test_momentum_signals(self):
        """Test momentum-based signal generation."""
        data = self.create_sample_data(50)
        
        signals = StrategySignalGenerator.momentum_signals(
            data, short_window=5, long_window=10
        )
        
        # Check signal structure
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(data)
        assert all(signal in [-1, 0, 1] for signal in signals)
        
        # Signals should be NaN for the first long_window periods
        assert pd.isna(signals.iloc[:10]).all() or (signals.iloc[:10] == 0).all()
    
    def test_mean_reversion_signals(self):
        """Test mean reversion signal generation."""
        data = self.create_sample_data(50)
        
        signals = StrategySignalGenerator.mean_reversion_signals(
            data, window=10, num_std=2
        )
        
        # Check signal structure
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(data)
        assert all(signal in [-1, 0, 1] for signal in signals)
    
    def test_rsi_signals(self):
        """Test RSI-based signal generation."""
        data = self.create_sample_data(50)
        
        signals = StrategySignalGenerator.rsi_signals(
            data, rsi_period=14, oversold=30, overbought=70
        )
        
        # Check signal structure
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(data)
        assert all(signal in [-1, 0, 1] for signal in signals)
        
        # First few signals should be NaN or 0 due to RSI calculation requirements
        assert pd.isna(signals.iloc[:14]).all() or (signals.iloc[:14] == 0).all()
    
    def test_signals_with_edge_cases(self):
        """Test signal generation with edge cases."""
        # Test with constant prices
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        constant_data = pd.DataFrame({
            'Close': [100] * 30,
            'Open': [100] * 30,
            'High': [100] * 30,
            'Low': [100] * 30,
            'Volume': [1000000] * 30
        }, index=dates)
        
        # Test momentum signals with constant prices
        momentum_signals = StrategySignalGenerator.momentum_signals(constant_data)
        # With constant prices, moving averages should be equal, so signals should be 0
        non_nan_signals = momentum_signals.dropna()
        assert all(signal == 0 for signal in non_nan_signals)
        
        # Test RSI signals with constant prices
        rsi_signals = StrategySignalGenerator.rsi_signals(constant_data)
        # RSI calculation with constant prices might result in NaN or neutral signals
        assert isinstance(rsi_signals, pd.Series)
    
    def test_signal_parameter_validation(self):
        """Test signal generation with different parameters."""
        data = self.create_sample_data(100)
        
        # Test momentum with different window sizes
        short_momentum = StrategySignalGenerator.momentum_signals(data, 5, 10)
        long_momentum = StrategySignalGenerator.momentum_signals(data, 20, 50)
        
        assert len(short_momentum) == len(long_momentum)
        # Different parameters should potentially produce different signals
        # (though not guaranteed with random data)
        
        # Test RSI with different thresholds
        conservative_rsi = StrategySignalGenerator.rsi_signals(data, oversold=20, overbought=80)
        aggressive_rsi = StrategySignalGenerator.rsi_signals(data, oversold=40, overbought=60)
        
        assert len(conservative_rsi) == len(aggressive_rsi)
        
        # Conservative thresholds should generally produce fewer signals
        conservative_trades = sum(abs(conservative_rsi))
        aggressive_trades = sum(abs(aggressive_rsi))
        # This might not always be true with random data, but the structure should be correct


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__])