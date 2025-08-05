"""
Strategy management component for QuantLab Professional.
"""
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from backtesting.advanced_backtester import AdvancedBacktester, PositionSizing
from backtester import StrategySignalGenerator
from utils.logger import get_logger

logger = get_logger(__name__)


class StrategyManager:
    """Manages trading strategies and backtesting."""
    
    def __init__(self):
        self.strategies = {}
        self.results = {}
        logger.info("StrategyManager initialized")
    
    def run_backtest(self,
                    data: pd.DataFrame,
                    strategy_type: str,
                    ml_results: Dict = None,
                    config: Dict = None) -> Dict[str, Any]:
        """
        Run backtest for specified strategy.
        
        Args:
            data: Market data
            strategy_type: Type of strategy to test
            ml_results: ML model results if using ML strategy
            config: Configuration parameters
        
        Returns:
            Backtest results
        """
        if config is None:
            config = {}
        
        logger.info(f"Running backtest for {strategy_type}")
        
        # Configure backtester
        position_sizing_map = {
            "Volatility Adjusted": PositionSizing.VOLATILITY_ADJUSTED,
            "Fixed Percentage": PositionSizing.FIXED_PERCENTAGE,
            "Kelly Criterion": PositionSizing.KELLY_CRITERION,
            "Risk Parity": PositionSizing.VOLATILITY_ADJUSTED  # Default fallback
        }
        
        position_sizing = position_sizing_map.get(
            config.get('position_sizing', 'Volatility Adjusted'),
            PositionSizing.VOLATILITY_ADJUSTED
        )
        
        backtester = AdvancedBacktester(
            initial_capital=config.get('initial_capital', 10000),
            transaction_cost=0.001,
            slippage=0.0005,
            position_sizing=position_sizing,
            max_position_pct=config.get('max_position_pct', 0.2),
            stop_loss_pct=config.get('stop_loss_pct', 0.05),
            take_profit_pct=config.get('take_profit_pct', 0.10)
        )
        
        # Generate signals based on strategy type
        signals = self._generate_signals(data, strategy_type, ml_results)
        
        if signals is None:
            logger.error(f"Failed to generate signals for {strategy_type}")
            return {}
        
        # Run backtest
        try:
            results = backtester.run_backtest(data, signals, ticker="BACKTEST")
            logger.info(f"Backtest completed - Total Return: {results['metrics']['total_return']:.2%}")
            return results
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return {}
    
    def _generate_signals(self,
                         data: pd.DataFrame,
                         strategy_type: str,
                         ml_results: Dict = None) -> Optional[pd.Series]:
        """Generate trading signals based on strategy type."""
        
        if strategy_type == "ML Prediction Based":
            if not ml_results:
                logger.warning("ML results required for ML strategy")
                return None
            
            # Use the best ML model for signals
            best_model = max(
                ml_results.items(),
                key=lambda x: x[1].get('metrics', {}).get('r2', -999)
            )
            
            if best_model:
                predictions = best_model[1]['predictions']
                test_indices = best_model[1]['test_data']['X_test'].index
                
                # Create signals based on predictions
                signals = pd.Series(0, index=data.index)
                
                for i, (idx, pred) in enumerate(zip(test_indices, predictions)):
                    if idx in data.index:
                        current_price = data.loc[idx, 'Close']
                        # Buy if predicted return > 1%, sell if < -1%
                        if pred > 0.01:
                            signals[idx] = 1
                        elif pred < -0.01:
                            signals[idx] = -1
                
                return signals
        
        elif strategy_type == "Moving Average Crossover":
            return StrategySignalGenerator.momentum_signals(data, short_window=10, long_window=20)
        
        elif strategy_type == "Mean Reversion":
            return StrategySignalGenerator.mean_reversion_signals(data, window=20, num_std=2)
        
        elif strategy_type == "RSI Strategy":
            return StrategySignalGenerator.rsi_signals(data, rsi_period=14, oversold=30, overbought=70)
        
        else:
            logger.warning(f"Unknown strategy type: {strategy_type}")
            return None