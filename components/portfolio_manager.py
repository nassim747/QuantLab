"""
Portfolio management component for QuantLab Professional.
"""
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from portfolio.optimization import PortfolioOptimizer, OptimizationResult
from components.data_handler import DataHandler
from utils.logger import get_logger

logger = get_logger(__name__)


class PortfolioManager:
    """Manages portfolio optimization and multi-asset strategies."""
    
    def __init__(self):
        self.data_handler = DataHandler()
        self.optimizers = {}
        self.results = {}
        logger.info("PortfolioManager initialized")
    
    def optimize_portfolio(self,
                          tickers: List[str],
                          period_years: int = 5) -> Dict[str, OptimizationResult]:
        """
        Optimize portfolio for given assets.
        
        Args:
            tickers: List of asset tickers
            period_years: Years of historical data to use
        
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Optimizing portfolio for {len(tickers)} assets")
        
        # Load data for all tickers
        multi_data = self.data_handler.load_multiple_tickers(
            tickers, days=period_years * 365
        )
        
        if len(multi_data) < 2:
            logger.error("Need at least 2 assets for portfolio optimization")
            return {}
        
        # Create returns matrix
        returns_data = pd.DataFrame()
        for ticker, data in multi_data.items():
            if len(data) > 252:  # At least 1 year of data
                returns_data[ticker] = data['Close'].pct_change()
        
        returns_data = returns_data.dropna()
        
        if len(returns_data.columns) < 2:
            logger.error("Insufficient clean data for optimization")
            return {}
        
        # Initialize optimizer
        optimizer = PortfolioOptimizer(returns_data, risk_free_rate=0.02)
        
        results = {}
        
        # Run different optimization strategies
        optimization_strategies = [
            ("Max Sharpe", optimizer.max_sharpe_optimization),
            ("Min Variance", optimizer.min_variance_optimization),
            ("Risk Parity", optimizer.risk_parity_optimization)
        ]
        
        for strategy_name, optimization_func in optimization_strategies:
            try:
                logger.info(f"Running {strategy_name} optimization")
                result = optimization_func()
                results[strategy_name] = result
                
                logger.info(f"{strategy_name} - Sharpe: {result.sharpe_ratio:.2f}, "
                           f"Return: {result.expected_return:.2%}, "
                           f"Vol: {result.volatility:.2%}")
                
            except Exception as e:
                logger.error(f"Error in {strategy_name} optimization: {e}")
                continue
        
        # Generate efficient frontier
        try:
            efficient_frontier = optimizer.efficient_frontier(n_portfolios=50)
            results['efficient_frontier'] = efficient_frontier
            
        except Exception as e:
            logger.error(f"Error generating efficient frontier: {e}")
        
        self.results = results
        return results
    
    def get_portfolio_composition(self, result: OptimizationResult, tickers: List[str]) -> Dict[str, float]:
        """Get portfolio composition as ticker -> weight mapping."""
        return dict(zip(tickers, result.weights))
    
    def calculate_portfolio_metrics(self, 
                                   weights: np.ndarray,
                                   returns_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive portfolio metrics."""
        portfolio_returns = returns_data.dot(weights)
        
        # Basic metrics
        annual_return = portfolio_returns.mean() * 252
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # VaR and CVaR
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95
        }