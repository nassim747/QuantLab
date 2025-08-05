"""
Portfolio optimization using Modern Portfolio Theory and advanced techniques.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OptimizationResult:
    """Container for portfolio optimization results."""
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float  # Value at Risk
    cvar_95: float  # Conditional Value at Risk


class PortfolioOptimizer:
    """
    Advanced portfolio optimization with multiple objectives and constraints.
    
    Implements:
    - Mean-Variance Optimization (Markowitz)
    - Risk Parity
    - Maximum Sharpe Ratio
    - Minimum Variance
    - Black-Litterman model
    - Risk budgeting
    """
    
    def __init__(self, returns_data: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        Initialize optimizer with returns data.
        
        Args:
            returns_data: DataFrame with returns for each asset (columns = assets)
            risk_free_rate: Annual risk-free rate for Sharpe calculations
        """
        self.returns = returns_data.dropna()
        self.assets = list(self.returns.columns)
        self.n_assets = len(self.assets)
        self.risk_free_rate = risk_free_rate
        
        # Calculate key statistics
        self.mean_returns = self.returns.mean() * 252  # Annualized
        self.cov_matrix = self.returns.cov() * 252  # Annualized
        self.corr_matrix = self.returns.corr()
        
        logger.info(f"Initialized optimizer for {self.n_assets} assets")
    
    def portfolio_performance(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate portfolio performance metrics.
        
        Args:
            weights: Asset weights (must sum to 1)
        
        Returns:
            Tuple of (expected_return, volatility, sharpe_ratio)
        """
        weights = np.array(weights)
        
        # Portfolio return
        portfolio_return = np.sum(self.mean_returns * weights)
        
        # Portfolio volatility
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        # Sharpe ratio
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
        
        return portfolio_return, portfolio_vol, sharpe
    
    def calculate_var_cvar(self, weights: np.ndarray, confidence: float = 0.05) -> Tuple[float, float]:
        """
        Calculate Value at Risk and Conditional Value at Risk.
        
        Args:
            weights: Portfolio weights
            confidence: Confidence level (0.05 for 95% VaR)
        
        Returns:
            Tuple of (VaR, CVaR)
        """
        portfolio_returns = self.returns.dot(weights)
        
        # Historical VaR
        var = np.percentile(portfolio_returns, confidence * 100)
        
        # CVaR (Expected Shortfall)
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        
        return var, cvar
    
    def max_sharpe_optimization(self, bounds: Optional[Tuple] = None) -> OptimizationResult:
        """
        Optimize for maximum Sharpe ratio.
        
        Args:
            bounds: Weight bounds for each asset (min, max)
        
        Returns:
            OptimizationResult with optimal weights and metrics
        """
        def negative_sharpe(weights):
            return -self.portfolio_performance(weights)[2]
        
        # Constraints
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds
        if bounds is None:
            bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Initial guess (equal weights)
        x0 = np.array([1/self.n_assets] * self.n_assets)
        
        # Optimize
        result = minimize(
            negative_sharpe,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            logger.warning("Max Sharpe optimization failed")
            return self._create_result(x0)
        
        return self._create_result(result.x)
    
    def min_variance_optimization(self, bounds: Optional[Tuple] = None) -> OptimizationResult:
        """Optimize for minimum variance."""
        def portfolio_variance(weights):
            return self.portfolio_performance(weights)[1] ** 2
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        if bounds is None:
            bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        x0 = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(
            portfolio_variance,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            logger.warning("Min variance optimization failed")
            return self._create_result(x0)
        
        return self._create_result(result.x)
    
    def risk_parity_optimization(self, bounds: Optional[Tuple] = None) -> OptimizationResult:
        """
        Optimize for risk parity (equal risk contribution).
        """
        def risk_budget_objective(weights):
            weights = np.array(weights)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            
            # Marginal risk contributions
            marginal_contrib = np.dot(self.cov_matrix, weights) / portfolio_vol
            
            # Risk contributions
            risk_contrib = weights * marginal_contrib / portfolio_vol
            
            # Target equal risk contribution
            target_contrib = np.ones(self.n_assets) / self.n_assets
            
            # Sum of squared differences from target
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        if bounds is None:
            bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        x0 = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(
            risk_budget_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            logger.warning("Risk parity optimization failed")
            return self._create_result(x0)
        
        return self._create_result(result.x)
    
    def efficient_frontier(self, n_portfolios: int = 100) -> pd.DataFrame:
        """
        Generate efficient frontier portfolios.
        
        Args:
            n_portfolios: Number of portfolios to generate
        
        Returns:
            DataFrame with efficient frontier data
        """
        # Get min and max return range
        min_ret = self.mean_returns.min()
        max_ret = self.mean_returns.max()
        target_returns = np.linspace(min_ret, max_ret, n_portfolios)
        
        efficient_portfolios = []
        
        for target_ret in target_returns:
            try:
                # Minimize variance for target return
                def portfolio_variance(weights):
                    return self.portfolio_performance(weights)[1] ** 2
                
                constraints = [
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                    {'type': 'eq', 'fun': lambda x: self.portfolio_performance(x)[0] - target_ret}
                ]
                
                bounds = tuple((0, 1) for _ in range(self.n_assets))
                x0 = np.array([1/self.n_assets] * self.n_assets)
                
                result = minimize(
                    portfolio_variance,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints
                )
                
                if result.success:
                    ret, vol, sharpe = self.portfolio_performance(result.x)
                    efficient_portfolios.append({
                        'Return': ret,
                        'Volatility': vol,
                        'Sharpe': sharpe,
                        'Weights': result.x
                    })
            except:
                continue
        
        return pd.DataFrame(efficient_portfolios)
    
    def black_litterman_optimization(self, 
                                   views: Dict[str, float],
                                   view_confidence: Dict[str, float],
                                   tau: float = 0.025) -> OptimizationResult:
        """
        Black-Litterman model implementation.
        
        Args:
            views: Dictionary of asset views {asset: expected_return}
            view_confidence: Dictionary of view confidences {asset: confidence}
            tau: Scaling factor for uncertainty
        
        Returns:
            OptimizationResult with Black-Litterman weights
        """
        # Market cap weights (equal for simplicity)
        market_weights = np.array([1/self.n_assets] * self.n_assets)
        
        # Implied equilibrium returns
        risk_aversion = 3.0  # Typical value
        pi = risk_aversion * np.dot(self.cov_matrix, market_weights)
        
        # Views setup
        P = np.zeros((len(views), self.n_assets))
        Q = np.zeros(len(views))
        
        for i, (asset, view_return) in enumerate(views.items()):
            if asset in self.assets:
                asset_idx = self.assets.index(asset)
                P[i, asset_idx] = 1
                Q[i] = view_return
        
        # View uncertainty matrix (Omega)
        omega_diag = []
        for i, (asset, confidence) in enumerate(view_confidence.items()):
            if asset in views:
                omega_diag.append(1 / confidence)
        
        Omega = np.diag(omega_diag)
        
        # Black-Litterman calculation
        tau_sigma = tau * self.cov_matrix
        
        # New expected returns
        M1 = np.linalg.inv(tau_sigma)
        M2 = np.dot(P.T, np.dot(np.linalg.inv(Omega), P))
        M3 = np.dot(np.linalg.inv(tau_sigma), pi)
        M4 = np.dot(P.T, np.dot(np.linalg.inv(Omega), Q))
        
        mu_bl = np.dot(np.linalg.inv(M1 + M2), M3 + M4)
        
        # New covariance matrix
        cov_bl = np.linalg.inv(M1 + M2)
        
        # Optimize with Black-Litterman inputs
        def negative_sharpe_bl(weights):
            portfolio_return = np.sum(mu_bl * weights)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_bl, weights)))
            return -(portfolio_return - self.risk_free_rate) / portfolio_vol
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        x0 = market_weights
        
        result = minimize(
            negative_sharpe_bl,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            logger.warning("Black-Litterman optimization failed")
            return self._create_result(market_weights)
        
        return self._create_result(result.x)
    
    def _create_result(self, weights: np.ndarray) -> OptimizationResult:
        """Create OptimizationResult from weights."""
        weights = np.array(weights)
        ret, vol, sharpe = self.portfolio_performance(weights)
        var, cvar = self.calculate_var_cvar(weights)
        
        # Calculate max drawdown
        portfolio_returns = self.returns.dot(weights)
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        return OptimizationResult(
            weights=weights,
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            var_95=var,
            cvar_95=cvar
        )
    
    def plot_efficient_frontier(self, highlight_portfolios: Dict[str, OptimizationResult] = None) -> go.Figure:
        """Plot efficient frontier with highlighted portfolios."""
        frontier_data = self.efficient_frontier()
        
        fig = go.Figure()
        
        # Efficient frontier
        fig.add_trace(go.Scatter(
            x=frontier_data['Volatility'],
            y=frontier_data['Return'],
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='blue', width=2)
        ))
        
        # Color by Sharpe ratio
        fig.add_trace(go.Scatter(
            x=frontier_data['Volatility'],
            y=frontier_data['Return'],
            mode='markers',
            marker=dict(
                color=frontier_data['Sharpe'],
                colorscale='Viridis',
                colorbar=dict(title='Sharpe Ratio'),
                size=6
            ),
            name='Sharpe Ratio',
            hovertemplate='Vol: %{x:.3f}<br>Return: %{y:.3f}<br>Sharpe: %{marker.color:.3f}'
        ))
        
        # Highlight special portfolios
        if highlight_portfolios:
            for name, result in highlight_portfolios.items():
                fig.add_trace(go.Scatter(
                    x=[result.volatility],
                    y=[result.expected_return],
                    mode='markers',
                    marker=dict(size=12, symbol='star'),
                    name=name,
                    hovertemplate=f'{name}<br>Vol: %{{x:.3f}}<br>Return: %{{y:.3f}}<br>Sharpe: {result.sharpe_ratio:.3f}'
                ))
        
        fig.update_layout(
            title='Efficient Frontier',
            xaxis_title='Volatility (Annual)',
            yaxis_title='Expected Return (Annual)',
            hovermode='closest'
        )
        
        return fig
    
    def plot_portfolio_composition(self, result: OptimizationResult, title: str = "Portfolio Composition") -> go.Figure:
        """Plot portfolio weights as pie chart."""
        fig = go.Figure(data=[go.Pie(
            labels=self.assets,
            values=result.weights,
            hole=0.3,
            textinfo='label+percent',
            textposition='outside'
        )])
        
        fig.update_layout(
            title=title,
            annotations=[dict(text=f'Sharpe: {result.sharpe_ratio:.3f}', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        return fig
    
    def compare_portfolios(self, portfolios: Dict[str, OptimizationResult]) -> pd.DataFrame:
        """Compare multiple portfolio optimization results."""
        comparison_data = []
        
        for name, result in portfolios.items():
            comparison_data.append({
                'Portfolio': name,
                'Expected Return': result.expected_return,
                'Volatility': result.volatility,
                'Sharpe Ratio': result.sharpe_ratio,
                'Max Drawdown': result.max_drawdown,
                'VaR 95%': result.var_95,
                'CVaR 95%': result.cvar_95
            })
        
        return pd.DataFrame(comparison_data)