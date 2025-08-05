"""
Advanced backtesting engine with proper risk management and position sizing.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from enum import Enum
from utils.logger import get_logger

logger = get_logger(__name__)


class PositionSizing(Enum):
    """Position sizing methods."""
    FIXED_AMOUNT = "fixed_amount"
    FIXED_PERCENTAGE = "fixed_percentage"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    KELLY_CRITERION = "kelly_criterion"


@dataclass
class Position:
    """Represents a trading position."""
    shares: float
    entry_price: float
    entry_date: pd.Timestamp
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class Trade:
    """Represents a completed trade."""
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: float
    pnl: float
    pnl_pct: float
    duration_days: int
    side: str  # 'long' or 'short'


class AdvancedBacktester:
    """
    Professional backtesting engine with realistic trading simulation.
    
    Features:
    - Proper position sizing with risk management
    - Transaction costs and slippage
    - Stop-loss and take-profit orders
    - Portfolio-level risk controls
    - Detailed performance analytics
    """
    
    def __init__(self, 
                 initial_capital: float = 10000.0,
                 transaction_cost: float = 0.001,
                 slippage: float = 0.0005,
                 position_sizing: PositionSizing = PositionSizing.VOLATILITY_ADJUSTED,
                 max_position_pct: float = 0.2,
                 min_position_size: float = 100.0,
                 stop_loss_pct: Optional[float] = 0.05,
                 take_profit_pct: Optional[float] = 0.10):
        
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.position_sizing = position_sizing
        self.max_position_pct = max_position_pct
        self.min_position_size = min_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Trading state
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.portfolio_value = []
        self.trades: List[Trade] = []
        self.equity_curve = pd.Series(dtype=float)
        
        logger.info(f"Initialized AdvancedBacktester with ${initial_capital:,.2f} capital")
    
    def calculate_position_size(self, 
                              price: float, 
                              signal_strength: float = 1.0,
                              volatility: Optional[float] = None) -> float:
        """
        Calculate position size based on configured method.
        
        Args:
            price: Current asset price
            signal_strength: Signal confidence (0-1)
            volatility: Asset volatility for risk-based sizing
        
        Returns:
            Number of shares to trade
        """
        available_capital = self.cash * self.max_position_pct
        
        if self.position_sizing == PositionSizing.FIXED_AMOUNT:
            shares = min(available_capital, self.min_position_size * 10) / price
            
        elif self.position_sizing == PositionSizing.FIXED_PERCENTAGE:
            shares = available_capital / price
            
        elif self.position_sizing == PositionSizing.VOLATILITY_ADJUSTED:
            if volatility is None or volatility == 0:
                volatility = 0.02  # Default 2% volatility
            
            # Risk-adjusted position size: target risk of ~2% account per trade.
            # Position dollar value scales inversely with daily volatility.
            # Example: at 2% vol we use full available_capital; at 4% vol we use half.
            target_risk_pct = 0.02  # 2% target risk budget
            risk_multiplier = target_risk_pct / max(volatility, 1e-6)
            risk_multiplier = min(risk_multiplier, 1.0)  # never exceed available_capital
            position_value = available_capital * risk_multiplier
            shares = position_value / price
            
        elif self.position_sizing == PositionSizing.KELLY_CRITERION:
            # Simplified Kelly criterion (requires win rate and avg win/loss)
            # For now, use conservative 0.25 Kelly fraction
            kelly_fraction = 0.25 * signal_strength
            shares = (self.cash * kelly_fraction) / price
        
        # Apply minimum position size constraint
        if shares * price < self.min_position_size:
            shares = 0
        
        # Ensure we don't exceed available cash
        shares = min(shares, (self.cash * 0.95) / price)  # 5% cash buffer
        
        return max(0, int(shares))
    
    def execute_trade(self, 
                     ticker: str,
                     price: float, 
                     shares: float, 
                     side: str,
                     date: pd.Timestamp) -> bool:
        """
        Execute a trade with realistic costs.
        
        Args:
            ticker: Asset ticker
            price: Execution price
            shares: Number of shares
            side: 'buy' or 'sell'
            date: Trade date
        
        Returns:
            True if trade executed successfully
        """
        if shares <= 0:
            return False
        
        # Apply slippage
        if side == 'buy':
            execution_price = price * (1 + self.slippage)
        else:
            execution_price = price * (1 - self.slippage)
        
        # Calculate total cost including transaction fees
        gross_amount = shares * execution_price
        transaction_fee = gross_amount * self.transaction_cost
        net_amount = gross_amount + transaction_fee
        
        if side == 'buy':
            if net_amount > self.cash:
                logger.warning(f"Insufficient cash for trade: ${net_amount:,.2f} > ${self.cash:,.2f}")
                return False
            
            self.cash -= net_amount
            
            # Set stop-loss and take-profit levels
            stop_loss = execution_price * (1 - self.stop_loss_pct) if self.stop_loss_pct else None
            take_profit = execution_price * (1 + self.take_profit_pct) if self.take_profit_pct else None
            
            self.positions[ticker] = Position(
                shares=shares,
                entry_price=execution_price,
                entry_date=date,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            logger.debug(f"BUY: {shares} shares of {ticker} at ${execution_price:.2f}")
            
        else:  # sell
            if ticker not in self.positions:
                logger.warning(f"Cannot sell {ticker}: no position held")
                return False
            
            position = self.positions[ticker]
            sell_shares = min(shares, position.shares)
            
            proceeds = sell_shares * execution_price - transaction_fee
            self.cash += proceeds
            
            # Calculate P&L for the trade
            pnl = (execution_price - position.entry_price) * sell_shares - transaction_fee
            pnl_pct = pnl / (position.entry_price * sell_shares)
            duration = (date - position.entry_date).days
            
            # Record the trade
            trade = Trade(
                entry_date=position.entry_date,
                exit_date=date,
                entry_price=position.entry_price,
                exit_price=execution_price,
                shares=sell_shares,
                pnl=pnl,
                pnl_pct=pnl_pct,
                duration_days=duration,
                side='long'
            )
            self.trades.append(trade)
            
            # Update or close position
            if sell_shares >= position.shares:
                del self.positions[ticker]
            else:
                position.shares -= sell_shares
            
            logger.debug(f"SELL: {sell_shares} shares of {ticker} at ${execution_price:.2f}, P&L: ${pnl:.2f}")
        
        return True
    
    def check_stop_orders(self, ticker: str, current_price: float, date: pd.Timestamp) -> bool:
        """Check and execute stop-loss/take-profit orders."""
        if ticker not in self.positions:
            return False
        
        position = self.positions[ticker]
        
        # Check stop-loss
        if position.stop_loss and current_price <= position.stop_loss:
            logger.info(f"Stop-loss triggered for {ticker} at ${current_price:.2f}")
            return self.execute_trade(ticker, current_price, position.shares, 'sell', date)
        
        # Check take-profit
        if position.take_profit and current_price >= position.take_profit:
            logger.info(f"Take-profit triggered for {ticker} at ${current_price:.2f}")
            return self.execute_trade(ticker, current_price, position.shares, 'sell', date)
        
        return False
    
    def calculate_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Calculate current portfolio value."""
        portfolio_value = self.cash
        
        for ticker, position in self.positions.items():
            if ticker in prices:
                portfolio_value += position.shares * prices[ticker]
        
        return portfolio_value
    
    def run_backtest(self, 
                    data: pd.DataFrame, 
                    signals: pd.Series,
                    ticker: str = "ASSET",
                    volatility_window: int = 20) -> Dict:
        """
        Run the backtest with advanced features.
        
        Args:
            data: OHLCV data
            signals: Trading signals (1=buy, -1=sell, 0=hold)
            ticker: Asset ticker symbol
            volatility_window: Window for volatility calculation
        
        Returns:
            Comprehensive backtest results
        """
        logger.info(f"Starting backtest for {ticker} with {len(data)} data points")
        
        # Calculate rolling volatility
        returns = data['Close'].pct_change()
        volatility = returns.rolling(volatility_window).std() * np.sqrt(252)
        
        # Align data
        aligned_data = pd.DataFrame({
            'Price': data['Close'],
            'Signal': signals,
            'Volatility': volatility
        }).dropna()
        
        equity_curve = []
        
        for date, row in aligned_data.iterrows():
            price = row['Price']
            signal = row['Signal']
            vol = row['Volatility']
            
            # Check stop orders first
            self.check_stop_orders(ticker, price, date)
            
            # Execute new signals
            if signal == 1 and ticker not in self.positions:  # Buy signal
                shares = self.calculate_position_size(price, 1.0, vol)
                self.execute_trade(ticker, price, shares, 'buy', date)
                
            elif signal == -1 and ticker in self.positions:  # Sell signal
                position = self.positions[ticker]
                self.execute_trade(ticker, price, position.shares, 'sell', date)
            
            # Calculate portfolio value
            portfolio_value = self.calculate_portfolio_value({ticker: price})
            equity_curve.append(portfolio_value)
        
        # Store equity curve
        self.equity_curve = pd.Series(equity_curve, index=aligned_data.index)
        
        # Calculate performance metrics
        metrics = self._calculate_comprehensive_metrics(aligned_data['Price'])
        
        results = {
            'equity_curve': self.equity_curve,
            'trades': self.trades,
            'metrics': metrics,
            'final_value': self.equity_curve.iloc[-1] if len(self.equity_curve) > 0 else self.initial_capital,
            'positions': self.positions.copy()
        }
        
        logger.info(f"Backtest completed. Final value: ${results['final_value']:,.2f}")
        return results
    
    def _calculate_comprehensive_metrics(self, price_series: pd.Series) -> Dict:
        """Calculate comprehensive performance metrics."""
        if len(self.equity_curve) == 0:
            return {}
        
        # Basic returns
        portfolio_returns = self.equity_curve.pct_change().dropna()
        benchmark_returns = price_series.pct_change().dropna()
        
        # Align series
        min_len = min(len(portfolio_returns), len(benchmark_returns))
        portfolio_returns = portfolio_returns.iloc[:min_len]
        benchmark_returns = benchmark_returns.iloc[:min_len]
        
        # Performance metrics
        total_return = (self.equity_curve.iloc[-1] / self.initial_capital) - 1
        benchmark_total_return = (price_series.iloc[-1] / price_series.iloc[0]) - 1
        
        # Risk metrics
        annual_return = (1 + total_return) ** (252 / len(self.equity_curve)) - 1
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Downside metrics
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        sortino = annual_return / downside_vol if downside_vol > 0 else 0
        
        # Drawdown analysis
        cumulative = self.equity_curve / self.initial_capital
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calculate recovery periods
        drawdown_periods = []
        in_drawdown = False
        start_dd = None
        
        for i, dd in enumerate(drawdown):
            if dd < -0.01 and not in_drawdown:  # Start of drawdown (>1%)
                in_drawdown = True
                start_dd = i
            elif dd >= -0.001 and in_drawdown:  # Recovery
                in_drawdown = False
                if start_dd is not None:
                    drawdown_periods.append(i - start_dd)
        
        avg_recovery_days = np.mean(drawdown_periods) if drawdown_periods else 0
        
        # Trading metrics
        if self.trades:
            winning_trades = [t for t in self.trades if t.pnl > 0]
            losing_trades = [t for t in self.trades if t.pnl <= 0]
            
            win_rate = len(winning_trades) / len(self.trades)
            avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            
            avg_trade_duration = np.mean([t.duration_days for t in self.trades])
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            avg_trade_duration = 0
        
        # Advanced metrics
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Information ratio (excess return vs benchmark)
        excess_returns = portfolio_returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = excess_returns.mean() * np.sqrt(252) / tracking_error if tracking_error > 0 else 0
        
        return {
            'total_return': total_return,
            'benchmark_return': benchmark_total_return,
            'excess_return': total_return - benchmark_total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio,
            'max_drawdown': max_drawdown,
            'avg_recovery_days': avg_recovery_days,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade_duration': avg_trade_duration,
            'total_trades': len(self.trades),
            'final_value': self.equity_curve.iloc[-1],
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
    
    def plot_results(self) -> go.Figure:
        """Create comprehensive performance visualization."""
        if len(self.equity_curve) == 0:
            return go.Figure()
        
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=[
                'Portfolio Value vs Benchmark',
                'Drawdown',
                'Rolling Sharpe Ratio (252-day)',
                'Trade P&L Distribution'
            ],
            vertical_spacing=0.08,
            row_heights=[0.4, 0.2, 0.2, 0.2]
        )
        
        # Portfolio value
        fig.add_trace(
            go.Scatter(
                x=self.equity_curve.index,
                y=self.equity_curve,
                name='Portfolio',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Drawdown
        cumulative = self.equity_curve / self.initial_capital
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown * 100,
                fill='tonexty',
                name='Drawdown %',
                line=dict(color='red')
            ),
            row=2, col=1
        )
        
        # Rolling Sharpe
        if len(self.equity_curve) > 252:
            returns = self.equity_curve.pct_change()
            rolling_sharpe = returns.rolling(252).mean() / returns.rolling(252).std() * np.sqrt(252)
            
            fig.add_trace(
                go.Scatter(
                    x=rolling_sharpe.index,
                    y=rolling_sharpe,
                    name='Rolling Sharpe',
                    line=dict(color='green')
                ),
                row=3, col=1
            )
        
        # Trade P&L
        if self.trades:
            trade_pnl = [t.pnl for t in self.trades]
            fig.add_trace(
                go.Histogram(
                    x=trade_pnl,
                    name='Trade P&L',
                    nbinsx=20,
                    marker_color='purple'
                ),
                row=4, col=1
            )
        
        fig.update_layout(
            title='Advanced Backtest Results',
            height=1000,
            showlegend=True
        )
        
        return fig