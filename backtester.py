import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class SimpleBacktester:
    """Simple backtesting engine for trading strategies."""
    
    def __init__(self, initial_capital=10000, transaction_cost=0.001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.results = None
    
    def run_backtest(self, data: pd.DataFrame, signals: pd.Series, 
                    prices: pd.Series, strategy_name: str = "Strategy") -> Dict:
        """
        Run backtest on given signals and prices.
        
        Args:
            data: DataFrame with price data
            signals: Series with trading signals (1 for buy, 0 for hold, -1 for sell)
            prices: Series with prices for the same dates
            strategy_name: Name of the strategy
        
        Returns:
            Dictionary with backtest results
        """
        # Ensure signals and prices are aligned
        aligned_data = pd.DataFrame({
            'Price': prices,
            'Signal': signals
        }).dropna()
        
        if len(aligned_data) == 0:
            raise ValueError("No valid data points after alignment")
        
        # Initialize tracking variables
        cash = self.initial_capital
        position = 0
        portfolio_value = []
        trades = []
        positions = []
        
        for i, (date, row) in enumerate(aligned_data.iterrows()):
            price = row['Price']
            signal = row['Signal']
            
            # Current portfolio value
            current_value = cash + position * price
            portfolio_value.append(current_value)
            positions.append(position)
            
            # Execute trades based on signals
            if signal == 1 and position == 0:  # Buy signal and not already long
                shares_to_buy = int((cash * (1 - self.transaction_cost)) // price)
                if shares_to_buy > 0:
                    cost = shares_to_buy * price * (1 + self.transaction_cost)
                    cash -= cost
                    position = shares_to_buy
                    trades.append({
                        'date': date,
                        'action': 'BUY',
                        'shares': shares_to_buy,
                        'price': price,
                        'cost': cost
                    })
            
            elif signal == -1 and position > 0:  # Sell signal and currently long
                proceeds = position * price * (1 - self.transaction_cost)
                cash += proceeds
                trades.append({
                    'date': date,
                    'action': 'SELL',
                    'shares': position,
                    'price': price,
                    'proceeds': proceeds
                })
                position = 0
        
        # Create results DataFrame
        results_df = aligned_data.copy()
        results_df['Portfolio_Value'] = portfolio_value
        results_df['Position'] = positions
        results_df['Cash'] = cash
        
        # Calculate returns
        results_df['Strategy_Return'] = results_df['Portfolio_Value'].pct_change().fillna(0)
        results_df['Market_Return'] = results_df['Price'].pct_change().fillna(0)
        
        # Calculate cumulative returns
        results_df['Strategy_Cumulative'] = (1 + results_df['Strategy_Return']).cumprod()
        results_df['Market_Cumulative'] = (1 + results_df['Market_Return']).cumprod()
        
        # Performance metrics
        metrics = self._calculate_metrics(results_df, trades)
        
        self.results = {
            'data': results_df,
            'trades': trades,
            'metrics': metrics,
            'strategy_name': strategy_name
        }
        
        return self.results
    
    def _calculate_metrics(self, results_df: pd.DataFrame, trades: List[Dict]) -> Dict:
        """Calculate performance metrics."""
        strategy_returns = results_df['Strategy_Return'].dropna()
        market_returns = results_df['Market_Return'].dropna()
        
        if len(strategy_returns) == 0:
            return {}
        
        # Basic metrics
        total_return = results_df['Strategy_Cumulative'].iloc[-1] - 1
        market_total_return = results_df['Market_Cumulative'].iloc[-1] - 1
        
        # Annualized metrics (assuming daily data)
        trading_days = len(strategy_returns)
        years = trading_days / 252
        
        ann_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        ann_vol = strategy_returns.std() * np.sqrt(252)
        market_ann_return = (1 + market_total_return) ** (1/years) - 1 if years > 0 else 0
        
        # Risk metrics
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        # Drawdown
        cumulative = results_df['Strategy_Cumulative']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trading metrics
        num_trades = len(trades)
        profitable_trades = len([t for t in trades if t.get('proceeds', 0) > t.get('cost', 0)])
        win_rate = profitable_trades / num_trades if num_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'market_total_return': market_total_return,
            'excess_return': total_return - market_total_return,
            'ann_return': ann_return,
            'market_ann_return': market_ann_return,
            'ann_vol': ann_vol,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'final_value': results_df['Portfolio_Value'].iloc[-1]
        }
    
    def plot_results(self, show_trades=True):
        """Plot backtest results."""
        if self.results is None:
            raise ValueError("No backtest results available. Run backtest first.")
        
        data = self.results['data']
        trades = self.results['trades']
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=['Portfolio Value vs Market', 'Price and Positions', 'Drawdown'],
            vertical_spacing=0.08,
            row_heights=[0.5, 0.3, 0.2]
        )
        
        # Portfolio value comparison
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Strategy_Cumulative'] * self.initial_capital,
                      name='Strategy', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Market_Cumulative'] * self.initial_capital,
                      name='Buy & Hold', line=dict(color='orange')),
            row=1, col=1
        )
        
        # Price and positions
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Price'], name='Price', line=dict(color='black')),
            row=2, col=1
        )
        
        # Add trade markers if requested
        if show_trades and trades:
            buy_dates = [t['date'] for t in trades if t['action'] == 'BUY']
            buy_prices = [t['price'] for t in trades if t['action'] == 'BUY']
            sell_dates = [t['date'] for t in trades if t['action'] == 'SELL']
            sell_prices = [t['price'] for t in trades if t['action'] == 'SELL']
            
            if buy_dates:
                fig.add_trace(
                    go.Scatter(x=buy_dates, y=buy_prices, mode='markers',
                              marker=dict(symbol='triangle-up', size=10, color='green'),
                              name='Buy'),
                    row=2, col=1
                )
            
            if sell_dates:
                fig.add_trace(
                    go.Scatter(x=sell_dates, y=sell_prices, mode='markers',
                              marker=dict(symbol='triangle-down', size=10, color='red'),
                              name='Sell'),
                    row=2, col=1
                )
        
        # Drawdown
        cumulative = data['Strategy_Cumulative']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        fig.add_trace(
            go.Scatter(x=data.index, y=drawdown, fill='tonexty',
                      name='Drawdown', line=dict(color='red')),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f"Backtest Results: {self.results['strategy_name']}",
            height=800,
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Price", row=2, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        return fig
    
    def get_trade_summary(self):
        """Get summary of all trades."""
        if self.results is None or not self.results['trades']:
            return pd.DataFrame()
        
        trades_df = pd.DataFrame(self.results['trades'])
        return trades_df


class StrategySignalGenerator:
    """Generate trading signals based on different strategies."""
    
    @staticmethod
    def prediction_based_signals(predictions, prices, threshold=0.0):
        """Generate signals based on ML predictions."""
        signals = pd.Series(0, index=predictions.index)
        
        # Buy when prediction is significantly higher than current price
        buy_condition = (predictions > prices * (1 + threshold))
        signals[buy_condition] = 1
        
        # Sell when prediction is significantly lower than current price
        sell_condition = (predictions < prices * (1 - threshold))
        signals[sell_condition] = -1
        
        return signals
    
    @staticmethod
    def momentum_signals(data, short_window=10, long_window=20):
        """Generate momentum-based signals using moving averages."""
        data = data.copy()
        data['MA_Short'] = data['Close'].rolling(window=short_window).mean()
        data['MA_Long'] = data['Close'].rolling(window=long_window).mean()
        
        signals = pd.Series(0, index=data.index)
        signals[data['MA_Short'] > data['MA_Long']] = 1
        signals[data['MA_Short'] < data['MA_Long']] = -1
        
        return signals
    
    @staticmethod
    def mean_reversion_signals(data, window=20, num_std=2):
        """Generate mean reversion signals using Bollinger Bands."""
        data = data.copy()
        data['BB_Middle'] = data['Close'].rolling(window=window).mean()
        data['BB_Std'] = data['Close'].rolling(window=window).std()
        data['BB_Upper'] = data['BB_Middle'] + (data['BB_Std'] * num_std)
        data['BB_Lower'] = data['BB_Middle'] - (data['BB_Std'] * num_std)
        
        signals = pd.Series(0, index=data.index)
        signals[data['Close'] < data['BB_Lower']] = 1  # Buy when oversold
        signals[data['Close'] > data['BB_Upper']] = -1  # Sell when overbought
        
        return signals
    
    @staticmethod
    def rsi_signals(data, rsi_period=14, oversold=30, overbought=70):
        """Generate signals based on RSI."""
        data = data.copy()
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        signals = pd.Series(0, index=data.index)
        signals[data['RSI'] < oversold] = 1  # Buy when oversold
        signals[data['RSI'] > overbought] = -1  # Sell when overbought
        
        return signals