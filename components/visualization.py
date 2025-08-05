"""
Visualization component for QuantLab Professional.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
from utils.logger import get_logger

logger = get_logger(__name__)


class Visualizer:
    """Advanced visualization component for financial data and results."""
    
    def __init__(self):
        self.color_palette = {
            'primary': '#2a5298',
            'secondary': '#1e3c72',
            'success': '#28a745',
            'danger': '#dc3545',
            'warning': '#ffc107',
            'info': '#17a2b8'
        }
        logger.info("Visualizer initialized")
    
    def create_price_chart(self, 
                          data: pd.DataFrame, 
                          ticker: str,
                          include_volume: bool = True) -> go.Figure:
        """
        Create an interactive price chart with volume.
        
        Args:
            data: OHLCV data
            ticker: Stock ticker symbol
            include_volume: Whether to include volume subplot
        
        Returns:
            Plotly figure
        """
        if include_volume and 'Volume' in data.columns:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=[f'{ticker} Price', 'Volume'],
                row_width=[0.2, 0.7]
            )
        else:
            fig = go.Figure()
        
        # Candlestick chart
        candlestick = go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=ticker,
            increasing_line_color=self.color_palette['success'],
            decreasing_line_color=self.color_palette['danger']
        )
        
        if include_volume and 'Volume' in data.columns:
            fig.add_trace(candlestick, row=1, col=1)
            
            # Volume bars
            colors = ['red' if close < open else 'green' 
                     for close, open in zip(data['Close'], data['Open'])]
            
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    marker_color=colors,
                    name='Volume',
                    opacity=0.7
                ),
                row=2, col=1
            )
        else:
            fig.add_trace(candlestick)
        
        # Update layout
        fig.update_layout(
            title=f'{ticker} Price Chart',
            yaxis_title='Price ($)',
            xaxis_rangeslider_visible=False,
            height=600 if include_volume else 400,
            showlegend=True,
            template='plotly_white'
        )
        
        if include_volume:
            fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
    
    def create_technical_indicators_chart(self, 
                                        data: pd.DataFrame, 
                                        indicators: List[str]) -> go.Figure:
        """
        Create technical indicators chart.
        
        Args:
            data: Data with technical indicators
            indicators: List of indicators to plot
        
        Returns:
            Plotly figure with indicators
        """
        fig = make_subplots(
            rows=len(indicators), cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=indicators
        )
        
        for i, indicator in enumerate(indicators, 1):
            if indicator == "SMA":
                # Simple Moving Averages
                if 'SMA_10' in data.columns:
                    fig.add_trace(
                        go.Scatter(x=data.index, y=data['SMA_10'], name='SMA 10'),
                        row=i, col=1
                    )
                if 'SMA_20' in data.columns:
                    fig.add_trace(
                        go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20'),
                        row=i, col=1
                    )
                if 'SMA_50' in data.columns:
                    fig.add_trace(
                        go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50'),
                        row=i, col=1
                    )
                    
            elif indicator == "EMA":
                # Exponential Moving Averages
                if 'EMA_12' in data.columns:
                    fig.add_trace(
                        go.Scatter(x=data.index, y=data['EMA_12'], name='EMA 12'),
                        row=i, col=1
                    )
                if 'EMA_26' in data.columns:
                    fig.add_trace(
                        go.Scatter(x=data.index, y=data['EMA_26'], name='EMA 26'),
                        row=i, col=1
                    )
                    
            elif indicator == "RSI":
                if 'RSI' in data.columns:
                    fig.add_trace(
                        go.Scatter(x=data.index, y=data['RSI'], name='RSI'),
                        row=i, col=1
                    )
                    # Add overbought/oversold lines
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=i, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=i, col=1)
                    
            elif indicator == "MACD":
                if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                    fig.add_trace(
                        go.Scatter(x=data.index, y=data['MACD'], name='MACD'),
                        row=i, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal'),
                        row=i, col=1
                    )
                    if 'MACD_Histogram' in data.columns:
                        fig.add_trace(
                            go.Bar(x=data.index, y=data['MACD_Histogram'], name='Histogram'),
                            row=i, col=1
                        )
                        
            elif indicator == "Bollinger Bands":
                if all(col in data.columns for col in ['BB_Upper', 'BB_Lower', 'Close']):
                    fig.add_trace(
                        go.Scatter(x=data.index, y=data['Close'], name='Price'),
                        row=i, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=data.index, y=data['BB_Upper'], name='Upper Band'),
                        row=i, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=data.index, y=data['BB_Lower'], name='Lower Band'),
                        row=i, col=1
                    )
        
        fig.update_layout(
            height=300 * len(indicators),
            title="Technical Indicators",
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def create_prediction_plot(self, ml_result: Dict) -> go.Figure:
        """
        Create ML prediction vs actual plot.
        
        Args:
            ml_result: ML model results
        
        Returns:
            Prediction comparison plot
        """
        test_data = ml_result['test_data']
        predictions = ml_result['predictions']
        actual = test_data['y_test']
        
        fig = go.Figure()
        
        # Actual vs Predicted scatter
        fig.add_trace(go.Scatter(
            x=actual,
            y=predictions,
            mode='markers',
            name='Predictions',
            marker=dict(
                color=self.color_palette['primary'],
                opacity=0.6
            )
        ))
        
        # Perfect prediction line
        min_val = min(actual.min(), predictions.min())
        max_val = max(actual.max(), predictions.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='ML Predictions vs Actual Returns',
            xaxis_title='Actual Returns',
            yaxis_title='Predicted Returns',
            template='plotly_white'
        )
        
        return fig
    
    def create_backtest_chart(self, results: Dict) -> go.Figure:
        """
        Create comprehensive backtest results chart.
        
        Args:
            results: Backtest results
        
        Returns:
            Backtest visualization
        """
        equity_curve = results['equity_curve']
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=['Portfolio Value', 'Drawdown', 'Returns Distribution'],
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Portfolio value
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve,
                name='Portfolio Value',
                line=dict(color=self.color_palette['primary'])
            ),
            row=1, col=1
        )
        
        # Drawdown
        cumulative = equity_curve / equity_curve.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown * 100,
                fill='tonexty',
                name='Drawdown %',
                line=dict(color=self.color_palette['danger'])
            ),
            row=2, col=1
        )
        
        # Returns distribution
        returns = equity_curve.pct_change().dropna()
        fig.add_trace(
            go.Histogram(
                x=returns * 100,
                name='Daily Returns %',
                nbinsx=30,
                marker_color=self.color_palette['info']
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            title='Backtest Results',
            height=800,
            template='plotly_white'
        )
        
        return fig
    
    def create_portfolio_composition_chart(self, 
                                         result: Any, 
                                         strategy_name: str) -> go.Figure:
        """
        Create portfolio composition pie chart.
        
        Args:
            result: Portfolio optimization result
            strategy_name: Name of the strategy
        
        Returns:
            Portfolio composition chart
        """
        # Extract weights and create labels
        weights = result.weights if hasattr(result, 'weights') else result.get('weights', [])
        
        # Create dummy asset names if not provided
        labels = [f'Asset {i+1}' for i in range(len(weights))]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=weights,
            hole=0.3,
            textinfo='label+percent',
            textposition='outside',
            marker=dict(
                colors=px.colors.qualitative.Set3[:len(weights)]
            )
        )])
        
        fig.update_layout(
            title=f'{strategy_name} Portfolio Composition',
            height=400,
            template='plotly_white',
            annotations=[dict(
                text=f'Sharpe: {getattr(result, "sharpe_ratio", 0):.3f}',
                x=0.5, y=0.5,
                font_size=16,
                showarrow=False
            )]
        )
        
        return fig
    
    def create_efficient_frontier_chart(self, results: Dict) -> go.Figure:
        """
        Create efficient frontier visualization.
        
        Args:
            results: Portfolio optimization results
        
        Returns:
            Efficient frontier chart
        """
        fig = go.Figure()
        
        # Add efficient frontier if available
        if 'efficient_frontier' in results:
            frontier_data = results['efficient_frontier']
            fig.add_trace(go.Scatter(
                x=frontier_data['Volatility'],
                y=frontier_data['Return'],
                mode='lines+markers',
                name='Efficient Frontier',
                line=dict(color=self.color_palette['primary'])
            ))
        
        # Add optimized portfolios
        for strategy_name, result in results.items():
            if strategy_name != 'efficient_frontier' and hasattr(result, 'volatility'):
                fig.add_trace(go.Scatter(
                    x=[result.volatility],
                    y=[result.expected_return],
                    mode='markers',
                    marker=dict(size=12, symbol='star'),
                    name=strategy_name
                ))
        
        fig.update_layout(
            title='Efficient Frontier & Optimized Portfolios',
            xaxis_title='Volatility (Annual)',
            yaxis_title='Expected Return (Annual)',
            template='plotly_white'
        )
        
        return fig
    
    def create_correlation_heatmap(self, data: pd.DataFrame) -> go.Figure:
        """
        Create correlation heatmap for assets.
        
        Args:
            data: Returns data for multiple assets
        
        Returns:
            Correlation heatmap
        """
        corr_matrix = data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.around(corr_matrix.values, decimals=2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Asset Correlation Matrix',
            template='plotly_white'
        )
        
        return fig