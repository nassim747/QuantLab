"""
QuantLab Professional - Advanced Trading Simulator
A professional-grade quantitative finance platform.
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import configurations and utilities
from config import config
from utils.logger import get_logger

# Import components
from components.data_handler import DataHandler
from components.ml_pipeline import MLPipeline
from components.strategy_manager import StrategyManager
from components.portfolio_manager import PortfolioManager
from components.visualization import Visualizer

# Initialize logger
logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title=config.ui.page_title,
    page_icon=config.ui.page_icon,
    layout=config.ui.layout,
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown(f"""
<style>
    .main-header {{
        background: linear-gradient(90deg, #1e3c72 0%, {config.ui.theme_primary_color} 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    .metric-card {{
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid {config.ui.theme_primary_color};
        margin-bottom: 1rem;
    }}
    .success-box {{
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }}
    .warning-box {{
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 0 24px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {config.ui.theme_primary_color};
        color: white;
    }}
</style>
""", unsafe_allow_html=True)


class QuantLabApp:
    """Main application class for QuantLab Professional."""
    
    def __init__(self):
        """Initialize the application."""
        self.data_handler = DataHandler()
        self.ml_pipeline = MLPipeline()
        self.strategy_manager = StrategyManager()
        self.portfolio_manager = PortfolioManager()
        self.visualizer = Visualizer()
        
        # Initialize session state
        self._init_session_state()
        
        logger.info("QuantLab Professional initialized")
    
    def _init_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'ml_results' not in st.session_state:
            st.session_state.ml_results = {}
        if 'backtest_results' not in st.session_state:
            st.session_state.backtest_results = {}
        if 'portfolio_results' not in st.session_state:
            st.session_state.portfolio_results = {}
        if 'current_ticker' not in st.session_state:
            st.session_state.current_ticker = config.data.default_ticker
    
    def render_header(self):
        """Render the application header."""
        st.markdown("""
        <div class="main-header">
            <h1>📈 QuantLab Professional</h1>
            <p>Advanced Quantitative Finance Platform with ML-Driven Trading Strategies</p>
            <div style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.9;">
                Professional backtesting • Portfolio optimization • Risk management
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar configuration."""
        with st.sidebar:
            st.markdown("## ⚙️ Configuration")
            
            # Data Configuration
            with st.expander("📊 Data Settings", expanded=True):
                ticker = st.text_input(
                    "Ticker Symbol", 
                    value=st.session_state.current_ticker,
                    help="Enter stock ticker (e.g., AAPL, MSFT, GOOGL)"
                )
                
                period_years = st.selectbox(
                    "Data Period",
                    options=[1, 2, 3, 5, 10],
                    index=3,
                    format_func=lambda x: f"{x} Year{'s' if x > 1 else ''}"
                )
                
                # Update session state if ticker changed
                if ticker != st.session_state.current_ticker:
                    st.session_state.current_ticker = ticker
                    st.session_state.data = None  # Reset data
            
            # ML Configuration
            with st.expander("🤖 ML Settings", expanded=False):
                forecast_horizon = st.slider(
                    "Forecast Horizon (days)", 
                    1, 30, 
                    int(config.ml.default_forecast_horizon)
                )
                
                use_log_returns = st.checkbox(
                    "Use Log Returns", 
                    value=True,
                    help="Log returns are stationary and better for ML models"
                )
                
                model_type = st.selectbox(
                    "Model Type",
                    ["Linear Regression", "Random Forest Regressor", "XGBoost Regressor"]
                )
                
                enable_advanced = st.checkbox(
                    "Enable Advanced Models",
                    help="LSTM and Prophet models (requires additional packages)"
                )
            
            # Strategy Configuration
            with st.expander("📈 Strategy Settings", expanded=False):
                strategy_type = st.selectbox(
                    "Strategy Type",
                    [
                        "ML Prediction Based",
                        "Moving Average Crossover",
                        "Mean Reversion",
                        "RSI Strategy",
                        "Portfolio Optimization"
                    ]
                )
                
                position_sizing = st.selectbox(
                    "Position Sizing",
                    ["Volatility Adjusted", "Fixed Percentage", "Kelly Criterion", "Risk Parity"]
                )
            
            # Risk Management
            with st.expander("⚠️ Risk Management", expanded=False):
                initial_capital = st.number_input(
                    "Initial Capital ($)", 
                    value=float(config.backtest.initial_capital),
                    min_value=1000.0,
                    step=1000.0
                )
                
                max_position_pct = st.slider(
                    "Max Position Size (%)",
                    5, 50,
                    int(config.backtest.max_position_pct * 100)
                ) / 100
                
                stop_loss_pct = st.slider(
                    "Stop Loss (%)",
                    0, 20,
                    5
                ) / 100
                
                take_profit_pct = st.slider(
                    "Take Profit (%)",
                    0, 50,
                    10
                ) / 100
            
            return {
                'ticker': ticker,
                'period_years': period_years,
                'forecast_horizon': forecast_horizon,
                'use_log_returns': use_log_returns,
                'model_type': model_type,
                'enable_advanced': enable_advanced,
                'strategy_type': strategy_type,
                'position_sizing': position_sizing,
                'initial_capital': initial_capital,
                'max_position_pct': max_position_pct,
                'stop_loss_pct': stop_loss_pct,
                'take_profit_pct': take_profit_pct
            }
    
    def load_and_process_data(self, ticker: str, period_years: int):
        """Load and process market data."""
        try:
            with st.spinner(f"Loading {ticker} data..."):
                data = self.data_handler.load_data(ticker, period_years * 365)
                
                if data is None or data.empty:
                    st.error(f"❌ No data found for {ticker}")
                    return None
                
                # Process data
                processed_data = self.data_handler.process_data(data)
                st.session_state.data = processed_data
                
                # Display data summary
                metrics = self.data_handler.calculate_summary_metrics(processed_data)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Latest Close", f"${metrics['latest_close']:.2f}")
                with col2:
                    st.metric("📈 Daily Change", f"${metrics['change']:.2f}", f"{metrics['change_pct']:.2f}%")
                with col3:
                    st.metric("📅 Data Points", f"{metrics['total_days']:,}")
                with col4:
                    st.metric("⏱️ Period", f"{metrics['start_date']} to {metrics['end_date']}")
                
                return processed_data
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            st.error(f"❌ Error loading data: {e}")
            return None
    
    def run(self):
        """Run the main application."""
        try:
            # Render UI components
            self.render_header()
            config_params = self.render_sidebar()
            
            # Load data
            data = self.load_and_process_data(
                config_params['ticker'], 
                config_params['period_years']
            )
            
            if data is None:
                st.info("👆 Please configure data settings in the sidebar and ensure the ticker is valid.")
                return
            
            # Create main tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Data Analysis",
                "🤖 ML Models", 
                "📈 Strategy Testing",
                "💼 Portfolio Optimization",
                "📋 Performance Dashboard"
            ])
            
            with tab1:
                self._render_data_analysis_tab(data, config_params)
            
            with tab2:
                self._render_ml_models_tab(data, config_params)
            
            with tab3:
                self._render_strategy_testing_tab(data, config_params)
            
            with tab4:
                self._render_portfolio_optimization_tab(data, config_params)
            
            with tab5:
                self._render_performance_dashboard_tab(config_params)
                
        except Exception as e:
            logger.error(f"Application error: {e}")
            st.error(f"❌ Application error: {e}")
    
    def _render_data_analysis_tab(self, data: pd.DataFrame, params: dict):
        """Render data analysis tab."""
        st.header("📊 Market Data Analysis")
        
        # Price chart
        fig_price = self.visualizer.create_price_chart(
            data, params['ticker'], include_volume=True
        )
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Technical indicators
        with st.expander("🔍 Technical Analysis", expanded=False):
            indicators = st.multiselect(
                "Select Indicators",
                ["SMA", "EMA", "MACD", "RSI", "Bollinger Bands"],
                default=["SMA", "RSI"]
            )
            
            if indicators:
                fig_tech = self.visualizer.create_technical_indicators_chart(data, indicators)
                st.plotly_chart(fig_tech, use_container_width=True)
        
        # Statistical summary
        with st.expander("📈 Statistical Summary", expanded=False):
            summary_stats = self.data_handler.calculate_statistical_summary(data)
            st.dataframe(summary_stats, use_container_width=True)
    
    def _render_ml_models_tab(self, data: pd.DataFrame, params: dict):
        """Render ML models tab."""
        st.header("🤖 Machine Learning Models")
        
        if st.button("🚀 Train Models", type="primary"):
            with st.spinner("Training ML models..."):
                results = self.ml_pipeline.train_models(
                    data, 
                    forecast_horizon=params['forecast_horizon'],
                    use_log_returns=params['use_log_returns'],
                    model_types=[params['model_type']],
                    enable_advanced=params['enable_advanced']
                )
                st.session_state.ml_results = results
        
        # Display results
        if st.session_state.ml_results:
            self._display_ml_results(st.session_state.ml_results)
    
    def _render_strategy_testing_tab(self, data: pd.DataFrame, params: dict):
        """Render strategy testing tab."""
        st.header("📈 Strategy Testing & Backtesting")
        
        if st.button("🔄 Run Backtest", type="primary"):
            with st.spinner("Running advanced backtest..."):
                results = self.strategy_manager.run_backtest(
                    data,
                    strategy_type=params['strategy_type'],
                    ml_results=st.session_state.ml_results,
                    config=params
                )
                st.session_state.backtest_results = results
        
        # Display backtest results
        if st.session_state.backtest_results:
            self._display_backtest_results(st.session_state.backtest_results)
    
    def _render_portfolio_optimization_tab(self, data: pd.DataFrame, params: dict):
        """Render portfolio optimization tab."""
        st.header("💼 Portfolio Optimization")
        
        # Multi-ticker selection
        tickers = st.multiselect(
            "Select Assets for Portfolio",
            ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"],
            default=["AAPL", "MSFT", "GOOGL", "AMZN"]
        )
        
        if len(tickers) >= 2:
            if st.button("🎯 Optimize Portfolio", type="primary"):
                with st.spinner("Optimizing portfolio..."):
                    results = self.portfolio_manager.optimize_portfolio(
                        tickers, period_years=params['period_years']
                    )
                    st.session_state.portfolio_results = results
            
            # Display optimization results
            if st.session_state.portfolio_results:
                self._display_portfolio_results(st.session_state.portfolio_results)
        else:
            st.info("Please select at least 2 assets for portfolio optimization.")
    
    def _render_performance_dashboard_tab(self, params: dict):
        """Render performance dashboard."""
        st.header("📋 Performance Dashboard")
        
        # Summary cards
        if any([st.session_state.ml_results, 
                st.session_state.backtest_results, 
                st.session_state.portfolio_results]):
            
            self._create_performance_summary()
            
            # Comparative analysis
            st.subheader("📊 Comparative Analysis")
            self._create_comparative_charts()
        else:
            st.info("Run ML models, backtests, or portfolio optimization to see performance metrics.")
    
    def _display_ml_results(self, results: dict):
        """Display ML model results."""
        for model_name, result in results.items():
            with st.expander(f"📈 {model_name} Results", expanded=True):
                metrics = result.get('metrics', {})
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
                with col2:
                    st.metric("R²", f"{metrics.get('r2', 0):.4f}")
                with col3:
                    st.metric("Hit Rate", f"{metrics.get('hit_rate', 0):.2%}")
                with col4:
                    st.metric("Direction Accuracy", f"{metrics.get('directional_accuracy', 0):.2%}")
                
                # Prediction plot
                if 'predictions' in result:
                    fig = self.visualizer.create_prediction_plot(result)
                    st.plotly_chart(fig, use_container_width=True)
    
    def _display_backtest_results(self, results: dict):
        """Display backtest results."""
        metrics = results.get('metrics', {})
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Return", f"{metrics.get('total_return', 0):.2%}")
        with col2:
            st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
        with col3:
            st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")
        with col4:
            st.metric("Win Rate", f"{metrics.get('win_rate', 0):.2%}")
        
        # Performance chart
        if 'equity_curve' in results:
            fig = self.visualizer.create_backtest_chart(results)
            st.plotly_chart(fig, use_container_width=True)
        
        # Trade analysis
        if 'trades' in results and results['trades']:
            with st.expander("📊 Trade Analysis"):
                trades_df = pd.DataFrame([
                    {
                        'Entry Date': trade.entry_date,
                        'Exit Date': trade.exit_date,
                        'Duration': trade.duration_days,
                        'P&L': trade.pnl,
                        'P&L %': trade.pnl_pct * 100,
                        'Shares': trade.shares
                    }
                    for trade in results['trades']
                ])
                st.dataframe(trades_df, use_container_width=True)
    
    def _display_portfolio_results(self, results: dict):
        """Display portfolio optimization results."""
        for strategy_name, result in results.items():
            with st.expander(f"🎯 {strategy_name}", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Expected Return", f"{result.expected_return:.2%}")
                with col2:
                    st.metric("Volatility", f"{result.volatility:.2%}")
                with col3:
                    st.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
                with col4:
                    st.metric("Max Drawdown", f"{result.max_drawdown:.2%}")
                
                # Portfolio composition
                fig = self.visualizer.create_portfolio_composition_chart(result, strategy_name)
                st.plotly_chart(fig, use_container_width=True)
        
        # Efficient frontier
        if 'efficient_frontier' in results:
            fig = self.visualizer.create_efficient_frontier_chart(results)
            st.plotly_chart(fig, use_container_width=True)
    
    def _create_performance_summary(self):
        """Create performance summary cards."""
        st.subheader("🎯 Performance Summary")
        
        summary_data = []
        
        # ML Results
        if st.session_state.ml_results:
            best_model = max(
                st.session_state.ml_results.items(),
                key=lambda x: x[1].get('metrics', {}).get('r2', 0)
            )
            summary_data.append({
                'Category': 'Best ML Model',
                'Name': best_model[0],
                'Key Metric': f"R² = {best_model[1].get('metrics', {}).get('r2', 0):.3f}"
            })
        
        # Backtest Results
        if st.session_state.backtest_results:
            metrics = st.session_state.backtest_results.get('metrics', {})
            summary_data.append({
                'Category': 'Strategy Performance',
                'Name': 'Backtest Result',
                'Key Metric': f"Sharpe = {metrics.get('sharpe_ratio', 0):.2f}"
            })
        
        # Portfolio Results
        if st.session_state.portfolio_results:
            best_portfolio = max(
                st.session_state.portfolio_results.items(),
                key=lambda x: x[1].sharpe_ratio if hasattr(x[1], 'sharpe_ratio') else 0
            )
            summary_data.append({
                'Category': 'Best Portfolio',
                'Name': best_portfolio[0],
                'Key Metric': f"Sharpe = {best_portfolio[1].sharpe_ratio:.2f}"
            })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    def _create_comparative_charts(self):
        """Create comparative performance charts."""
        # This would create charts comparing different strategies
        # Implementation depends on available results
        st.info("Comparative charts will be displayed here based on available results.")


def main():
    """Main application entry point."""
    try:
        app = QuantLabApp()
        app.run()
    except Exception as e:
        logger.error(f"Critical application error: {e}")
        st.error(f"❌ Critical error: {e}")
        st.error("Please check the logs and restart the application.")


if __name__ == "__main__":
    main()