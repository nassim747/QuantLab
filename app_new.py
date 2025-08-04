import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

# Import our custom modules
from data_utils import get_data_info, prepare_ml_features, calculate_basic_metrics
from ml_models import MLModelTrainer, TimeSeriesValidator, calculate_trading_metrics
from backtester import SimpleBacktester, StrategySignalGenerator
from advanced_models import AdvancedMLPipeline, TENSORFLOW_AVAILABLE, PROPHET_AVAILABLE

# Page config
st.set_page_config(
    page_title="QuantLab - Advanced Trading Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2a5298;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>📈 QuantLab - Advanced Trading Simulator</h1>
    <p>Professional-grade ML-driven trading strategy development and backtesting</p>
    <div style="font-size: 0.9rem; margin-top: 0.5rem;">
        Built by <a href="https://www.linkedin.com/in/nassim-a-265944286/" target="_blank" style="color: #87CEEB;">Nassim Ameur</a>
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'ml_trainer' not in st.session_state:
        st.session_state.ml_trainer = MLModelTrainer()
    if 'advanced_pipeline' not in st.session_state:
        st.session_state.advanced_pipeline = AdvancedMLPipeline()

init_session_state()

# Sidebar configuration
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Data settings
    st.subheader("Data Settings")
    ticker_symbol = st.text_input("Ticker Symbol", value="AAPL", help="Enter stock ticker (e.g., AAPL, MSFT)")
    data_period = st.selectbox("Data Period", 
                              options=[1, 2, 3, 5], 
                              index=3, 
                              format_func=lambda x: f"{x} Year{'s' if x > 1 else ''}")
    
    # ML settings
    st.subheader("ML Settings")
    forecast_horizon = st.slider("Forecast Horizon (days)", 1, 30, 5)
    use_log_returns = st.checkbox("Use Log Returns", value=False)
    
    # Model selection
    model_type = st.selectbox("Model Type", [
        "Linear Regression", 
        "Random Forest Regressor", 
        "XGBoost Regressor"
    ])
    
    # Advanced models toggle
    use_advanced_models = st.checkbox("Use Advanced Models (LSTM, Prophet)", 
                                     value=False,
                                     help="Enable LSTM and Prophet models (requires additional packages)")
    
    # Strategy settings
    st.subheader("Strategy Settings")
    strategy_type = st.selectbox("Strategy Type", [
        "ML Prediction Based",
        "Moving Average Crossover", 
        "Mean Reversion",
        "RSI Strategy"
    ])
    
    # Backtesting settings
    st.subheader("Backtesting")
    initial_capital = st.number_input("Initial Capital ($)", value=10000, min_value=1000)
    transaction_cost = st.slider("Transaction Cost (%)", 0.0, 1.0, 0.1, step=0.01) / 100

# Main content area
if ticker_symbol:
    # Load data
    with st.spinner(f"Loading {ticker_symbol} data..."):
        data, error = get_data_info(ticker_symbol, days=data_period * 365)
    
    if error:
        st.error(error)
        st.stop()
    
    if data is None or data.empty:
        st.error(f"No data found for {ticker_symbol}")
        st.stop()
    
    st.session_state.data = data
    
    # Calculate basic metrics
    metrics = calculate_basic_metrics(data)
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Latest Close", f"${metrics['latest_close']:.2f}")
    with col2:
        st.metric("Change", f"${metrics['change']:.2f}", f"{metrics['change_pct']:.2f}%")
    with col3:
        st.metric("Data Points", metrics['total_days'])
    with col4:
        st.metric("Period", f"{metrics['start_date']} to {metrics['end_date']}")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Analysis", 
        "🤖 ML Models", 
        "📈 Strategy Testing",
        "🔍 Advanced Analysis",
        "📋 Performance Summary"
    ])
    
    with tab1:
        st.header("Data Analysis & Visualization")
        
        # Price chart
        fig_price = px.line(data, y='Close', title=f"{ticker_symbol} Price History")
        fig_price.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Volume chart (if available)
        if 'Volume' in data.columns:
            fig_volume = px.bar(data, y='Volume', title=f"{ticker_symbol} Volume")
            fig_volume.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_volume, use_container_width=True)
        
        # Technical indicators preview
        with st.expander("Technical Indicators Preview"):
            ml_data, feature_cols = prepare_ml_features(data, forecast_horizon, use_log_returns)
            
            if len(ml_data) > 0:
                st.write(f"Generated {len(feature_cols)} features:")
                st.write(feature_cols[:10])  # Show first 10 features
                
                # Show correlation with target
                corr_with_target = ml_data[feature_cols + ['Target']].corr()['Target'].sort_values(key=abs, ascending=False)
                
                fig_corr = px.bar(
                    x=corr_with_target.index[1:11],  # Top 10 features (excluding target itself)
                    y=corr_with_target.values[1:11],
                    title="Top 10 Features by Correlation with Target"
                )
                st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab2:
        st.header("Machine Learning Models")
        
        # Prepare ML data
        ml_data, feature_cols = prepare_ml_features(data, forecast_horizon, use_log_returns)
        
        if len(ml_data) < 50:
            st.warning("Not enough data for reliable ML training. Need at least 50 data points.")
        else:
            # Train/test split
            split_idx = int(len(ml_data) * 0.8)
            X_train = ml_data[feature_cols].iloc[:split_idx]
            X_test = ml_data[feature_cols].iloc[split_idx:]
            y_train = ml_data['Target'].iloc[:split_idx]
            y_test = ml_data['Target'].iloc[split_idx:]
            
            col1, col2 = st.columns([2, 1])
            
            with col2:
                st.subheader("Model Configuration")
                
                # Hyperparameters based on model type
                hyperparams = {}
                if model_type == "Random Forest Regressor":
                    hyperparams['n_estimators'] = st.slider("N Estimators", 50, 500, 100)
                    hyperparams['max_depth'] = st.slider("Max Depth", 5, 50, 10)
                elif model_type == "XGBoost Regressor":
                    hyperparams['n_estimators'] = st.slider("N Estimators", 50, 500, 100)
                    hyperparams['learning_rate'] = st.slider("Learning Rate", 0.01, 0.3, 0.1)
                    hyperparams['max_depth'] = st.slider("Max Depth", 3, 15, 6)
                
                # Training button
                if st.button("Train Model", type="primary"):
                    with st.spinner("Training model..."):
                        # Train the model
                        model = st.session_state.ml_trainer.train_model(
                            X_train, y_train, model_type, **hyperparams
                        )
                        
                        # Evaluate model
                        metrics, y_pred = st.session_state.ml_trainer.evaluate_model(X_test, y_test)
                        
                        # Store results in session state
                        st.session_state.ml_metrics = metrics
                        st.session_state.ml_predictions = y_pred
                        st.session_state.y_test = y_test
                        st.session_state.X_test = X_test
                
                # Advanced models
                if use_advanced_models:
                    st.subheader("Advanced Models")
                    if st.button("Train Advanced Models"):
                        with st.spinner("Training LSTM and Prophet..."):
                            try:
                                results = st.session_state.advanced_pipeline.train_all_models(
                                    data[['Close']], target_col='Close'
                                )
                                st.session_state.advanced_results = results
                                st.success(f"Trained {len(results)} advanced models")
                            except Exception as e:
                                st.error(f"Advanced model training failed: {e}")
            
            with col1:
                st.subheader("Model Performance")
                
                # Display results if available
                if hasattr(st.session_state, 'ml_metrics'):
                    metrics = st.session_state.ml_metrics
                    
                    # Metrics display
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    with metric_col1:
                        st.metric("RMSE", f"{metrics['rmse']:.4f}")
                    with metric_col2:
                        st.metric("MAE", f"{metrics['mae']:.4f}")
                    with metric_col3:
                        st.metric("R²", f"{metrics['r2']:.4f}")
                    with metric_col4:
                        st.metric("Hit Rate", f"{metrics['hit_rate']:.2%}")
                    
                    # Prediction plot
                    if hasattr(st.session_state, 'ml_predictions'):
                        pred_df = pd.DataFrame({
                            'Date': st.session_state.X_test.index,
                            'Actual': st.session_state.y_test.values,
                            'Predicted': st.session_state.ml_predictions
                        })
                        
                        fig_pred = px.line(pred_df, x='Date', y=['Actual', 'Predicted'],
                                         title="Actual vs Predicted Prices")
                        st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Feature importance
                    importance = st.session_state.ml_trainer.get_feature_importance()
                    if importance is not None:
                        with st.expander("Feature Importance"):
                            fig_imp = px.bar(importance.head(10), 
                                           x='importance' if 'importance' in importance.columns else 'coefficient',
                                           y='feature',
                                           orientation='h',
                                           title="Top 10 Most Important Features")
                            st.plotly_chart(fig_imp, use_container_width=True)
                
                # Advanced model results
                if hasattr(st.session_state, 'advanced_results'):
                    st.subheader("Advanced Model Results")
                    for model_name, result in st.session_state.advanced_results.items():
                        with st.expander(f"{model_name} Results"):
                            metrics = result['metrics']
                            if 'error' not in metrics:
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("RMSE", f"{metrics['rmse']:.4f}")
                                with col2:
                                    st.metric("MAE", f"{metrics['mae']:.4f}")
                                with col3:
                                    st.metric("Direction Accuracy", f"{metrics['direction_accuracy']:.2%}")
    
    with tab3:
        st.header("Strategy Testing & Backtesting")
        
        # Check if we have predictions to work with
        has_predictions = hasattr(st.session_state, 'ml_predictions')
        
        if not has_predictions and strategy_type == "ML Prediction Based":
            st.warning("Train a model first to use ML-based strategy")
        else:
            # Generate signals based on strategy type
            backtester = SimpleBacktester(initial_capital, transaction_cost)
            
            if strategy_type == "ML Prediction Based" and has_predictions:
                # Use ML predictions
                test_data = data.loc[st.session_state.X_test.index]
                signals = StrategySignalGenerator.prediction_based_signals(
                    pd.Series(st.session_state.ml_predictions, index=st.session_state.X_test.index),
                    test_data['Close'],
                    threshold=0.01
                )
                prices = test_data['Close']
                
            elif strategy_type == "Moving Average Crossover":
                signals = StrategySignalGenerator.momentum_signals(data, short_window=10, long_window=20)
                prices = data['Close']
                
            elif strategy_type == "Mean Reversion":
                signals = StrategySignalGenerator.mean_reversion_signals(data, window=20, num_std=2)
                prices = data['Close']
                
            elif strategy_type == "RSI Strategy":
                signals = StrategySignalGenerator.rsi_signals(data, rsi_period=14, oversold=30, overbought=70)
                prices = data['Close']
            
            # Run backtest
            if st.button("Run Backtest", type="primary"):
                with st.spinner("Running backtest..."):
                    try:
                        results = backtester.run_backtest(data, signals, prices, strategy_type)
                        st.session_state.backtest_results = results
                        
                        # Display results
                        metrics = results['metrics']
                        
                        # Key metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Return", f"{metrics['total_return']:.2%}")
                        with col2:
                            st.metric("Annual Return", f"{metrics['ann_return']:.2%}")
                        with col3:
                            st.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
                        with col4:
                            st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
                        
                        # Additional metrics
                        col5, col6, col7, col8 = st.columns(4)
                        with col5:
                            st.metric("Win Rate", f"{metrics['win_rate']:.2%}")
                        with col6:
                            st.metric("Total Trades", metrics['num_trades'])
                        with col7:
                            st.metric("Final Value", f"${metrics['final_value']:,.2f}")
                        with col8:
                            st.metric("Excess Return", f"{metrics['excess_return']:.2%}")
                        
                        # Plot results
                        fig = backtester.plot_results()
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Trade summary
                        trade_df = backtester.get_trade_summary()
                        if not trade_df.empty:
                            with st.expander("Trade Summary"):
                                st.dataframe(trade_df)
                    
                    except Exception as e:
                        st.error(f"Backtest failed: {e}")
    
    with tab4:
        st.header("Advanced Analysis")
        
        # Model comparison
        if hasattr(st.session_state, 'ml_metrics'):
            st.subheader("Model Performance Comparison")
            
            # Collect all model results
            all_results = {}
            
            # Traditional ML model
            all_results[model_type] = st.session_state.ml_metrics
            
            # Advanced models
            if hasattr(st.session_state, 'advanced_results'):
                for model_name, result in st.session_state.advanced_results.items():
                    if 'error' not in result['metrics']:
                        all_results[model_name] = result['metrics']
            
            if len(all_results) > 1:
                # Create comparison DataFrame
                comparison_df = pd.DataFrame(all_results).T
                
                # Plot comparison
                metrics_to_compare = ['rmse', 'mae', 'r2']
                available_metrics = [m for m in metrics_to_compare if m in comparison_df.columns]
                
                if available_metrics:
                    fig_comp = make_subplots(
                        rows=1, cols=len(available_metrics),
                        subplot_titles=available_metrics
                    )
                    
                    for i, metric in enumerate(available_metrics):
                        fig_comp.add_trace(
                            go.Bar(x=comparison_df.index, y=comparison_df[metric], name=metric),
                            row=1, col=i+1
                        )
                    
                    fig_comp.update_layout(height=400, title="Model Performance Comparison")
                    st.plotly_chart(fig_comp, use_container_width=True)
        
        # Risk analysis
        if hasattr(st.session_state, 'backtest_results'):
            st.subheader("Risk Analysis")
            
            results_data = st.session_state.backtest_results['data']
            
            # Monte Carlo simulation
            if st.button("Run Monte Carlo Simulation"):
                with st.spinner("Running Monte Carlo simulation..."):
                    # Simple Monte Carlo for strategy returns
                    returns = results_data['Strategy_Return'].dropna()
                    
                    if len(returns) > 0:
                        n_simulations = 1000
                        n_days = 252  # 1 year
                        
                        # Generate random returns based on historical distribution
                        simulated_returns = np.random.choice(returns, size=(n_simulations, n_days), replace=True)
                        
                        # Calculate cumulative returns for each simulation
                        simulated_cumulative = np.cumprod(1 + simulated_returns, axis=1)
                        
                        # Plot results
                        fig_mc = go.Figure()
                        
                        # Plot a sample of simulations
                        for i in range(min(100, n_simulations)):
                            fig_mc.add_trace(go.Scatter(
                                y=simulated_cumulative[i] * initial_capital,
                                opacity=0.1,
                                line=dict(color='lightblue'),
                                showlegend=False
                            ))
                        
                        # Add percentiles
                        percentiles = [5, 25, 50, 75, 95]
                        for p in percentiles:
                            pct_values = np.percentile(simulated_cumulative * initial_capital, p, axis=0)
                            fig_mc.add_trace(go.Scatter(
                                y=pct_values,
                                name=f'{p}th percentile',
                                line=dict(width=2)
                            ))
                        
                        fig_mc.update_layout(
                            title="Monte Carlo Simulation - 1 Year Projection",
                            xaxis_title="Days",
                            yaxis_title="Portfolio Value ($)",
                            height=500
                        )
                        st.plotly_chart(fig_mc, use_container_width=True)
    
    with tab5:
        st.header("Performance Summary")
        
        # Create comprehensive summary
        summary_data = {}
        
        # Data summary
        summary_data['Data'] = {
            'Ticker': ticker_symbol,
            'Period': f"{data_period} year(s)",
            'Data Points': len(data),
            'Start Date': data.index[0].strftime('%Y-%m-%d'),
            'End Date': data.index[-1].strftime('%Y-%m-%d')
        }
        
        # ML summary
        if hasattr(st.session_state, 'ml_metrics'):
            summary_data['ML Model'] = {
                'Model Type': model_type,
                'RMSE': f"{st.session_state.ml_metrics['rmse']:.4f}",
                'R²': f"{st.session_state.ml_metrics['r2']:.4f}",
                'Hit Rate': f"{st.session_state.ml_metrics['hit_rate']:.2%}"
            }
        
        # Backtest summary
        if hasattr(st.session_state, 'backtest_results'):
            metrics = st.session_state.backtest_results['metrics']
            summary_data['Backtest'] = {
                'Strategy': strategy_type,
                'Total Return': f"{metrics['total_return']:.2%}",
                'Sharpe Ratio': f"{metrics['sharpe']:.2f}",
                'Max Drawdown': f"{metrics['max_drawdown']:.2%}",
                'Number of Trades': metrics['num_trades']
            }
        
        # Display summary
        for section, data_dict in summary_data.items():
            st.subheader(section)
            for key, value in data_dict.items():
                st.write(f"**{key}:** {value}")
            st.divider()
        
        # Export functionality
        if st.button("Export Results to CSV"):
            if hasattr(st.session_state, 'backtest_results'):
                results_df = st.session_state.backtest_results['data']
                csv = results_df.to_csv()
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{ticker_symbol}_backtest_results.csv",
                    mime="text/csv"
                )
else:
    st.info("Please enter a ticker symbol to begin analysis.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    QuantLab v2.0 - Advanced Trading Simulator | Built with Streamlit
</div>
""", unsafe_allow_html=True)