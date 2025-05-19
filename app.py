import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Page config
st.set_page_config(
    page_title="Stock Price Tracker",
    layout="wide"
)

# Header with title and author info
st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: center;">
      <h1 style="margin: 0;">📈 Stock Price Tracker</h1>
      <div style="text-align: right;">
        <span style="font-size:0.9rem; color: #555;">Built by </span>
        <a href="https://www.linkedin.com/in/nassim-a-265944286/" target="_blank"
           style="font-weight: bold; color: #0A66C2; text-decoration: none;">
          Nassim Ameur
        </a>
      </div>
    </div>
    <hr style="margin-top: 0.5rem; margin-bottom: 1rem;">
    """,
    unsafe_allow_html=True
)

# Data loading function with caching
@st.cache_data(ttl=3600)
def load_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)

# Sidebar: ML preprocessing & model selection
with st.sidebar:
    st.header("ML Preprocessing")

    # Forecast horizon slider
    forecast_horizon = st.slider(
        "Forecast Horizon (days)",
        1, 30, 5,
        help="How many days ahead to predict"
    )

    # Return type selection
    use_log_returns = st.checkbox(
        "Use Log Returns",
        value=False,
        help="Use ln(P_t / P_{t-1}) if checked, otherwise pct change"
    )
    st.info("Using " + ("log returns" if use_log_returns else "percent returns"))

    st.header("ML Model")
    model_type = st.selectbox(
        "Choose model",
        ["Linear Regression", "Random Forest Regressor", "XGBoost Regressor"]
    )

# Ticker input
ticker_symbol = st.text_input(
    "Enter a ticker symbol (e.g., AAPL, MSFT, GOOG):",
    "AAPL"
)

if ticker_symbol:
    # Download data
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5 * 365)
        st.info(f"Downloading {ticker_symbol} data…")
        data = load_data(ticker_symbol, start_date, end_date)
        
        # Handle MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            st.info("MultiIndex columns detected, flattening...")
            # Flatten the MultiIndex to single level - use only the first level
            data.columns = data.columns.get_level_values(0)
        
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        st.stop()

    if data.empty:
        st.error(f"No data found for ticker '{ticker_symbol}'.")
        st.stop()

    # Clean index
    data.index = pd.to_datetime(data.index, errors="coerce")
    data = data[~data.index.duplicated()].sort_index()

    # Create ML columns
    data["Target"] = data["Close"].shift(-forecast_horizon)
    if use_log_returns:
        data["LogReturn"] = np.log(data["Close"] / data["Close"].shift(1)).fillna(0)
        feature_col = "LogReturn"
    else:
        data["Return"] = data["Close"].pct_change().fillna(0)
        feature_col = "Return"

    st.success(f"Downloaded {len(data)} days of data for {ticker_symbol}")

    # Prepare ML dataset
    required = [feature_col, "Target"]
    missing = [col for col in required if col not in data.columns]
    if missing:
        st.error(f"Missing columns before ML split: {missing}")
        st.stop()

    ml_data = data[required].dropna()
    X = ml_data[[feature_col]]
    y = ml_data["Target"]

    # Train/test split (80/20)
    split_idx = int(len(ml_data) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # ML Dataset Info
    with st.expander("ML Dataset Info", expanded=True):
        st.write(f"Total points: {len(ml_data)}")
        st.write(f"Train samples: {len(X_train)}")
        st.write(f"Test samples:  {len(X_test)}")
        st.write(f"Feature col:   {feature_col}")
        st.write(f"Target col:    Target ({forecast_horizon}-day ahead price)")
        # Compute correlation matrix for feature and target only
        corr_matrix = ml_data[[feature_col, "Target"]].corr()
        corr = corr_matrix.iloc[0, 1]
        st.write(f"Correlation:   {corr:.4f}")

    # Raw data expander
    with st.expander("Show raw data (head & tail)"):
        st.write("Current columns:", data.columns.tolist())
        st.write(data.head())
        st.write(data.tail())

    # Key metrics
    latest = float(data["Close"].iloc[-1])
    prev = float(data["Close"].iloc[-2])
    delta = latest - prev
    pct = delta / prev * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Latest Close", f"${latest:.2f}")
    c2.metric("Change", f"${delta:.2f}", f"{pct:.2f}%")
    c3.metric(
        "Data Period",
        f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}"
    )

    # Plots
    if len(data) > 30:
        st.subheader("Closing Price Over Time")
        
        # Prepare plot data correctly - convert to DataFrame with proper columns
        df_plot = pd.DataFrame({
            'Date': data.index,
            'Close': data['Close'].values
        })
        
        # Plot closing price
        fig1 = px.line(
            df_plot, 
            x="Date", 
            y="Close",
            title=f"{ticker_symbol} Close (5Y)", 
            labels={"Close": "USD"}
        )
        fig1.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig1, use_container_width=True)

        # Plot feature (returns)
        st.subheader(f"{feature_col} Over Time")
        df_feat = pd.DataFrame({
            'Date': data.index,
            feature_col: data[feature_col].values
        })
        
        fig2 = px.line(
            df_feat, 
            x="Date", 
            y=feature_col,
            title=f"{ticker_symbol} {feature_col} (5Y)"
        )
        fig2.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Not enough data points to plot.")

    # ML Model Training placeholder
    # Check if retraining is requested to determine if expander should be open
    if 'retrain_requested' not in st.session_state:
        st.session_state['retrain_requested'] = False
    
    with st.expander("🔧 ML Model Training", expanded=st.session_state.get('retrain_requested', False)):
        # Initialize session state variables
        if 'training_started' not in st.session_state:
            st.session_state['training_started'] = False
            
        # Button to start training or show training is in progress
        if not st.session_state['training_started']:
            start_training = st.button("Train Model")
            if start_training:
                st.session_state['training_started'] = True
                st.session_state['retrain_requested'] = True
                st.rerun()  # Rerun to update UI with training state
        else:
            st.write("**Training in progress...**")
            
            # Allow changing model within the training section
            model_type = st.selectbox(
                "Choose model to train",
                ["Linear Regression", "Random Forest Regressor", "XGBoost Regressor"],
                index=["Linear Regression", "Random Forest Regressor", "XGBoost Regressor"].index(model_type)
            )
            
            st.write(f"Selected model: **{model_type}**")
            
            # Define default hyperparameters
            n_estimators = 100  # Default value
            max_depth = 10      # Default value
            learning_rate = 0.1  # Default value for XGBoost
            
            # Using a checkbox instead of a nested expander
            show_hyperparams = st.checkbox("⚙️ Show Advanced Hyperparameters", value=False)
            
            if show_hyperparams:
                if model_type == "Random Forest Regressor":
                    n_estimators = st.slider("RF: n_estimators", 10, 500, 100, step=10)
                    max_depth = st.slider("RF: max_depth", 1, 50, 10)
                elif model_type == "XGBoost Regressor":
                    n_estimators = st.slider("XGB: n_estimators", 10, 500, 100, step=10)
                    learning_rate = st.slider("XGB: learning_rate", 0.01, 0.3, 0.1, step=0.01)
                    max_depth = st.slider("XGB: max_depth", 1, 10, 3)
                elif model_type == "Linear Regression":
                    st.info("Linear Regression has no hyperparameters to tune.")
            
            # Retrain button
            if st.button("Retrain Model"):
                st.session_state['retrain_requested'] = True
                st.rerun()
            
            # Reset button to restart from scratch
            if st.button("Reset Training"):
                st.session_state['training_started'] = False
                st.session_state['retrain_requested'] = False
                if 'model' in st.session_state:
                    del st.session_state['model']
                if 'y_pred' in st.session_state:
                    del st.session_state['y_pred']
                st.rerun()
            
            # Check if model already exists in session state or training is requested
            retrain_requested = st.session_state['retrain_requested']
            
            if "model" not in st.session_state or retrain_requested:
                # Instantiate the appropriate model based on selection
                if model_type == "Linear Regression":
                    model = LinearRegression()
                elif model_type == "Random Forest Regressor":
                    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                else:  # XGBoost
                    model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, use_label_encoder=False, eval_metric="rmse", random_state=42)
                
                # Show training status
                st.info(f"Training {model_type} on {len(X_train)} samples…")
                
                # Train the model
                with st.spinner("Training model, please wait…"):
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Store in session state
                st.session_state["model"] = model
                st.session_state["y_pred"] = y_pred
                
                # Reset retrain flag
                st.session_state['retrain_requested'] = False
            else:
                # Use existing model and predictions
                model = st.session_state["model"]
                y_pred = st.session_state["y_pred"]
                st.info("Using previously trained model")
            
            # Compute evaluation metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Create DataFrame for plotting
            df_pred = pd.DataFrame({
                "Date": X_test.index,
                "Actual": y_test.values,
                "Predicted": y_pred
            })
            
            # Display metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("RMSE", f"{rmse:.4f}")
            c2.metric("MAE", f"{mae:.4f}")
            c3.metric("R²", f"{r2:.4f}")
            
            # Plot actual vs predicted
            fig = px.line(df_pred, x="Date", y=["Actual", "Predicted"],
                      labels={"value": "Price", "variable": "Series"})
            fig.update_layout(title="Actual vs Predicted Prices")
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("✅ Model trained successfully!")

    # Strategy Simulator expander
    with st.expander("🚀 Strategy Simulator", expanded=False):
        # Trading parameters
        entry_threshold = st.slider(
            "Entry Threshold", 
            min_value=-0.05, 
            max_value=0.05, 
            value=0.0, 
            step=0.005
        )
        
        transaction_cost = st.number_input(
            "Transaction Cost (per trade, as fraction)",
            min_value=0.0,
            max_value=0.1,
            value=0.001,
            step=0.001,
            format="%.4f"
        )
        
        # Only show backtest results if model has been trained
        if 'training_started' in st.session_state and st.session_state['training_started'] and 'y_pred' in st.session_state:
            # Get predictions from session state
            y_pred = st.session_state['y_pred']
            
            # Backtest trading strategy
            df_bt = pd.DataFrame({
                "Date": X_test.index,
                "Price": data["Close"].loc[X_test.index].values,
                "Pred": y_pred
            })
            
            # Generate signal: 1 if Pred > Price + threshold else 0
            df_bt["Signal"] = (df_bt["Pred"] > df_bt["Price"] + entry_threshold).astype(int)
            
            # Daily market returns
            df_bt["Market_Return"] = df_bt["Price"].pct_change().fillna(0)
            
            # Position lags the signal by one day
            df_bt["Position"] = df_bt["Signal"].shift(1).fillna(0)
            
            # Raw strategy returns
            df_bt["Strat_Return"] = df_bt["Position"] * df_bt["Market_Return"]
            
            # Subtract transaction costs on changes in Position
            df_bt["Strat_Return"] -= transaction_cost * df_bt["Position"].diff().abs().fillna(0)
            
            # Build equity curve
            df_bt["Equity"] = (1 + df_bt["Strat_Return"]).cumprod()
            
            # Compute annualized stats
            ann_return = df_bt["Equity"].iloc[-1] ** (252/len(df_bt)) - 1
            ann_vol    = df_bt["Strat_Return"].std() * (252**0.5)
            sharpe     = ann_return / ann_vol
            dd         = (df_bt["Equity"].cummax() - df_bt["Equity"]) / df_bt["Equity"].cummax()
            max_dd     = dd.max()
            
            # Show via st.metric
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Ann Return", f"{ann_return:.2%}")
            r2.metric("Ann Vol",    f"{ann_vol:.2%}")
            r3.metric("Sharpe",      f"{sharpe:.2f}")
            r4.metric("Max Drawdown", f"{max_dd:.2%}")
            
            # Plot equity curve
            fig_eq = px.line(
                df_bt, x="Date", y="Equity",
                title="Strategy Equity Curve"
            )
            fig_eq.update_xaxes(rangeslider_visible=False)
            st.plotly_chart(fig_eq, use_container_width=True)

    # Recent data table
    st.subheader("Recent Price Data")
    st.dataframe(data.tail(10))