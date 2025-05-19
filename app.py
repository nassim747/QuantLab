import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

# Page config
st.set_page_config(
    page_title="Stock Price Tracker",
    layout="wide"
)

# Title
st.title("Stock Price Tracker")

# Add sidebar with ML preprocessing options
with st.sidebar:
    st.header("ML Preprocessing")
    
    # Forecast horizon slider
    forecast_horizon = st.slider(
        "Forecast Horizon (days)",
        min_value=1,
        max_value=30,
        value=5,
        help="Number of days to forecast into the future"
    )
    
    # Return type selection
    use_log_returns = st.checkbox(
        "Use Log Returns",
        value=False,
        help="If checked, use log returns. Otherwise, use percent returns."
    )
    
    if use_log_returns:
        st.info("Using logarithmic returns for calculations")
    else:
        st.info("Using percentage returns for calculations")

# Ticker input
ticker_symbol = st.text_input(
    "Enter a ticker symbol (e.g., AAPL, MSFT, GOOG):",
    "AAPL"
)

if ticker_symbol:
    try:
        # Date range (5 years)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5 * 365)

        # Download
        st.info(f"Downloading data for {ticker_symbol}...")
        data = yf.download(
            ticker_symbol,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False
        )

        if not data.empty:
            # Ensure clean DatetimeIndex
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            data = data[~data.index.duplicated(keep="first")]
            data = data.sort_index()

            # Create target column by shifting Close prices by -forecast_horizon
            data["Target"] = data["Close"].shift(-forecast_horizon)
            
            # Calculate returns based on user selection
            if use_log_returns:
                # Log returns: ln(price_t / price_{t-1})
                data["LogReturn"] = np.log(data["Close"] / data["Close"].shift(1))
                data["LogReturn"].fillna(0, inplace=True)
                feature_col = "LogReturn"
            else:
                # Percentage returns: (price_t - price_{t-1}) / price_{t-1}
                data["Return"] = data["Close"].pct_change()
                data["Return"].fillna(0, inplace=True)
                feature_col = "Return"
            
            st.success(f"Downloaded {len(data)} days of data for {ticker_symbol}")

            # ML data preparation
            # Drop rows with NaNs in feature or target columns
            ml_data = data.dropna(subset=[feature_col, "Target"])
            
            # Split into X (features) and y (target)
            X = ml_data[[feature_col]]
            y = ml_data["Target"]
            
            # Split into train/test sets (80/20) without shuffling to preserve time order
            split_idx = int(len(ml_data) * 0.8)
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
            
            # Show the split info
            with st.expander("ML Dataset Info"):
                st.write(f"Total clean data points: {len(ml_data)}")
                st.write(f"Training set: {len(X_train)} samples")
                st.write(f"Test set: {len(X_test)} samples")
                st.write(f"Feature column: {feature_col}")
                st.write(f"Target column: Target (Close price in {forecast_horizon} days)")
                
                # Show correlation between feature and target
                correlation = ml_data[[feature_col, "Target"]].corr().iloc[0, 1]
                st.write(f"Correlation between {feature_col} and Target: {correlation:.4f}")

            # Debug raw data
            with st.expander("Show raw data (head & tail)"):
                st.write(data.head())
                st.write(data.tail())

            # Key metrics
            latest_price    = data["Close"].iloc[-1].item()
            prev_close      = data["Close"].iloc[-2].item()
            price_change    = latest_price - prev_close
            price_change_pct = (price_change / prev_close) * 100

            col1, col2, col3 = st.columns(3)
            col1.metric("Latest Close", f"${latest_price:.2f}")
            col2.metric("Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")

            start_str = data.index[0].strftime("%Y-%m-%d")
            end_str   = data.index[-1].strftime("%Y-%m-%d")
            col3.metric("Data Period", f"{start_str} to {end_str}")

            # Visualization
            if len(data) > 30:
                st.subheader("Closing Price Over Time")

                # Extract a 1D series
                close_series = data["Close"]
                # (optional) force it to a Series even if it's a single-col DataFrame:
                if isinstance(close_series, pd.DataFrame):
                    close_series = close_series.iloc[:, 0]

                # Prepare df_plot
                df_plot = close_series.reset_index()
                df_plot.columns = ["Date", "Close"]

                # Now plot
                fig = px.line(
                    df_plot,
                    x="Date",
                    y="Close",
                    title=f"{ticker_symbol} Stock Price (5 Years)",
                    labels={"Close": "Price (USD)", "Date": "Date"}
                )
                fig.update_xaxes(rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Plot returns
                st.subheader(f"{feature_col} Over Time")
                return_series = data[feature_col]
                df_return = return_series.reset_index()
                df_return.columns = ["Date", feature_col]
                
                return_fig = px.line(
                    df_return,
                    x="Date",
                    y=feature_col,
                    title=f"{ticker_symbol} {feature_col} (5 Years)",
                )
                return_fig.update_xaxes(rangeslider_visible=False)
                st.plotly_chart(return_fig, use_container_width=True)
            else:
                st.warning("Not enough data points to create a chart.")

            # Recent data table
            st.subheader("Recent Price Data")
            st.dataframe(data.tail(10))

        else:
            st.error(f"No data found for ticker: {ticker_symbol}")

    except Exception as e:
        st.error(f"Error retrieving data for {ticker_symbol}: {e}")
        st.info("Please check if the ticker symbol is correct and try again.")