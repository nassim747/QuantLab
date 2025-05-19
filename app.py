import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Stock Price Tracker",
    layout="wide"
)

# Title
st.title("Stock Price Tracker")

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

            st.success(f"Downloaded {len(data)} days of data for {ticker_symbol}")

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