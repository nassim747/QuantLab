import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st


@st.cache_data(ttl=3600)
def load_stock_data(ticker, start_date, end_date):
    """Load stock data from Yahoo Finance with caching."""
    return yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)


def clean_data(data):
    """Clean and preprocess stock data."""
    # Handle MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # Clean index
    data.index = pd.to_datetime(data.index, errors="coerce")
    data = data[~data.index.duplicated()].sort_index()
    
    return data


def add_technical_indicators(data):
    """Add technical indicators to the dataset."""
    # Simple Moving Averages
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    # Exponential Moving Averages
    data['EMA_12'] = data['Close'].ewm(span=12).mean()
    data['EMA_26'] = data['Close'].ewm(span=26).mean()
    
    # MACD
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
    data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    bb_std = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
    data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
    data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
    data['BB_Position'] = (data['Close'] - data['BB_Lower']) / data['BB_Width']
    
    # Volume indicators (if available)
    if 'Volume' in data.columns:
        data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
    
    # Price-based features
    data['High_Low_Ratio'] = data['High'] / data['Low']
    data['Close_Open_Ratio'] = data['Close'] / data['Open']
    
    # Lagged returns
    for lag in [1, 2, 3, 5, 10]:
        data[f'Return_Lag_{lag}'] = data['Close'].pct_change(lag)
        data[f'LogReturn_Lag_{lag}'] = np.log(data['Close'] / data['Close'].shift(lag))
    
    # Rolling statistics
    for window in [5, 10, 20]:
        data[f'Return_Mean_{window}'] = data['Close'].pct_change().rolling(window).mean()
        data[f'Return_Std_{window}'] = data['Close'].pct_change().rolling(window).std()
        data[f'Price_Min_{window}'] = data['Close'].rolling(window).min()
        data[f'Price_Max_{window}'] = data['Close'].rolling(window).max()
    
    return data


def prepare_ml_features(data, forecast_horizon=5, use_log_returns=False):
    """Prepare features for machine learning."""
    # Add technical indicators
    data = add_technical_indicators(data)
    
    # Create target variable
    data["Target"] = data["Close"].shift(-forecast_horizon)
    
    # Basic return features
    if use_log_returns:
        data["LogReturn"] = np.log(data["Close"] / data["Close"].shift(1)).fillna(0)
        return_col = "LogReturn"
    else:
        data["Return"] = data["Close"].pct_change().fillna(0)
        return_col = "Return"
    
    # Define feature columns (excluding target and basic price columns)
    feature_cols = [
        return_col, 'SMA_10', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
        'MACD', 'MACD_Signal', 'MACD_Histogram', 'RSI',
        'BB_Position', 'BB_Width', 'High_Low_Ratio', 'Close_Open_Ratio'
    ]
    
    # Add lagged returns
    feature_cols.extend([f'Return_Lag_{lag}' for lag in [1, 2, 3, 5, 10]])
    
    # Add rolling statistics
    for window in [5, 10, 20]:
        feature_cols.extend([
            f'Return_Mean_{window}', f'Return_Std_{window}',
            f'Price_Min_{window}', f'Price_Max_{window}'
        ])
    
    # Add volume features if available
    if 'Volume' in data.columns:
        feature_cols.extend(['Volume_Ratio'])
    
    # Filter existing columns
    existing_features = [col for col in feature_cols if col in data.columns]
    
    # Create ML dataset
    ml_data = data[existing_features + ["Target"]].dropna()
    
    return ml_data, existing_features


def get_data_info(ticker_symbol, days=5*365):
    """Get stock data and basic information."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Load and clean data
        data = load_stock_data(ticker_symbol, start_date, end_date)
        data = clean_data(data)
        
        if data.empty:
            return None, f"No data found for ticker '{ticker_symbol}'"
        
        return data, None
        
    except Exception as e:
        return None, f"Error downloading data: {e}"


def calculate_basic_metrics(data):
    """Calculate basic price metrics."""
    latest = float(data["Close"].iloc[-1])
    prev = float(data["Close"].iloc[-2])
    delta = latest - prev
    pct = delta / prev * 100
    
    return {
        'latest_close': latest,
        'change': delta,
        'change_pct': pct,
        'start_date': data.index[0].strftime('%Y-%m-%d'),
        'end_date': data.index[-1].strftime('%Y-%m-%d'),
        'total_days': len(data)
    }