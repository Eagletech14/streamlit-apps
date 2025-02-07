import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import json
import io

# ---------------------------
# Helper Functions
# ---------------------------
def fetch_data(ticker, period, interval):
    """
    Fetch historical data for a given ticker using yfinance.
    """
    try:
        data = yf.download(ticker, period=period, interval=interval)
        if not data.empty:
            data.index = pd.to_datetime(data.index)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def calculate_rsi(series, period=14):
    """Calculate Relative Strength Index (RSI) for a given price series."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_pivot_points(high, low, close, method='Traditional'):
    """
    Calculate pivot points based on the high, low, and close values.
    """
    if method == 'Traditional':
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        return {"Pivot": pivot, "R1": r1, "S1": s1, "R2": r2, "S2": s2}
    elif method == 'Camarilla':
        range_val = high - low
        r1 = close + (range_val * 1.1 / 12)
        r2 = close + (range_val * 1.1 / 6)
        r3 = close + (range_val * 1.1 / 4)
        r4 = close + (range_val * 1.1 / 2)
        s1 = close - (range_val * 1.1 / 12)
        s2 = close - (range_val * 1.1 / 6)
        s3 = close - (range_val * 1.1 / 4)
        s4 = close - (range_val * 1.1 / 2)
        return {"R1": r1, "R2": r2, "R3": r3, "R4": r4,
                "S1": s1, "S2": s2, "S3": s3, "S4": s4}
    else:
        return {}

def get_date_series(df):
    """Helper to return the appropriate date series for the x-axis."""
    if 'Date' in df.columns:
        return df['Date']
    elif 'Datetime' in df.columns:
        return df['Datetime']
    else:
        return df.index

# ---------------------------
# Main Application
# ---------------------------
def main():
    st.title("Forex Market Visualizer")
    st.write("Analyze and visualize Forex data in real time with interactive charts.")

    # Sidebar Configuration
    st.sidebar.header("Chart Configuration")

    # Forex Pair Selection
    forex_pairs = {
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "USD/JPY": "USDJPY=X",
        "AUD/USD": "AUDUSD=X",
        "USD/CHF": "USDCHF=X"
    }
    selected_pair_name = st.sidebar.selectbox("Select Forex Pair", list(forex_pairs.keys()))
    selected_pair = forex_pairs[selected_pair_name]

    # Interval Selection (from 1 minute up to 24 hours)
    interval_options = ["1m", "5m", "15m", "30m", "1h", "1d"]
    selected_interval = st.sidebar.selectbox("Select Timeframe Interval", interval_options)

    # Fetch data
    st.subheader(f"Fetching data for {selected_pair_name}...")
    with st.spinner("Downloading data..."):
        df = fetch_data(selected_pair, period="7d", interval=selected_interval)
    if df.empty:
        st.error("No data fetched. Please check your ticker, date range, or interval settings.")
        return

    # Reset index so that the date becomes a column
    df.reset_index(inplace=True)
    date_series = get_date_series(df)

    # Build the Chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=date_series,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Price"
    ))
    fig.update_layout(
        title=f"{selected_pair_name} Price Chart ({selected_interval} interval)",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
