import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
from io import BytesIO

# ------------------------------
# Helper Functions
# ------------------------------

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

def compute_rsi(series, window):
    """
    Compute the Relative Strength Index (RSI) for a given price series.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss.replace(to_replace=0, method="ffill")
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_pivot_points(df):
    """
    Compute standard pivot points using the last row of the dataframe.
    """
    if df.empty:
        return {}
    last = df.iloc[-1]
    high = last["High"]
    low = last["Low"]
    close = last["Close"]
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    return {"pivot": pivot, "R1": r1, "S1": s1}

def create_candlestick_chart(df, ticker, show_ma=False, ma_window=20, show_pivots=False, pivot_points=None):
    """
    Create an interactive Plotly candlestick chart and overlay moving average and pivot points if required.
    """
    fig = go.Figure(data=[go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Candlesticks"
    )])
    
    if show_ma and 'MA' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MA'], mode='lines', line=dict(width=2), name=f"MA ({ma_window})"
        ))
    
    if show_pivots and pivot_points:
        for key, value in pivot_points.items():
            fig.add_trace(go.Scatter(
                x=df.index, y=[value] * len(df.index), mode='lines', line=dict(dash='dash'), name=key
            ))
    
    fig.update_layout(title=f"{ticker} Candlestick Chart", xaxis_rangeslider_visible=False)
    return fig

def main():
    st.title("Forex Market Visualizer")
    st.markdown("### Analyze and visualize Forex market data with interactive charts")

    ticker = st.sidebar.text_input("Forex Pair Symbol", value="EURUSD=X")
    period = st.sidebar.selectbox("Data Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"], index=5)
    interval = st.sidebar.selectbox("Data Interval", ["1m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"], index=7)

    df = fetch_data(ticker, period, interval)
    if df.empty:
        st.error("No data fetched. Check the Forex pair symbol or settings.")
        return

    show_ma = st.sidebar.checkbox("Show Moving Average", value=True)
    ma_window = st.sidebar.number_input("Moving Average Window", min_value=1, max_value=200, value=20) if show_ma else None
    if show_ma:
        df["MA"] = df["Close"].rolling(window=ma_window).mean()

    show_pivots = st.sidebar.checkbox("Show Pivot Points", value=True)
    pivot_points = compute_pivot_points(df) if show_pivots else None

    fig = create_candlestick_chart(df, ticker, show_ma, ma_window, show_pivots, pivot_points)
    st.plotly_chart(fig, use_container_width=True)
    
    symbols_input = st.text_input("Enter multiple Forex pair symbols separated by commas", value="EURUSD=X,GBPUSD=X,USDJPY=X")
    symbols = [s.strip() for s in symbols_input.split(",") if s.strip()]
    
    if symbols:
        price_data = {}
        for sym in symbols:
            data = fetch_data(sym, period, interval)
            if not data.empty:
                price_data[sym] = data["Close"]
        
        if price_data:
            price_df = pd.DataFrame(price_data)
            if not price_df.empty:
                corr_matrix = price_df.corr()
                st.write("### Correlation Matrix", corr_matrix)
                heatmap_fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='Viridis'
                ))
                heatmap_fig.update_layout(title="Forex Pair Correlation Heatmap")
                st.plotly_chart(heatmap_fig, use_container_width=True)
            else:
                st.error("No valid data available for correlation analysis.")

if __name__ == '__main__':
    main()
