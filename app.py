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
    # Additional pivot points (R2, S2) can be computed similarly if desired.
    return {"pivot": pivot, "R1": r1, "S1": s1}

def create_candlestick_chart(df, ticker, show_ma=False, ma_window=20, show_pivots=False, pivot_points=None):
    """
    Create an interactive Plotly candlestick chart and overlay moving average and pivot points if required.
    """
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Candlesticks"
    )])

    # Overlay Moving Average
    if show_ma and 'MA' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MA'],
            mode='lines',
            line=dict(width=2),
            name=f"MA ({ma_window})"
        ))

    # Overlay Pivot Points
    if show_pivots and pivot_points:
        for key, value in pivot_points.items():
            fig.add_trace(go.Scatter(
                x=df.index,
                y=[value] * len(df.index),
                mode='lines',
                line=dict(dash='dash'),
                name=key
            ))

    fig.update_layout(
        title=f"{ticker} Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )
    return fig

def create_rsi_chart(df, rsi_overbought, rsi_oversold):
    """
    Create an RSI chart using Plotly.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["RSI"],
        mode='lines',
        name="RSI"
    ))
    # Add horizontal lines for overbought and oversold thresholds
    fig.add_hline(y=rsi_overbought, line_dash="dash", line_color="red", annotation_text="Overbought", annotation_position="top left")
    fig.add_hline(y=rsi_oversold, line_dash="dash", line_color="green", annotation_text="Oversold", annotation_position="bottom left")
    fig.update_layout(
        title="Relative Strength Index (RSI)",
        xaxis_title="Date",
        yaxis_title="RSI Value",
        yaxis=dict(range=[0, 100])
    )
    return fig

# ------------------------------
# Main App Function
# ------------------------------

def main():
    st.title("Forex Market Visualizer")
    st.markdown("### Analyze and visualize Forex market data with interactive charts")

    # ------------------------------
    # Sidebar: Configuration and Settings
    # ------------------------------
    st.sidebar.header("User Configuration")

    # Option to load a saved configuration
    config_file = st.sidebar.file_uploader("Upload Configuration JSON", type=["json"])
    if config_file is not None:
        try:
            config = json.load(config_file)
        except Exception as e:
            st.sidebar.error(f"Error reading configuration: {e}")
            config = {}
    else:
        config = {}

    # Forex pair symbol input (yfinance uses symbols like 'EURUSD=X')
    ticker = st.sidebar.text_input("Forex Pair Symbol", value=config.get("ticker", "EURUSD=X"))

    # Data period and interval settings
    period = st.sidebar.selectbox("Data Period", 
                                  options=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
                                  index=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"].index(config.get("period", "1y")))
    interval = st.sidebar.selectbox("Data Interval", 
                                    options=["1m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"],
                                    index=["1m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"].index(config.get("interval", "1d")))

    # Moving Average options
    show_ma = st.sidebar.checkbox("Show Moving Average", value=config.get("show_ma", True))
    if show_ma:
        ma_window = st.sidebar.number_input("Moving Average Window", min_value=1, max_value=200, value=int(config.get("ma_window", 20)))
    else:
        ma_window = None

    # RSI options
    show_rsi = st.sidebar.checkbox("Show RSI", value=config.get("show_rsi", True))
    if show_rsi:
        rsi_window = st.sidebar.number_input("RSI Window", min_value=1, max_value=50, value=int(config.get("rsi_window", 14)))
        rsi_overbought = st.sidebar.slider("RSI Overbought Level", min_value=50, max_value=100, value=int(config.get("rsi_overbought", 70)))
        rsi_oversold = st.sidebar.slider("RSI Oversold Level", min_value=0, max_value=50, value=int(config.get("rsi_oversold", 30)))
    else:
        rsi_window, rsi_overbought, rsi_oversold = None, None, None

    # Pivot points options
    show_pivots = st.sidebar.checkbox("Show Pivot Points", value=config.get("show_pivots", True))

    # Export options
    st.sidebar.markdown("### Export Options")
    export_format = st.sidebar.selectbox("Export Chart Format", options=["PNG", "HTML"], index=0)

    # ------------------------------
    # Data Fetching and Processing
    # ------------------------------

    st.subheader("Forex Data Chart")
    df = fetch_data(ticker, period, interval)
    if df.empty:
        st.error("No data fetched. Check the Forex pair symbol or settings.")
        return

    # Compute Moving Average if enabled
    if show_ma and ma_window:
        df["MA"] = df["Close"].rolling(window=ma_window).mean()

    # Compute RSI if enabled
    if show_rsi and rsi_window:
        df["RSI"] = compute_rsi(df["Close"], rsi_window)

    # Compute Pivot Points if enabled
    pivot_points = compute_pivot_points(df) if show_pivots else None

    # ------------------------------
    # Chart Visualization
    # ------------------------------

    # Create and display candlestick chart with overlays
    fig = create_candlestick_chart(df, ticker, show_ma, ma_window, show_pivots, pivot_points)
    st.plotly_chart(fig, use_container_width=True)

    # Display RSI chart if enabled
    if show_rsi and rsi_window:
        st.plotly_chart(create_rsi_chart(df, rsi_overbought, rsi_oversold), use_container_width=True)

    # ------------------------------
    # Portfolio Tracking
    # ------------------------------

    st.markdown("## Portfolio Tracking")
    st.info("Upload a CSV or Excel file containing your portfolio details (e.g., symbols, positions, cost basis).")
    portfolio_file = st.file_uploader("Upload Portfolio File", type=["csv", "xlsx"])
    if portfolio_file:
        try:
            if portfolio_file.name.endswith('.csv'):
                portfolio_df = pd.read_csv(portfolio_file)
            else:
                portfolio_df = pd.read_excel(portfolio_file)
            st.write("### Your Portfolio", portfolio_df)
            # You can add calculations for financial ratios or P&L analysis here.
        except Exception as e:
            st.error(f"Error reading portfolio file: {e}")

    # ------------------------------
    # Forex Correlation Analysis
    # ------------------------------

    st.markdown("## Forex Pair Correlation Analysis")
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

    # ------------------------------
    # Export Chart Functionality
    # ------------------------------

    st.markdown("## Export Chart")
    if st.button("Export Current Candlestick Chart"):
        if export_format == "PNG":
            # Export as PNG image (requires kaleido)
            try:
                img_bytes = fig.to_image(format="png")
                st.download_button(label="Download PNG", data=img_bytes, file_name="chart.png", mime="image/png")
            except Exception as e:
                st.error(f"Error exporting PNG: {e}")
        else:
            # Export as HTML
            html_bytes = fig.to_html()
            st.download_button(label="Download HTML", data=html_bytes, file_name="chart.html", mime="text/html")

    # ------------------------------
    # Save and Share User Configuration
    # ------------------------------

    st.sidebar.markdown("### Save Current Configuration")
    if st.sidebar.button("Save Configuration"):
        config_to_save = {
            "ticker": ticker,
            "period": period,
            "interval": interval,
            "show_ma": show_ma,
            "ma_window": ma_window,
            "show_rsi": show_rsi,
            "rsi_window": rsi_window,
            "rsi_overbought": rsi_overbought,
            "rsi_oversold": rsi_oversold,
            "show_pivots": show_pivots,
        }
        config_json = json.dumps(config_to_save, indent=4)
        st.download_button("Download Configuration", data=config_json, file_name="config.json", mime="application/json")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("Developed by *Your Name*")
    st.sidebar.markdown("For version control and collaboration, push this code to [GitHub](https://github.com/).")

# ------------------------------
# Run the App
# ------------------------------

if __name__ == '__main__':
    main()

