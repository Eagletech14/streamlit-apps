import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json

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

def compute_pivot_points_from_row(row, pivot_type):
    """
    Compute pivot points based on the last aggregated row of data and the chosen method.
    For 'Traditional', we compute the standard pivots.
    For 'Camarilla', we use the Camarilla formulas.
    """
    high = row["High"]
    low = row["Low"]
    close = row["Close"]
    
    if pivot_type == "Traditional":
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        return {"Pivot": pivot, "R1": r1, "S1": s1, "R2": r2, "S2": s2}
    elif pivot_type == "Camarilla":
        # Camarilla pivot point formulas (using a common factor 1.1)
        r4 = close + (high - low) * 1.1 / 2
        r3 = close + (high - low) * 1.1 / 4
        r2 = close + (high - low) * 1.1 / 6
        r1 = close + (high - low) * 1.1 / 12
        s1 = close - (high - low) * 1.1 / 12
        s2 = close - (high - low) * 1.1 / 6
        s3 = close - (high - low) * 1.1 / 4
        s4 = close - (high - low) * 1.1 / 2
        # We also include the close price as a reference pivot level.
        return {"R4": r4, "R3": r3, "R2": r2, "R1": r1, "Pivot": close, "S1": s1, "S2": s2, "S3": s3, "S4": s4}
    else:
        return {}

def create_chart(df, ticker, chart_type="Candlestick", 
                 up_color="#26a69a", down_color="#ef5350", line_color="#000000",
                 show_ma=False, ma_window=20, ma_color="#FFA500",
                 show_pivots=False, pivot_points=None):
    """
    Create an interactive chart based on the selected chart type.
    Overlays (MA and Pivot Points) are added if enabled.
    """
    if chart_type == "Candlestick":
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            increasing_line_color=up_color,
            decreasing_line_color=down_color,
            name="Candlesticks"
        )])
    elif chart_type == "Bars":
        fig = go.Figure(data=[go.Ohlc(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            increasing_line_color=up_color,
            decreasing_line_color=down_color,
            name="OHLC Bars"
        )])
    elif chart_type == "Line":
        fig = go.Figure(data=[go.Scatter(
            x=df.index,
            y=df["Close"],
            mode='lines',
            line=dict(color=line_color),
            name="Close Price"
        )])
    else:
        fig = go.Figure()  # Fallback if needed

    # Overlay Moving Average if enabled
    if show_ma and "MA" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["MA"],
            mode='lines',
            line=dict(color=ma_color, width=2),
            name=f"MA ({ma_window})"
        ))

    # Overlay Pivot Points if enabled
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
        title=f"{ticker} {chart_type} Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )
    return fig

def create_rsi_chart(df, rsi_overbought, rsi_oversold, rsi_color="#0000FF"):
    """
    Create an RSI chart using Plotly.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["RSI"],
        mode='lines',
        line=dict(color=rsi_color),
        name="RSI"
    ))
    # Add horizontal lines for overbought and oversold thresholds
    fig.add_hline(y=rsi_overbought, line_dash="dash", line_color="red",
                  annotation_text="Overbought", annotation_position="top left")
    fig.add_hline(y=rsi_oversold, line_dash="dash", line_color="green",
                  annotation_text="Oversold", annotation_position="bottom left")
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

    # Chart Type and Colours
    chart_type = st.sidebar.selectbox("Chart Type", options=["Candlestick", "Bars", "Line"], index=0)
    up_color = st.sidebar.color_picker("Up Color", value=config.get("up_color", "#26a69a"))
    down_color = st.sidebar.color_picker("Down Color", value=config.get("down_color", "#ef5350"))
    line_color = st.sidebar.color_picker("Line Color", value=config.get("line_color", "#000000"))

    # Moving Average options
    show_ma = st.sidebar.checkbox("Show Moving Average", value=config.get("show_ma", True))
    if show_ma:
        ma_window = st.sidebar.number_input("Moving Average Window", min_value=1, max_value=200, value=int(config.get("ma_window", 20)))
        ma_color = st.sidebar.color_picker("MA Line Color", value=config.get("ma_color", "#FFA500"))
    else:
        ma_window, ma_color = None, None

    # RSI options
    show_rsi = st.sidebar.checkbox("Show RSI", value=config.get("show_rsi", True))
    if show_rsi:
        rsi_window = st.sidebar.number_input("RSI Window", min_value=1, max_value=50, value=int(config.get("rsi_window", 14)))
        rsi_overbought = st.sidebar.slider("RSI Overbought Level", min_value=50, max_value=100, value=int(config.get("rsi_overbought", 70)))
        rsi_oversold = st.sidebar.slider("RSI Oversold Level", min_value=0, max_value=50, value=int(config.get("rsi_oversold", 30)))
        rsi_color = st.sidebar.color_picker("RSI Line Color", value=config.get("rsi_color", "#0000FF"))
    else:
        rsi_window, rsi_overbought, rsi_oversold, rsi_color = None, None, None, None

    # Pivot Points options
    show_pivots = st.sidebar.checkbox("Show Pivot Points", value=config.get("show_pivots", True))
    if show_pivots:
        pivot_calc_timeframe = st.sidebar.selectbox("Pivot Calculation Time Frame", 
                                                    options=["5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "24h"],
                                                    index=config.get("pivot_calc_timeframe_index", 3))  # Default "1h"
        pivot_type = st.sidebar.selectbox("Pivot Calculation Method", options=["Traditional", "Camarilla"], index=0)
    else:
        pivot_calc_timeframe, pivot_type = None, None

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
    if show_pivots and pivot_calc_timeframe and pivot_type:
        # Mapping of pivot timeframe to pandas offset aliases
        timeframe_map = {
            "5m": "5T",
            "15m": "15T",
            "30m": "30T",
            "1h": "1H",
            "2h": "2H",
            "4h": "4H",
            "6h": "6H",
            "12h": "12H",
            "24h": "24H"
        }
        pivot_freq = timeframe_map[pivot_calc_timeframe]
        # Resample data to the chosen pivot calculation timeframe
        df_pivot = df.resample(pivot_freq).agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last"
        }).dropna()
        if not df_pivot.empty:
            last_row = df_pivot.iloc[-1]
            pivot_points = compute_pivot_points_from_row(last_row, pivot_type)
        else:
            pivot_points = {}
    else:
        pivot_points = {}

    # ------------------------------
    # Chart Visualization
    # ------------------------------

    # Create and display the selected chart with overlays
    fig = create_chart(df, ticker, chart_type, up_color, down_color, line_color,
                       show_ma, ma_window, ma_color, show_pivots, pivot_points)
    st.plotly_chart(fig, use_container_width=True)

    # Display RSI chart if enabled
    if show_rsi and rsi_window:
        st.plotly_chart(create_rsi_chart(df, rsi_overbought, rsi_oversold, rsi_color), use_container_width=True)

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
    # Export Chart Functionality
    # ------------------------------

    st.markdown("## Export Chart")
    if st.button("Export Current Chart"):
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
            "chart_type": chart_type,
            "up_color": up_color,
            "down_color": down_color,
            "line_color": line_color,
            "show_ma": show_ma,
            "ma_window": ma_window,
            "ma_color": ma_color,
            "show_rsi": show_rsi,
            "rsi_window": rsi_window,
            "rsi_overbought": rsi_overbought,
            "rsi_oversold": rsi_oversold,
            "rsi_color": rsi_color,
            "show_pivots": show_pivots,
            "pivot_calc_timeframe_index": ["5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "24h"].index(pivot_calc_timeframe) if pivot_calc_timeframe else 3
            # You might also want to store pivot_type if needed.
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
