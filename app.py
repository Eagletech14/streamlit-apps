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
    For Traditional method:
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
    For Camarilla method:
        r1 = close + (high - low) * 1.1/12
        r2 = close + (high - low) * 1.1/6
        r3 = close + (high - low) * 1.1/4
        r4 = close + (high - low) * 1.1/2
        s1 = close - (high - low) * 1.1/12
        s2 = close - (high - low) * 1.1/6
        s3 = close - (high - low) * 1.1/4
        s4 = close - (high - low) * 1.1/2
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

    # ---------------------------
    # Sidebar Configuration
    # ---------------------------
    st.sidebar.header("Chart Configuration")

    # Forex Pair Manual Input (no pre-defined list)
    selected_pair = st.sidebar.text_input("Enter Forex Pair Ticker (e.g., EURUSD=X)", "EURUSD=X")

    # Date Range Selection
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

    # Interval Selection (from 1 minute up to 24 hours)
    interval_options = ["1m", "5m", "15m", "30m", "1h", "1d"]
    selected_interval = st.sidebar.selectbox("Select Timeframe Interval", interval_options)

    # Chart Type
    chart_type = st.sidebar.selectbox("Select Chart Type", ["Candlestick", "Bar", "Line"])

    # Overlays: Moving Average
    add_ma = st.sidebar.checkbox("Add Moving Average", value=True)
    if add_ma:
        ma_period = st.sidebar.number_input("Moving Average Period", min_value=1, value=20, step=1)
        ma_color = st.sidebar.color_picker("Moving Average Color", "#FF5733")
    else:
        ma_period = None

    # Overlays: RSI Indicator
    add_rsi = st.sidebar.checkbox("Add RSI Indicator", value=False)

    # Overlays: Pivot Points
    add_pivot = st.sidebar.checkbox("Add Pivot Points", value=False)
    if add_pivot:
        pivot_method = st.sidebar.selectbox("Select Pivot Method", ["Traditional", "Camarilla"])
    else:
        pivot_method = None

    # Chart Aesthetics
    chart_color = st.sidebar.color_picker("Chart Color", "#1f77b4")

    # Save and share configuration (download as JSON)
    config = {
        "forex_pair": selected_pair,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "interval": selected_interval,
        "chart_type": chart_type,
        "add_ma": add_ma,
        "ma_period": ma_period,
        "ma_color": ma_color if add_ma else None,
        "add_rsi": add_rsi,
        "add_pivot": add_pivot,
        "pivot_method": pivot_method,
        "chart_color": chart_color
    }
    if st.sidebar.button("Save Configuration"):
        config_json = json.dumps(config, indent=4)
        st.sidebar.download_button("Download Config", config_json, file_name="config.json", mime="application/json")

    # ---------------------------
    # Data Fetching
    # ---------------------------
    st.subheader(f"Fetching data for {selected_pair}...")
    with st.spinner("Downloading data..."):
        df = yf.download(selected_pair, start=start_date, end=end_date, interval=selected_interval)
    if df.empty:
        st.error("No data fetched. Please check your ticker, date range, or interval settings.")
        return

    # Reset index so that the date becomes a column
    df.reset_index(inplace=True)
    date_series = get_date_series(df)

    # ---------------------------
    # Indicator Calculations
    # ---------------------------
    if add_ma:
        df['MA'] = df['Close'].rolling(window=ma_period).mean()

    if add_rsi:
        df['RSI'] = calculate_rsi(df['Close'], period=14)

    # Calculate Pivot Points using the last available row (for overlaying horizontal lines)
    pivot_levels = None
    if add_pivot:
        last_row = df.iloc[-1]
        pivot_levels = calculate_pivot_points(last_row['High'], last_row['Low'], last_row['Close'], method=pivot_method)

    # ---------------------------
    # Build the Chart
    # ---------------------------
    fig = go.Figure()

    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=date_series,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Price"
        ))
    elif chart_type == "Bar":
        fig.add_trace(go.Bar(
            x=date_series,
            y=df['Close'],
            name="Close Price",
            marker_color=chart_color
        ))
    elif chart_type == "Line":
        fig.add_trace(go.Scatter(
            x=date_series,
            y=df['Close'],
            mode='lines',
            name="Close Price",
            line=dict(color=chart_color)
        ))

    # Add Moving Average Overlay
    if add_ma:
        fig.add_trace(go.Scatter(
            x=date_series,
            y=df['MA'],
            mode='lines',
            name=f"MA ({ma_period})",
            line=dict(color=ma_color, width=2)
        ))

    # Overlay Pivot Points as horizontal lines
    if add_pivot and pivot_levels is not None:
        for level, value in pivot_levels.items():
            fig.add_hline(
                y=value,
                line_dash="dash",
                annotation_text=level,
                annotation_position="top right"
            )

    fig.update_layout(
        title=f"{selected_pair} Price Chart ({selected_interval} interval)",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # RSI Chart (if selected)
    # ---------------------------
    if add_rsi:
        st.subheader("RSI Indicator")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=date_series,
            y=df['RSI'],
            mode='lines',
            name="RSI",
            line=dict(color="#e74c3c")
        ))
        # Highlight overbought/oversold regions
        fig_rsi.add_shape(
            type="rect",
            xref="paper",
            yref="y",
            x0=0,
            x1=1,
            y0=70,
            y1=100,
            fillcolor="red",
            opacity=0.2,
            layer="below"
        )
        fig_rsi.add_shape(
            type="rect",
            xref="paper",
            yref="y",
            x0=0,
            x1=1,
            y0=0,
            y1=30,
            fillcolor="green",
            opacity=0.2,
            layer="below"
        )
        fig_rsi.update_layout(xaxis_title="Time", yaxis_title="RSI", template="plotly_white")
        st.plotly_chart(fig_rsi, use_container_width=True)

    # ---------------------------
    # Portfolio Tracking Section
    # ---------------------------
    st.subheader("Portfolio Tracking")
    st.write("Upload a CSV or Excel file with your portfolio. (The file should include a column named 'Ticker'.)")
    uploaded_file = st.file_uploader("Upload Portfolio File", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                portfolio_df = pd.read_csv(uploaded_file)
            else:
                portfolio_df = pd.read_excel(uploaded_file)
            st.write("### Portfolio Data")
            st.dataframe(portfolio_df)

            if 'Ticker' in portfolio_df.columns:
                tickers = portfolio_df['Ticker'].unique().tolist()
                st.write("Fetching historical data for tickers:", tickers)
                # Fetch 1 year of data for each ticker (adjust period as needed)
                portfolio_data = yf.download(tickers, period="1y")['Close']
                st.line_chart(portfolio_data)

                # Stock correlation analysis
                st.write("### Correlation Matrix")
                corr = portfolio_data.corr()
                st.dataframe(corr)
            else:
                st.warning("The uploaded file does not contain a 'Ticker' column.")
        except Exception as e:
            st.error(f"Error processing file: {e}")

    # ---------------------------
    # Export Chart Options
    # ---------------------------
    st.subheader("Export Chart")
    export_format = st.selectbox("Select export format", ["PNG", "HTML"])
    if st.button("Export Chart"):
        if export_format == "PNG":
            try:
                # Ensure that the kaleido package is installed for image export.
                img_bytes = fig.to_image(format="png")
                st.download_button("Download PNG", data=img_bytes, file_name="chart.png", mime="image/png")
            except Exception as e:
                st.error(f"Error exporting PNG: {e}")
        elif export_format == "HTML":
            html_bytes = fig.to_html(full_html=True)
            st.download_button("Download HTML", data=html_bytes, file_name="chart.html", mime="text/html")

    st.info("Remember to push this code to GitHub for version control and collaboration!")

if __name__ == "__main__":
    main()
