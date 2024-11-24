import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay
from tqdm import tqdm

# Function to calculate RSI
def calculate_rsi(data, window=14):
    if 'Close' not in data or data['Close'].empty:
        raise ValueError("Invalid data: 'Close' column missing or empty")

    delta = data['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi, index=data.index)

# Function to calculate SMA
def calculate_sma(data, window=200):
    if 'Close' not in data or data['Close'].empty:
        raise ValueError("Invalid data: 'Close' column missing or empty")
    return data['Close'].rolling(window=window).mean()

# Streamlit App Title
st.title("Market Analyzer: Stock Screener")
st.write("Analyze stocks for potential trading opportunities based on RSI and SMA.")

# Sidebar Settings
st.sidebar.header("Settings")
market_choice = st.sidebar.selectbox("Select Market", ["S&P 500", "NASDAQ"])
rsi_threshold = st.sidebar.slider("RSI Threshold (Oversold)", 10, 50, 30)
sma_window = st.sidebar.slider("SMA Window (days)", 50, 250, 200)

# Load Tickers Based on Market
if market_choice == "S&P 500":
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500_table = pd.read_html(sp500_url)[0]
    tickers = sp500_table['Symbol'].tolist()
else:
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]

if not tickers:
    st.error("No tickers found. Please check your market choice.")
    st.stop()

# Get the last business day
last_business_day = pd.Timestamp.today() - BDay(1)
last_business_day_str = last_business_day.strftime('%Y-%m-%d')

# Initialize analysis
st.sidebar.write(f"Analyzing {len(tickers)} stocks...")
print(f"Analyzing tickers: {tickers}")
flagged_stocks = []

# Analyze each ticker
for ticker in tqdm(tickers):
    try:
        # Fetch stock data
        data = yf.download(ticker, start="2023-01-01", end=last_business_day_str, progress=False)

        if data.empty:
            print(f"No data available for {ticker}")
            continue

        # Ensure Close column exists
        if "Close" not in data.columns:
            raise ValueError(f"Missing 'Close' column for {ticker}")

        # Calculate RSI and SMA
        data['RSI'] = calculate_rsi(data)
        data['SMA'] = calculate_sma(data, window=sma_window)

        # Ensure RSI and SMA calculations are valid
        if data['RSI'].isna().all():
            print(f"Insufficient data for RSI calculation for {ticker}")
            continue

        # Flag stocks based on criteria
        latest_rsi = data['RSI'].iloc[-1]
        latest_close = data['Close'].iloc[-1]
        latest_sma = data['SMA'].iloc[-1]

        if latest_rsi < rsi_threshold and latest_close > latest_sma:
            flagged_stocks.append({
                "Ticker": ticker,
                "RSI": round(latest_rsi, 2),
                "Close": round(latest_close, 2),
                "SMA": round(latest_sma, 2),
            })
        else:
            print(f"{ticker} did not match criteria. RSI: {latest_rsi}, Close: {latest_close}, SMA: {latest_sma}")
    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")

# Display flagged stocks
if flagged_stocks:
    st.subheader("Flagged Stocks")
    results_df = pd.DataFrame(flagged_stocks)
    st.dataframe(results_df)
    csv = results_df.to_csv(index=False)
    st.download_button("Download Results", csv, "flagged_stocks.csv", "text/csv")
else:
    st.warning("No stocks matched the criteria.")
    st.write("Try adjusting the RSI threshold or SMA window to broaden the criteria.")

