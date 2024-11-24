import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay
from tqdm import tqdm

# Debug and reset `str` if overwritten
print("Type of str at start:", type(str))
try:
    del str  # Reset to built-in
except NameError:
    pass

# Function to calculate RSI
def calculate_rsi(data, window=14):
    if 'Close' not in data or data['Close'].empty:
        raise ValueError("Invalid data: 'Close' column missing or empty")

    delta = data['Close'].diff().values.flatten()
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
    return data['Close'].rolling(window=window).mean().rename("SMA")

# Preprocess tickers for Yahoo Finance compatibility
def preprocess_ticker(ticker):
    return ticker.replace(".B", "-B").replace(".A", "-A")

# Streamlit app
st.title("Market Analyzer: Stock Screener")
st.write("Analyze multiple stocks and flag potential trading opportunities based on RSI and SMA.")

# Sidebar for settings
st.sidebar.header("Settings")
market_choice = st.sidebar.selectbox("Select Market", ["S&P 500", "NASDAQ"])
rsi_threshold = st.sidebar.slider("RSI Threshold (Oversold)", 10, 50, 30)
sma_window = st.sidebar.slider("SMA Window (days)", 50, 250, 200)

# Define tickers based on market choice
if market_choice == "S&P 500":
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500_table = pd.read_html(sp500_url)[0]
    tickers = sp500_table['Symbol'].tolist()
else:
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]

if not tickers:
    raise ValueError("No tickers found. Please check the market choice.")

st.sidebar.write(f"Analyzing {len(tickers)} stocks...")
print(f"Analyzing tickers: {tickers}")

# Get the last business day
last_business_day = pd.Timestamp.today() - BDay(1)
last_business_day_str = last_business_day.strftime('%Y-%m-%d')
print(f"Last business day: {last_business_day_str}")

# Analyze stocks
flagged_stocks = []
for ticker in tqdm(tickers):
    try:
        # Preprocess ticker
        formatted_ticker = preprocess_ticker(ticker)

        # Fetch stock data
        print(f"Fetching data for {formatted_ticker}...")
        data = yf.download(formatted_ticker, start="2023-01-01", end=last_business_day_str, progress=False)

        if data.empty:
            print(f"No data available for {formatted_ticker}")
            continue

        # Validate and fix column names if necessary
        expected_columns = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]
        if not all(col in expected_columns for col in data.columns):
            print(f"Fixing column names for {formatted_ticker}")
            data.columns = expected_columns

        # Ensure Close column exists
        if "Close" not in data.columns:
            raise ValueError(f"Missing 'Close' column for {formatted_ticker}")

        # Calculate RSI and SMA
        data['RSI'] = calculate_rsi(data)
        data['SMA'] = calculate_sma(data, window=sma_window)

        # Debugging: Print the calculated metrics
        print(f"RSI for {formatted_ticker}: {data['RSI'].dropna().tail()}")
        print(f"SMA for {formatted_ticker}: {data['SMA'].dropna().tail()}")
        print(f"Close Price for {formatted_ticker}: {data['Close'].tail()}")

        # Flag stocks based on criteria
        if data['RSI'].iloc[-1] < rsi_threshold and data['Close'].iloc[-1] > data['SMA'].iloc[-1]:
            flagged_stocks.append({
                "Ticker": ticker,
                "RSI": round(data['RSI'].iloc[-1], 2),
                "Close": round(data['Close'].iloc[-1], 2),
                "SMA": round(data['SMA'].iloc[-1], 2),
            })
        else:
            # Debugging: Show why a stock was not flagged
            print(f"{formatted_ticker} did not match criteria:")
            print(f"RSI: {data['RSI'].iloc[-1]} | Close: {data['Close'].iloc[-1]} | SMA: {data['SMA'].iloc[-1]}")
    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")
        st.write(f"Error analyzing {ticker}: {e}")

# Display results
if flagged_stocks:
    st.subheader("Flagged Stocks")
    results_df = pd.DataFrame(flagged_stocks)
    st.dataframe(results_df)

    csv = results_df.to_csv(index=False)
    st.download_button("Download Results", csv, "flagged_stocks.csv", "text/csv")
else:
    print("No stocks matched the criteria.")
    st.write("No stocks matched the criteria.")

