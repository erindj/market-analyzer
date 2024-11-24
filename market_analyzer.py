import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay

# Ensure str is not overwritten
print("Type of str at start:", type(str))
try:
    del str
except NameError:
    pass

# Function to calculate RSI
def calculate_rsi(data, window=14):
    print(f"Calculating RSI for data:\n{data.head()}")
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
    print(f"Calculating SMA for data:\n{data.head()}")
    if 'Close' not in data or data['Close'].empty:
        raise ValueError("Invalid data: 'Close' column missing or empty")
    return data['Close'].rolling(window=window).mean().rename("SMA")

# Preprocess tickers for Yahoo Finance compatibility
def preprocess_ticker(ticker):
    print(f"Preprocessing ticker: {ticker}")
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
    print("Fetching S&P 500 tickers...")
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500_table = pd.read_html(sp500_url)[0]
    tickers = sp500_table['Symbol'].tolist()
else:
    print("Using predefined NASDAQ tickers...")
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]

# Verify tickers
if not tickers:
    raise ValueError("Tickers list is empty. Please check the market choice or data source.")
print(f"Tickers to analyze: {tickers}")

st.sidebar.write(f"Analyzing {len(tickers)} stocks...")

# Get the last business day
last_business_day = pd.Timestamp.today() - BDay(1)
last_business_day_str = last_business_day.strftime('%Y-%m-%d')
print(f"Last business day: {last_business_day_str}")

# Analyze stocks
flagged_stocks = []
for ticker in tickers:
    try:
        # Preprocess ticker
        formatted_ticker = preprocess_ticker(ticker)

        # Fetch stock data
        print(f"Fetching data for {formatted_ticker}...")
        data = yf.download(formatted_ticker, start="2023-01-01", end=last_business_day_str, progress=False)

        # Validate data
        if data.empty:
            print(f"No data available for {formatted_ticker}")
            continue

        # Fix column names if necessary
        print(f"Columns for {formatted_ticker}: {data.columns}")
        if not all(col in data.columns for col in ["Close", "Adj Close"]):
            print(f"Fixing column names for {formatted_ticker}")
            data.columns = [col.split()[0] for col in data.columns]

        # Ensure Close column exists
        if "Close" not in data.columns:
            raise ValueError(f"Missing 'Close' column for {formatted_ticker}")

        # Calculate RSI and SMA
        data['RSI'] = calculate_rsi(data)
        data['SMA'] = calculate_sma(data, window=sma_window)

        # Flag stocks based on criteria
        if data['RSI'].iloc[-1] < rsi_threshold and data['Close'].iloc[-1] > data['SMA'].iloc[-1]:
            flagged_stocks.append({
                "Ticker": ticker,
                "RSI": round(data['RSI'].iloc[-1], 2),
                "Close": round(data['Close'].iloc[-1], 2),
                "SMA": round(data['SMA'].iloc[-1], 2),
            })
    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")
        st.write(f"Error analyzing {ticker}: {e}")

# Display results
if flagged_stocks:
    st.subheader("Flagged Stocks")
    results_df = pd.DataFrame(flagged_stocks)
    print(f"Flagged stocks:\n{results_df}")
    st.dataframe(results_df)

    # Download option
    csv = results_df.to_csv(index=False)
    st.download_button("Download Results", csv, "flagged_stocks.csv", "text/csv")
else:
    print("No stocks matched the criteria.")
    st.write("No stocks matched the criteria.")
