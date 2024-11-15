import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate SMA
def calculate_sma(data, window=200):
    return data['Close'].rolling(window=window).mean()

# Streamlit app
st.title("Market Analyzer: Stock Screener")
st.write("Analyze multiple stocks and flag potential trading opportunities based on RSI and SMA.")

# Sidebar for settings
st.sidebar.header("Settings")
market_choice = st.sidebar.selectbox("Select Market", ["S&P 500", "NASDAQ"])
rsi_threshold = st.sidebar.slider("RSI Threshold (Oversold)", 10, 50, 30)
sma_window = st.sidebar.slider("SMA Window (days)", 50, 250, 200)

# Load tickers for the chosen market
if market_choice == "S&P 500":
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500_table = pd.read_html(sp500_url)[0]
    tickers = sp500_table['Symbol'].tolist()
else:
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]  # Sample NASDAQ tickers

st.sidebar.write(f"Analyzing {len(tickers)} stocks...")

# Analyze stocks
flagged_stocks = []
for ticker in tickers:
    try:
        # Fetch stock data
        data = yf.download(ticker, period="6mo", progress=False)
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
        st.write(f"Error analyzing {ticker}: {e}")

# Display results
if flagged_stocks:
    st.subheader("Flagged Stocks")
    results_df = pd.DataFrame(flagged_stocks)
    st.dataframe(results_df)

    # Download option
    csv = results_df.to_csv(index=False)
    st.download_button("Download Results", csv, "flagged_stocks.csv", "text/csv")
else:
    st.write("No stocks matched the criteria.")
