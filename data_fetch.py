import yfinance as yf
import pandas as pd

def get_stock_data(ticker):
    df = yf.download(ticker, period="5y")  # Fetch 5 years of stock data
    if df.empty:
        return None  # Return None if no data is available

    df.reset_index(inplace=True)  # Ensure "Date" is a column

    # Calculate Moving Averages (10-day & 50-day)
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()
    
    # Keep only relevant columns
    df = df[["Date", "Close", "MA10", "MA50", "Volume"]].dropna()

    return df
