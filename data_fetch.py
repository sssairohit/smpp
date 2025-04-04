import pandas as pd
import yfinance as yf
import requests
from io import StringIO

def get_stock_data(ticker):
    df = yf.download(ticker, period="1y")
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()
    df = df.dropna()
    df["Day"] = range(len(df))
    return df

def get_nasdaq_tickers():
    url = "https://datahub.io/core/nasdaq-listings/r/nasdaq-listed.csv"
    response = requests.get(url)
    df = pd.read_csv(StringIO(response.text))
    return sorted(df["Symbol"].dropna().tolist())
