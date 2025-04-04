import streamlit as st
import pandas as pd
from datetime import date
from data_fetch import get_nasdaq_tickers, get_stock_data
from model import predict_price

st.set_page_config(page_title="Stock Market Price Prediction", layout="centered")
st.title("Stock Market Price Prediction")

tickers = get_nasdaq_tickers()
ticker = st.selectbox("Select Stock Ticker", tickers)

future_date = st.date_input("Select a future date:", min_value=date.today())

if st.button("Calculate"):
    predicted_price = predict_price(ticker, future_date)
    if predicted_price:
        st.subheader(f"Predicted price for {ticker} on {future_date}: ${predicted_price}")
    else:
        st.error("Prediction failed. Check model or data.")

    df = get_stock_data(ticker)
    st.line_chart(df["Close"])
