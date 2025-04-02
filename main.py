import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
from data_fetch import get_stock_data

# Load trained model & scaler
model = joblib.load("assets/stock_model.pkl")
scaler = joblib.load("assets/scaler.pkl")

st.title("📈 Stock Market Price Prediction")

# Stock Ticker Input
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL")

# Fetch stock data
df = get_stock_data(ticker)

if df is not None and not df.empty:
    # Display Stock Price History
    st.subheader(f"📊 {ticker} Stock Price History")
    st.line_chart(df["Close"])

    # Future Date Input
    future_date = st.date_input("Select a future date:", datetime.date.today())

    if st.button("Calculate Prediction"):
        # Get latest values
        last_row = df.iloc[-1]

        # Ensure the values are scalars, not sequences
        last_ma10 = last_row["MA10"] if isinstance(last_row["MA10"], (int, float)) else float(last_row["MA10"])
        last_ma50 = last_row["MA50"] if isinstance(last_row["MA50"], (int, float)) else float(last_row["MA50"])
        last_volume = last_row["Volume"] if isinstance(last_row["Volume"], (int, float)) else float(last_row["Volume"])

        # Prepare the input for prediction (30 days ahead)
        future_input = np.array([[30, last_ma10, last_ma50, last_volume]])  # 30 days ahead

        # Prepare input with latest values and scale
        future_scaled = scaler.transform(future_input)

        # Predict the future price
        predicted_price = model.predict(future_scaled)[0]

        # Display the prediction
        st.success(f"📌 Predicted price for **{ticker}** on **{future_date}** is: **${predicted_price:.2f}**")

else:
    st.error("⚠️ Failed to fetch stock data. Check the ticker.")
