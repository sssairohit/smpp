import joblib
import numpy as np
import os
from data_fetch import get_stock_data
from train_model import train_and_save_model

def predict_price(ticker, future_date):
    df = get_stock_data(ticker)
    if df.empty:
        return None
    future_day_index = len(df) + (future_date - df.index[-1].date()).days
    last_row = df.iloc[-1]
    last_ma10 = float(last_row["MA10"])
    last_ma50 = float(last_row["MA50"])
    last_volume = float(last_row["Volume"])
    X_future = np.array([[future_day_index, last_ma10, last_ma50, last_volume]])
    model_path = f"{ticker}_model.pkl"
    if not os.path.exists(model_path):
        train_and_save_model(ticker)
    model = joblib.load(model_path)
    prediction = model.predict(X_future)
    return round(prediction[0], 2)
