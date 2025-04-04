import joblib
import numpy as np
from data_fetch import get_stock_data
from sklearn.ensemble import RandomForestRegressor

def train_and_save_model(ticker):
    df = get_stock_data(ticker)
    features = df[["Day", "MA10", "MA50", "Volume"]]
    target = df["Close"]
    model = RandomForestRegressor(n_estimators=100)
    model.fit(features, target)
    joblib.dump(model, f"{ticker}_model.pkl")
