import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from data_fetch import get_stock_data

def train_model(ticker="AAPL"):
    print(f"📌 Fetching stock data for {ticker}...")
    df = get_stock_data(ticker)
    
    if df is None or df.empty:
        print("⚠️ No data available for the given ticker.")
        return
    
    print("✅ Data fetched! Preprocessing...")
    
    # Features and target variable
    df["Day"] = np.arange(len(df))  # Adding a numerical day count
    X = df[["Day", "MA10", "MA50", "Volume"]]  # Feature set
    y = df["Close"]  # Target (closing price)
    
    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"📊 Training model on {len(X_train)} samples...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the trained model
    joblib.dump(model, "assets/stock_model.pkl")
    print("🎯 Model trained and saved successfully!")

    # Scaling the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Save the scaler for future predictions
    joblib.dump(scaler, "assets/scaler.pkl")
    print("✅ Scaler saved successfully!")

def predict_price(ticker, future_date):
    print(f"🔮 Predicting {ticker} price for {future_date}...")
    df = get_stock_data(ticker)
    
    if df is None or df.empty:
        return "⚠️ No data available"
    
    # Load trained model and scaler
    model = joblib.load("assets/stock_model.pkl")
    scaler = joblib.load("assets/scaler.pkl")
    
    # Calculate future day index
    last_day = df["Day"].max()
    future_days = last_day + (pd.to_datetime(future_date) - df["Date"].max()).days
    
    # Prepare input for prediction
    last_ma10 = df["MA10"].iloc[-1]
    last_ma50 = df["MA50"].iloc[-1]
    last_volume = df["Volume"].iloc[-1]
    
    X_future = np.array([[future_days, last_ma10, last_ma50, last_volume]])
    
    # Scale input features
    X_future_scaled = scaler.transform(X_future)
    
    # Predict the future price
    predicted_price = model.predict(X_future_scaled)[0]
    
    return f"📌 Predicted price for {ticker} on {future_date} is: ${predicted_price:.2f}"
