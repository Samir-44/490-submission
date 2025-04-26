import yfinance as yf
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os

# ====== Load Pretrained Model and Scaler (only once) ======
MODEL_PATH = os.path.join("models", "spy_lstm_model.h5")
SCALER_PATH = os.path.join("models", "scaler.save")

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ====== Predict Function ======
def predict_with_spy_model(ticker: str, window_size: int = 60) -> float:
    """
    Predict the next closing price for the given ticker using a pretrained SPY-based model.

    Parameters:
        ticker (str): Stock ticker symbol (e.g., "AAPL", "TSLA")
        window_size (int): Number of days to use for prediction (default: 60)

    Returns:
        float: Predicted next closing price in USD
    """
    # Download the last 90 days of closing data (in case of weekends/holidays)
    df = yf.download(ticker, period="90d")['Close'].dropna()

    if len(df) < window_size:
        raise ValueError(f"Not enough data to predict for {ticker}. Need at least {window_size} days.")

    # Get the last window_size days
    last_60 = df[-window_size:].values.reshape(-1, 1)

    # Scale the data
    scaled = scaler.transform(last_60)

    # Reshape to (1, 60, 1) for LSTM
    input_seq = scaled.reshape((1, window_size, 1))

    # Predict
    prediction_scaled = model.predict(input_seq)
    prediction = scaler.inverse_transform(prediction_scaled)[0][0]

    return float(prediction)

def predict_next_n_days(ticker: str, n_days: int = 5, window_size: int = 60) -> list:
    """
    Predict the next N closing prices using recursive forecasting with the SPY-based model.

    Parameters:
        ticker (str): Stock ticker symbol (e.g., "AAPL")
        n_days (int): Number of days to predict
        window_size (int): Number of past days to use as input

    Returns:
        list: List of predicted prices (1 per day)
    """
    # Download recent data
    df = yf.download(ticker, period="90d")['Close'].dropna()

    if len(df) < window_size:
        raise ValueError(f"Not enough data to predict for {ticker}. Need at least {window_size} days.")

    current_input = df[-window_size:].values.reshape(-1, 1)
    predictions = []

    for _ in range(n_days):
        scaled_input = scaler.transform(current_input)
        input_seq = scaled_input.reshape((1, window_size, 1))
        next_scaled = model.predict(input_seq)
        next_price = scaler.inverse_transform(next_scaled)[0][0]
        predictions.append(float(next_price))

        # Slide window: remove first, append prediction
        current_input = np.append(current_input[1:], [[next_price]], axis=0)

    return predictions

