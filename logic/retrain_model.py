import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import keras_tuner as kt
from kerastuner.tuners import RandomSearch
import tensorflow as tf

# Fix seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def retrain_and_predict(ticker: str) -> float:
    # ====== 1. Download Historical Data ======
    df = yf.download(ticker, start="2018-01-01", end="2025-04-23")
    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'.")

    # ====== 2. Feature Engineering ======
    df_feat = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    for w in (10, 20, 50):
        df_feat[f"SMA_{w}"] = df_feat['Close'].rolling(w).mean()

    # RSI
    delta = df_feat['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df_feat['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    fast, slow, sig = 12, 26, 9
    ema_fast = df_feat['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df_feat['Close'].ewm(span=slow, adjust=False).mean()
    df_feat['MACD'] = ema_fast - ema_slow
    df_feat['MACD_SIGNAL'] = df_feat['MACD'].ewm(span=sig, adjust=False).mean()

    df_feat.dropna(inplace=True)

    # ====== 3. Scaling & Sequence Creation ======
    data = df_feat.values
    scaler = MinMaxScaler((0, 1))
    scaled = scaler.fit_transform(data)
    target_idx = df_feat.columns.get_loc('Close')
    close_min = scaler.data_min_[target_idx]
    close_max = scaler.data_max_[target_idx]

    SEQ_LEN = 30
    X, y = [], []
    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i-SEQ_LEN:i, :])
        y.append(scaled[i, target_idx])

    X, y = np.array(X), np.array(y)

    # ====== 4. Train/Val/Test Split ======
    n = len(X)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

    # ====== 5. Hyperparameter Tuning ======
    def build_model(hp):
        model = Sequential()
        units = hp.Int('units', 32, 128, step=16)
        model.add(LSTM(units, return_sequences=False,
                       input_shape=(SEQ_LEN, X.shape[2]),
                       dropout=hp.Float('dropout', 0.1, 0.5, step=0.1),
                       recurrent_dropout=hp.Float('rec_dropout', 0.1, 0.5, step=0.1)))
        model.add(Dropout(hp.Float('post_dropout', 0.1, 0.5, step=0.1)))

        opt_name = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'])
        lr = hp.Float('lr', 1e-4, 1e-2, sampling='log')
        if opt_name == 'adam':
            opt = Adam(learning_rate=lr)
        elif opt_name == 'rmsprop':
            opt = RMSprop(learning_rate=lr)
        else:
            opt = SGD(learning_rate=lr)

        model.add(Dense(1, activation='linear'))
        model.compile(optimizer=opt, loss='mse')
        return model

    tuner = RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=5,
        executions_per_trial=1,
        directory='tuner_dir',
        project_name=f'{ticker}_lstm'
    )

    tuner.search(X_train, y_train,
                 validation_data=(X_val, y_val),
                 epochs=10,
                 callbacks=[EarlyStopping(monitor='val_loss', patience=3)],
                 verbose=0)

    best_model = tuner.get_best_models(num_models=1)[0]

    # ====== 6. Final Training on Full Train + Val Set ======
    X_final = np.concatenate([X_train, X_val])
    y_final = np.concatenate([y_train, y_val])

    best_model.fit(X_final, y_final,
                   epochs=15,
                   batch_size=32,
                   validation_split=0.1,
                   callbacks=[EarlyStopping(monitor='val_loss', patience=3)],
                   verbose=0)

    # ====== 7. Predict the Next Day ======
    recent_seq = scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, X.shape[2])
    next_scaled_close = best_model.predict(recent_seq)[0][0]
    next_unscaled_close = next_scaled_close * (close_max - close_min) + close_min

    return float(next_unscaled_close)


def retrain_and_predict_multi(ticker: str, n_days: int = 5) -> list:
    """
    Retrain the LSTM model on the given stock and predict the next N closing prices recursively.

    Parameters:
        ticker (str): Stock ticker symbol (e.g., "AAPL")
        n_days (int): Number of future days to predict

    Returns:
        list: List of predicted prices (1 per day)
    """
    # === Reuse Steps 1 to 6 from retrain_and_predict ===
    df = yf.download(ticker, start="2018-01-01", end="2025-04-23")
    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'.")

    df_feat = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    for w in (10, 20, 50):
        df_feat[f"SMA_{w}"] = df_feat['Close'].rolling(w).mean()

    delta = df_feat['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df_feat['RSI'] = 100 - (100 / (1 + rs))

    fast, slow, sig = 12, 26, 9
    ema_fast = df_feat['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df_feat['Close'].ewm(span=slow, adjust=False).mean()
    df_feat['MACD'] = ema_fast - ema_slow
    df_feat['MACD_SIGNAL'] = df_feat['MACD'].ewm(span=sig, adjust=False).mean()

    df_feat.dropna(inplace=True)
    data = df_feat.values
    scaler = MinMaxScaler((0, 1))
    scaled = scaler.fit_transform(data)
    target_idx = df_feat.columns.get_loc('Close')
    close_min = scaler.data_min_[target_idx]
    close_max = scaler.data_max_[target_idx]

    SEQ_LEN = 30
    X, y = [], []
    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i-SEQ_LEN:i, :])
        y.append(scaled[i, target_idx])

    X, y = np.array(X), np.array(y)
    n = len(X)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val:]

    def build_model(hp):
        model = Sequential()
        units = hp.Int('units', 32, 128, step=16)
        model.add(LSTM(units, return_sequences=False,
                       input_shape=(SEQ_LEN, X.shape[2]),
                       dropout=hp.Float('dropout', 0.1, 0.5, step=0.1),
                       recurrent_dropout=hp.Float('rec_dropout', 0.1, 0.5, step=0.1)))
        model.add(Dropout(hp.Float('post_dropout', 0.1, 0.5, step=0.1)))

        opt_name = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'])
        lr = hp.Float('lr', 1e-4, 1e-2, sampling='log')
        if opt_name == 'adam':
            opt = Adam(learning_rate=lr)
        elif opt_name == 'rmsprop':
            opt = RMSprop(learning_rate=lr)
        else:
            opt = SGD(learning_rate=lr)

        model.add(Dense(1, activation='linear'))
        model.compile(optimizer=opt, loss='mse')
        return model

    tuner = RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=5,
        executions_per_trial=1,
        directory='tuner_dir',
        project_name=f'{ticker}_lstm_multi'
    )

    tuner.search(X_train, y_train,
                 validation_data=(X_val, y_val),
                 epochs=10,
                 callbacks=[EarlyStopping(monitor='val_loss', patience=3)],
                 verbose=0)

    best_model = tuner.get_best_models(num_models=1)[0]

    # Final Training
    X_final = np.concatenate([X_train, X_val])
    y_final = np.concatenate([y_train, y_val])
    best_model.fit(X_final, y_final,
                   epochs=15,
                   batch_size=32,
                   validation_split=0.1,
                   callbacks=[EarlyStopping(monitor='val_loss', patience=3)],
                   verbose=0)

    # Predict multiple days recursively
    predictions = []
    recent_seq = scaled[-SEQ_LEN:].copy()

    for _ in range(n_days):
        input_seq = recent_seq.reshape(1, SEQ_LEN, recent_seq.shape[1])
        next_scaled = best_model.predict(input_seq)[0][0]
        next_unscaled = next_scaled * (close_max - close_min) + close_min
        predictions.append(float(next_unscaled))

        # Create new feature vector for next step
        new_row = recent_seq[-1].copy()
        new_row[target_idx] = next_scaled
        recent_seq = np.vstack([recent_seq[1:], new_row])

    return predictions

