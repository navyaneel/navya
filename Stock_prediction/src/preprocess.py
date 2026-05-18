import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def add_features(df):
    df = df.copy()
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['RSI'] = compute_rsi(df['Close'], 14)
    df['Returns'] = df['Close'].pct_change()
    df.dropna(inplace=True)
    return df


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def prepare_data(df, seq_length=60, train_ratio=0.8):
    df = add_features(df)
    feature_cols = ['Close', 'Volume', 'MA_5', 'MA_20', 'RSI', 'Returns']
    data = df[feature_cols].values

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    split = int(len(scaled) * train_ratio)
    train_data = scaled[:split]
    test_data = scaled[split - seq_length:]

    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)

    close_scaler = MinMaxScaler()
    close_scaler.fit(df[['Close']].values[:split])

    return X_train, y_train, X_test, y_test, scaler, close_scaler, df
