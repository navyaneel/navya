import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from Stock_prediction.src.fetch_data import fetch_spy_data
from Stock_prediction.src.preprocess import prepare_data
from Stock_prediction.src.lstm_model import build_model, train_model
from Stock_prediction.src.evaluate import compute_metrics

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def main():
    csv_path = os.path.join(DATA_DIR, 'spy_daily.csv')
    if not os.path.exists(csv_path):
        print("Fetching SPY data...")
        df = fetch_spy_data(period='5y', save=True)
    else:
        print("Loading existing data...")
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    print(f"Dataset: {len(df)} rows, {df.columns.tolist()}")

    seq_length = 60
    X_train, y_train, X_test, y_test, scaler, close_scaler, df_proc = \
        prepare_data(df, seq_length=seq_length, train_ratio=0.8)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    model = build_model(
        input_shape=(X_train.shape[1], X_train.shape[2]), units=64
    )
    model.summary()

    history = train_model(
        model, X_train, y_train, X_test, y_test,
        epochs=50, batch_size=32
    )

    y_pred_scaled = model.predict(X_test).flatten()
    y_true_prices = close_scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).flatten()
    y_pred_prices = close_scaler.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).flatten()

    metrics = compute_metrics(y_true_prices, y_pred_prices)
    print("\n=== Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    split_idx = int(len(df_proc) * 0.8)
    test_dates = df_proc.index[split_idx:split_idx + len(y_true_prices)]
    if len(test_dates) != len(y_true_prices):
        test_dates = list(range(len(y_true_prices)))
    results_df = pd.DataFrame({
        'Date': test_dates,
        'Actual': y_true_prices,
        'Predicted': y_pred_prices
    })
    results_df.to_csv(os.path.join(DATA_DIR, 'predictions.csv'), index=False)
    print(f"\nSaved predictions to {DATA_DIR}/predictions.csv")


if __name__ == '__main__':
    main()
