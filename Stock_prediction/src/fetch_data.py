import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def fetch_spy_data(period='5y', save=True):
    try:
        import yfinance as yf
        spy = yf.Ticker('SPY')
        df = spy.history(period=period)
        df.index = pd.to_datetime(df.index)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.dropna(inplace=True)
    except Exception as e:
        print(f"yfinance unavailable ({e}), generating synthetic SPY data...")
        df = _generate_synthetic_spy()
    if save:
        os.makedirs(DATA_DIR, exist_ok=True)
        path = os.path.join(DATA_DIR, 'spy_daily.csv')
        df.to_csv(path)
        print(f"Saved {len(df)} rows to {path}")
    return df


def _generate_synthetic_spy():
    np.random.seed(42)
    dates = pd.bdate_range(start='2020-01-02', end='2024-12-31')
    n = len(dates)
    ann_return = 0.12
    ann_vol = 0.18
    daily_mu = ann_return / 252
    daily_sigma = ann_vol / np.sqrt(252)
    log_returns = np.random.normal(daily_mu, daily_sigma, n)
    close = 320.0 * np.exp(np.cumsum(log_returns))
    intraday_range = np.random.uniform(0.003, 0.015, n)
    high = close * (1 + intraday_range / 2)
    low = close * (1 - intraday_range / 2)
    open_noise = np.random.normal(0, 0.003, n)
    open_price = close * (1 + open_noise)
    volume = np.random.randint(50_000_000, 150_000_000, n)
    df = pd.DataFrame({
        'Open': open_price, 'High': high, 'Low': low,
        'Close': close, 'Volume': volume
    }, index=dates)
    df.index.name = 'Date'
    return df


if __name__ == '__main__':
    fetch_spy_data()
