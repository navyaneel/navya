import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(__file__))
from Stock_prediction.src.fetch_data import fetch_spy_data
from Stock_prediction.src.preprocess import prepare_data, add_features
from Stock_prediction.src.lstm_model import build_model, train_model
from Stock_prediction.src.evaluate import compute_metrics, plot_predictions, plot_training_history, plot_residuals

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

st.set_page_config(page_title="SPY Stock Predictor", page_icon="📈", layout="wide")

st.title("📈 SPY Stock Price Prediction")
st.markdown("**LSTM Neural Network** — Next-day closing price prediction")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Configuration")
    seq_length = st.slider("Sequence Length (days)", 20, 120, 60, 10)
    epochs = st.slider("Training Epochs", 10, 100, 50, 5)
    batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
    lstm_units = st.selectbox("LSTM Units", [32, 64, 128], index=1)
    train_ratio = st.slider("Train/Test Split", 0.6, 0.9, 0.8, 0.05)
    st.markdown("---")
    fetch_btn = st.button("📥 Fetch Fresh Data")
    train_btn = st.button("🚀 Train Model")

csv_path = os.path.join(DATA_DIR, 'spy_daily.csv')

if fetch_btn:
    with st.spinner("Downloading SPY data (5 years)..."):
        df = fetch_spy_data(period='5y', save=True)
    st.success(f"Downloaded {len(df)} trading days")

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Explorer", "🧠 Train & Predict", "📉 Results", "📋 Metrics"
    ])

    with tab1:
        st.subheader("Raw Price Data")
        col1, col2, col3 = st.columns(3)
        col1.metric("Latest Close", f"${df['Close'].iloc[-1]:.2f}")
        col2.metric("5Y High", f"${df['High'].max():.2f}")
        col3.metric("5Y Low", f"${df['Low'].min():.2f}")

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'], name='SPY'
        ))
        fig.update_layout(
            title='SPY Daily OHLC', template='plotly_dark',
            height=500, xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)

        df_feat = add_features(df)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_feat.index, y=df_feat['Close'], name='Close'))
        fig2.add_trace(go.Scatter(x=df_feat.index, y=df_feat['MA_5'], name='MA 5'))
        fig2.add_trace(go.Scatter(x=df_feat.index, y=df_feat['MA_20'], name='MA 20'))
        fig2.update_layout(
            title='Close with Moving Averages', template='plotly_dark', height=400
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Dataset Preview")
        st.dataframe(df.tail(20), use_container_width=True)

    with tab2:
        if train_btn:
            with st.spinner("Preparing data..."):
                X_train, y_train, X_test, y_test, scaler, close_scaler, df_proc = \
                    prepare_data(df, seq_length=seq_length, train_ratio=train_ratio)

            st.info(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples | Features: {X_train.shape[2]}")

            with st.spinner("Training LSTM model..."):
                progress = st.progress(0)
                model = build_model(
                    input_shape=(X_train.shape[1], X_train.shape[2]),
                    units=lstm_units
                )
                st.text(f"Model parameters: {model.count_params():,}")
                history = train_model(
                    model, X_train, y_train, X_test, y_test,
                    epochs=epochs, batch_size=batch_size
                )
                progress.progress(100)

            st.success("Training complete!")
            fig_hist = plot_training_history(history)
            st.plotly_chart(fig_hist, use_container_width=True)

            with st.spinner("Generating predictions..."):
                y_pred_scaled = model.predict(X_test).flatten()
                y_true_prices = close_scaler.inverse_transform(
                    y_test.reshape(-1, 1)
                ).flatten()
                y_pred_prices = close_scaler.inverse_transform(
                    y_pred_scaled.reshape(-1, 1)
                ).flatten()

            split_idx = int(len(df_proc) * train_ratio)
            test_dates = df_proc.index[split_idx:split_idx + len(y_true_prices)]

            st.session_state['y_true'] = y_true_prices
            st.session_state['y_pred'] = y_pred_prices
            st.session_state['dates'] = test_dates
            st.session_state['history'] = history
            st.session_state['trained'] = True

            os.makedirs(DATA_DIR, exist_ok=True)
            results_df = pd.DataFrame({
                'Date': test_dates,
                'Actual': y_true_prices,
                'Predicted': y_pred_prices
            })
            results_df.to_csv(os.path.join(DATA_DIR, 'predictions.csv'), index=False)

            metrics = compute_metrics(y_true_prices, y_pred_prices)
            with open(os.path.join(DATA_DIR, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)

            st.session_state['metrics'] = metrics
        else:
            st.info("Configure parameters in the sidebar and click **🚀 Train Model**")

    with tab3:
        if st.session_state.get('trained'):
            y_true = st.session_state['y_true']
            y_pred = st.session_state['y_pred']
            dates = st.session_state['dates']

            fig_pred = plot_predictions(dates, y_true, y_pred)
            st.plotly_chart(fig_pred, use_container_width=True)

            fig_res = plot_residuals(y_true, y_pred)
            st.plotly_chart(fig_res, use_container_width=True)

            last_n = st.slider("Zoom: last N days", 10, len(y_true), 60)
            fig_zoom = plot_predictions(
                dates[-last_n:], y_true[-last_n:], y_pred[-last_n:],
                title=f'Last {last_n} Days — Actual vs Predicted'
            )
            st.plotly_chart(fig_zoom, use_container_width=True)
        else:
            st.info("Train the model first to see results")

    with tab4:
        if st.session_state.get('metrics'):
            m = st.session_state['metrics']
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("RMSE", f"${m['RMSE']:.2f}")
            col2.metric("MAE", f"${m['MAE']:.2f}")
            col3.metric("MAPE", f"{m['MAPE']:.1f}%")
            col4.metric("Direction Accuracy", f"{m['Direction_Accuracy']:.1f}%")

            st.markdown("---")
            st.subheader("Prediction Samples")
            pred_path = os.path.join(DATA_DIR, 'predictions.csv')
            if os.path.exists(pred_path):
                pred_df = pd.read_csv(pred_path)
                pred_df['Error'] = pred_df['Actual'] - pred_df['Predicted']
                pred_df['Error_%'] = (pred_df['Error'] / pred_df['Actual'] * 100).round(2)
                st.dataframe(pred_df.tail(30), use_container_width=True)
        else:
            st.info("Train the model first to see metrics")
else:
    st.warning("No data found. Click **📥 Fetch Fresh Data** in the sidebar.")

st.markdown("---")
st.caption("Built with Streamlit • LSTM via TensorFlow/Keras • Data from Yahoo Finance")
