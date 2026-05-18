# 📈 SPY Stock Price Prediction (LSTM)

Predict next-day SPY closing price using a Long Short-Term Memory (LSTM) neural network trained on 5 years of daily data.

## Project Structure

```
Stock_Prediction/
├── data/                   # SPY daily OHLCV (5 years)
├── src/
│   ├── fetch_data.py       # Download SPY data via yfinance
│   ├── preprocess.py       # Scaling, feature engineering, sequences
│   ├── lstm_model.py       # Keras LSTM model definition & training
│   └── evaluate.py         # RMSE, MAE, MAPE, direction accuracy, plots
├── app.py                  # Streamlit interactive dashboard
├── run_pipeline.py         # CLI training pipeline
├── notebook.ipynb          # Jupyter notebook (predictions vs actual)
└── README.md
```

## Model Architecture

| Layer       | Config                    |
|-------------|---------------------------|
| LSTM        | 64 units, return_sequences |
| Dropout     | 0.2                       |
| LSTM        | 64 units                  |
| Dropout     | 0.2                       |
| Dense       | 32 units, ReLU            |
| Dense       | 1 unit (output)           |

**Features**: Close, Volume, MA(5), MA(20), RSI(14), Daily Returns
**Sequence Length**: 60 days
**Optimizer**: Adam | **Loss**: MSE | **Early Stopping**: patience=5

## Performance

| Metric              | Target   |
|---------------------|----------|
| Direction Accuracy   | ≥ 55%   |
| MAPE                | < 5%     |

## Quick Start

```bash
# Set Python path
setenv PYTHONPATH /home/nsubrama/apps

# 1. Fetch data
cd Stock_Prediction
python src/fetch_data.py

# 2. Run full pipeline (CLI)
python run_pipeline.py

# 3. Launch Streamlit app
streamlit run app.py
```

## Streamlit App Features

- **Data Explorer**: Candlestick charts, moving averages, dataset preview
- **Train & Predict**: Configurable hyperparameters, live training progress
- **Results**: Actual vs predicted overlay, residual distribution, zoom slider
- **Metrics**: RMSE, MAE, MAPE, direction accuracy dashboard

## Tech Stack

- **Model**: TensorFlow 2.11 / Keras LSTM
- **Data**: yfinance (Yahoo Finance API)
- **Visualization**: Plotly, Streamlit
- **Preprocessing**: scikit-learn MinMaxScaler
