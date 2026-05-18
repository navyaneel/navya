import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go


def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    direction_true = np.diff(y_true) > 0
    direction_pred = np.diff(y_pred) > 0
    direction_acc = np.mean(direction_true == direction_pred) * 100
    return {
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
        'MAPE': round(mape, 2),
        'Direction_Accuracy': round(direction_acc, 2)
    }


def plot_predictions(dates, y_true, y_pred, title='SPY Price Prediction'):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=y_true, mode='lines', name='Actual',
        line=dict(color='#2196F3', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=y_pred, mode='lines', name='Predicted',
        line=dict(color='#FF5722', width=2, dash='dot')
    ))
    fig.update_layout(
        title=title, xaxis_title='Date', yaxis_title='Price ($)',
        template='plotly_dark', height=500,
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01)
    )
    return fig


def plot_training_history(history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=history.history['loss'], mode='lines', name='Train Loss',
        line=dict(color='#4CAF50')
    ))
    fig.add_trace(go.Scatter(
        y=history.history['val_loss'], mode='lines', name='Val Loss',
        line=dict(color='#FF9800')
    ))
    fig.update_layout(
        title='Training & Validation Loss',
        xaxis_title='Epoch', yaxis_title='MSE Loss',
        template='plotly_dark', height=400
    )
    return fig


def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=residuals, nbinsx=50, name='Residuals',
        marker_color='#9C27B0', opacity=0.7
    ))
    fig.update_layout(
        title='Prediction Residuals Distribution',
        xaxis_title='Residual ($)', yaxis_title='Count',
        template='plotly_dark', height=400
    )
    return fig
