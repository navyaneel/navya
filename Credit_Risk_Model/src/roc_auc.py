import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)

def compute_metrics(y_true, y_pred, y_proba):
    return {
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
        'f1': round(f1_score(y_true, y_pred, zero_division=0), 4),
        'auc_roc': round(roc_auc_score(y_true, y_proba), 4)
    }

def plot_roc_curve(y_true, y_proba_lr, y_proba_xgb):
    fpr_lr, tpr_lr, _ = roc_curve(y_true, y_proba_lr)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_true, y_proba_xgb)
    auc_lr = round(auc(fpr_lr, tpr_lr), 4)
    auc_xgb = round(auc(fpr_xgb, tpr_xgb), 4)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr_lr, y=tpr_lr, mode='lines',
        name=f'Logistic Regression (AUC={auc_lr})'
    ))
    fig.add_trace(go.Scatter(
        x=fpr_xgb, y=tpr_xgb, mode='lines',
        name=f'XGBoost (AUC={auc_xgb})'
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines',
        name='Baseline', line=dict(dash='dash', color='gray')
    ))
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        template='plotly_dark',
        height=500
    )
    return fig

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    labels = ['Non-Default', 'Default']
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        showscale=True
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Predicted',
        yaxis_title='Actual',
        template='plotly_dark'
    )
    return fig

def plot_precision_recall_curve(y_true, y_proba_lr, y_proba_xgb):
    prec_lr, rec_lr, _ = precision_recall_curve(y_true, y_proba_lr)
    prec_xgb, rec_xgb, _ = precision_recall_curve(y_true, y_proba_xgb)
    auc_lr = round(auc(rec_lr, prec_lr), 4)
    auc_xgb = round(auc(rec_xgb, prec_xgb), 4)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rec_lr, y=prec_lr, mode='lines',
        name=f'Logistic Regression (AUC={auc_lr})'
    ))
    fig.add_trace(go.Scatter(
        x=rec_xgb, y=prec_xgb, mode='lines',
        name=f'XGBoost (AUC={auc_xgb})'
    ))
    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        template='plotly_dark'
    )
    return fig

def plot_feature_importance(importance_df, title='Feature Importance', top_n=15):
    df = importance_df.head(top_n).sort_values('importance', ascending=True)
    fig = go.Figure(go.Bar(
        x=df['importance'],
        y=df['feature'],
        orientation='h'
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Importance',
        yaxis_title='Feature',
        template='plotly_dark'
    )
    return fig

def classification_report_df(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return pd.DataFrame(report).transpose()
