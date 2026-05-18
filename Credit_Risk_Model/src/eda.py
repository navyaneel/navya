import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_data():
    return pd.read_csv(os.path.join(DATA_DIR, 'lending_club.csv'))

def summary_stats(df):
    return {
        'describe': df.describe(include='all'),
        'missing': df.isnull().sum(),
        'dtypes': df.dtypes
    }

def plot_target_distribution(df):
    counts = df['default'].value_counts().reset_index()
    counts.columns = ['default', 'count']
    counts['label'] = counts['default'].map({0: 'Non-Default', 1: 'Default'})
    fig = px.bar(counts, x='label', y='count', color='label',
                 title='Target Distribution: Default vs Non-Default',
                 template='plotly_dark',
                 labels={'label': 'Status', 'count': 'Count'})
    return fig

def plot_numeric_distributions(df):
    cols = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'revol_util']
    fig = go.Figure()
    for col in cols:
        if col in df.columns:
            fig.add_trace(go.Histogram(x=df[col].dropna(), name=col, opacity=0.75))
    fig.update_layout(
        title='Numeric Feature Distributions',
        barmode='overlay',
        template='plotly_dark',
        xaxis_title='Value',
        yaxis_title='Count'
    )
    return fig

def plot_correlation_matrix(df):
    numeric_df = df.select_dtypes(include='number')
    corr = numeric_df.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale='RdBu',
        zmid=0,
        text=corr.round(2).values,
        texttemplate='%{text}',
        showscale=True
    ))
    fig.update_layout(title='Correlation Matrix', template='plotly_dark')
    return fig

def plot_categorical_default_rates(df):
    cats = ['grade', 'home_ownership', 'purpose', 'term']
    figs = {}
    for col in cats:
        if col not in df.columns:
            continue
        rates = df.groupby(col)['default'].mean().reset_index()
        rates.columns = [col, 'default_rate']
        rates = rates.sort_values('default_rate', ascending=False)
        fig = px.bar(rates, x=col, y='default_rate',
                     title=f'Default Rate by {col.replace("_", " ").title()}',
                     template='plotly_dark',
                     labels={col: col.replace('_', ' ').title(), 'default_rate': 'Default Rate'})
        figs[col] = fig
    return figs

def plot_box_by_default(df, col):
    fig = px.box(df, x='default', y=col, color='default',
                 title=f'{col} Distribution by Default Status',
                 template='plotly_dark',
                 labels={'default': 'Default Status', col: col},
                 category_orders={'default': [0, 1]})
    return fig
