# 🏦 Credit Risk Model (Loan Default Prediction)
Predict loan defaults using **Logistic Regression** and **XGBoost** on LendingClub-style data. The project covers end-to-end ML: synthetic data generation, exploratory data analysis, model training, evaluation, and an interactive Streamlit dashboard.
## Project Structure
```
Credit_Risk_Model/
├── data/
│   ├── lending_club.csv       # Generated or real dataset
│   └── metrics.json           # Saved model evaluation metrics
├── src/
│   ├── generate_data.py       # Synthetic LendingClub data generator
│   ├── eda.py                 # EDA plotting functions (Plotly)
│   ├── model.py               # Preprocessing pipeline + model training
│   └── roc_auc.py             # ROC, feature importance, confusion matrix, reports
├── notebook.ipynb             # End-to-end Jupyter analysis notebook
├── app.py                     # Streamlit interactive dashboard
└── README.md
```
## Features
The model is trained on 17 borrower and loan features:
| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `loan_amnt` | Numeric | Requested loan amount ($) |
| 2 | `term` | Categorical | Loan term (36 or 60 months) |
| 3 | `int_rate` | Numeric | Interest rate (%) |
| 4 | `installment` | Numeric | Monthly payment amount ($) |
| 5 | `grade` | Categorical | LendingClub loan grade (A–G) |
| 6 | `emp_length` | Numeric | Employment length (years) |
| 7 | `annual_inc` | Numeric | Borrower annual income ($) |
| 8 | `dti` | Numeric | Debt-to-income ratio |
| 9 | `home_ownership` | Categorical | RENT / OWN / MORTGAGE / OTHER |
| 10 | `purpose` | Categorical | Loan purpose (debt consolidation, etc.) |
| 11 | `open_acc` | Numeric | Number of open credit lines |
| 12 | `revol_bal` | Numeric | Revolving credit balance ($) |
| 13 | `revol_util` | Numeric | Revolving line utilization rate (%) |
| 14 | `total_acc` | Numeric | Total credit lines ever opened |
| 15 | `delinq_2yrs` | Numeric | Delinquencies in past 2 years |
| 16 | `pub_rec` | Numeric | Number of derogatory public records |
| 17 | `mort_acc` | Numeric | Number of mortgage accounts |
## Model Comparison
| Metric | Logistic Regression | XGBoost |
|--------|-------------------|---------|
| ROC-AUC | > 0.75 | > 0.75 |
| Precision | Baseline | Higher |
| Recall | Baseline | Higher |
| F1-Score | Baseline | Higher |
| Training Speed | Fast | Moderate |
| Interpretability | High (coefficients) | Medium (SHAP/gain) |
> Both models target **ROC-AUC > 0.75** on the held-out test set (20% split, stratified).
## Quick Start
**1. Set Python path to include src modules:**
```tcsh
setenv PYTHONPATH /path/to/site-packages
```
**2. Generate synthetic data and run the notebook:**
```tcsh
cd Credit_Risk_Model
jupyter notebook notebook.ipynb
```
**3. Launch the Streamlit app:**
```tcsh
cd Credit_Risk_Model
streamlit run app.py
```
## Streamlit App Features
The interactive dashboard (`app.py`) provides four tabs:
| Tab | Description |
|-----|-------------|
| **EDA** | Target distribution, numeric histograms, correlation heatmap, categorical default rates |
| **Train Models** | One-click training of Logistic Regression and XGBoost with live progress |
| **Model Evaluation** | Side-by-side ROC curves, confusion matrices, classification reports, feature importances |
| **Predict** | Enter loan details manually and get a real-time default probability from both models |
## Tech Stack
| Library | Purpose |
|---------|---------|
| `scikit-learn` | Logistic Regression, preprocessing pipeline, metrics |
| `XGBoost` | Gradient-boosted tree classifier |
| `Streamlit` | Interactive web dashboard |
| `Plotly` | All interactive visualizations |
| `pandas` | Data manipulation and feature engineering |
| `numpy` | Numerical operations |
