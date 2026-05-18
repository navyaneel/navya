import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from src.generate_data import generate_lending_data, save_data
from src.eda import load_data, summary_stats, plot_target_distribution, \
    plot_numeric_distributions, plot_correlation_matrix, \
    plot_categorical_default_rates, plot_box_by_default
from src.model import run_pipeline, get_feature_importance, XGBOOST_AVAILABLE
from src.roc_auc import compute_metrics, plot_roc_curve, plot_confusion_matrix, \
    plot_precision_recall_curve, plot_feature_importance, classification_report_df

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

st.set_page_config(page_title="Credit Risk Model", page_icon="🏦", layout="wide")
st.title("🏦 Credit Risk Model — Loan Default Prediction")
st.markdown("**Logistic Regression & XGBoost** on LendingClub-style data")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Settings")
    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    st.markdown("---")
    gen_btn = st.button("📥 Generate Data")
    train_btn = st.button("🚀 Train Models")

csv_path = os.path.join(DATA_DIR, 'lending_club.csv')

if gen_btn:
    with st.spinner("Generating synthetic LendingClub data..."):
        save_data()
    st.success("Data generated successfully!")

if not os.path.exists(csv_path):
    st.warning("No data found. Click **📥 Generate Data** in the sidebar.")
    st.stop()

df = pd.read_csv(csv_path)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 EDA", "🧠 Train Models", "📉 Model Evaluation", "🔍 Predict"
])

with tab1:
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Loans", f"{len(df):,}")
    col2.metric("Default Rate", f"{df['default'].mean():.1%}")
    col3.metric("Avg Loan", f"${df['loan_amnt'].mean():,.0f}")
    col4.metric("Avg Interest", f"{df['int_rate'].mean():.1f}%")

    st.plotly_chart(plot_target_distribution(df), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_numeric_distributions(df), use_container_width=True)
    with c2:
        st.plotly_chart(plot_correlation_matrix(df), use_container_width=True)

    st.subheader("Default Rates by Category")
    cat_figs = plot_categorical_default_rates(df)
    cols = st.columns(2)
    for i, (name, fig) in enumerate(cat_figs.items()):
        with cols[i % 2]:
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Distribution by Default")
    box_col = st.selectbox("Select Feature", ['int_rate', 'dti', 'annual_inc', 'loan_amnt', 'revol_util'])
    st.plotly_chart(plot_box_by_default(df, box_col), use_container_width=True)

    st.subheader("Data Preview")
    st.dataframe(df.head(50), use_container_width=True)

with tab2:
    if train_btn:
        with st.spinner("Training models..."):
            results = run_pipeline(df, test_size=test_size)
        st.session_state['results'] = results
        st.session_state['trained'] = True

        lr = results['lr_model']
        xgb = results['xgb_model']
        xgb_available = results.get('xgboost_available', False)
        X_test = results['X_test']
        y_test = results['y_test']
        feat_names = results['feature_names']

        lr_pred = lr.predict(X_test)
        lr_proba = lr.predict_proba(X_test)[:, 1]
        lr_metrics = compute_metrics(y_test, lr_pred, lr_proba)

        if xgb_available and xgb is not None:
            xgb_pred = xgb.predict(X_test)
            xgb_proba = xgb.predict_proba(X_test)[:, 1]
            xgb_metrics = compute_metrics(y_test, xgb_pred, xgb_proba)
        else:
            xgb_pred = None
            xgb_proba = None
            xgb_metrics = None
            st.warning("XGBoost is unavailable in this environment. Only Logistic Regression results are shown.")

        st.session_state['lr_pred'] = lr_pred
        st.session_state['lr_proba'] = lr_proba
        st.session_state['xgb_pred'] = xgb_pred
        st.session_state['xgb_proba'] = xgb_proba
        st.session_state['lr_metrics'] = lr_metrics
        st.session_state['xgb_metrics'] = xgb_metrics
        st.session_state['xgb_available'] = xgb_available

        st.success("Training complete!")

        st.subheader("Model Comparison")
        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown("**Logistic Regression**")
            for k, v in lr_metrics.items():
                st.metric(k.upper(), f"{v:.4f}")
        with mc2:
            st.markdown("**XGBoost**")
            if xgb_metrics is not None:
                for k, v in xgb_metrics.items():
                    st.metric(k.upper(), f"{v:.4f}")
            else:
                st.info("XGBoost metrics unavailable.")

        st.subheader("Feature Importance")
        fi1, fi2 = st.columns(2)
        with fi1:
            lr_imp = get_feature_importance(lr, feat_names, 'logistic')
            st.plotly_chart(plot_feature_importance(lr_imp, 'Logistic Regression'), use_container_width=True)
        with fi2:
            if xgb is not None:
                xgb_imp = get_feature_importance(xgb, feat_names, 'xgboost')
                st.plotly_chart(plot_feature_importance(xgb_imp, 'XGBoost'), use_container_width=True)
            else:
                st.info("XGBoost feature importance unavailable.")

        os.makedirs(DATA_DIR, exist_ok=True)
        metrics_out = {'logistic_regression': lr_metrics, 'xgboost': xgb_metrics}
        with open(os.path.join(DATA_DIR, 'metrics.json'), 'w') as f:
            json.dump(metrics_out, f, indent=2)
    else:
        st.info("Click **🚀 Train Models** in the sidebar to begin.")

with tab3:
    if st.session_state.get('trained'):
        y_test = st.session_state['results']['y_test']
        lr_proba = st.session_state['lr_proba']
        xgb_proba = st.session_state['xgb_proba']
        lr_pred = st.session_state['lr_pred']
        xgb_pred = st.session_state['xgb_pred']
        xgb_available = st.session_state.get('xgb_available', False)

        if xgb_available and xgb_proba is not None:
            st.plotly_chart(plot_roc_curve(y_test, lr_proba, xgb_proba), use_container_width=True)
            st.plotly_chart(plot_precision_recall_curve(y_test, lr_proba, xgb_proba), use_container_width=True)

            cm1, cm2 = st.columns(2)
            with cm1:
                st.plotly_chart(plot_confusion_matrix(y_test, lr_pred, 'Logistic Regression'), use_container_width=True)
            with cm2:
                st.plotly_chart(plot_confusion_matrix(y_test, xgb_pred, 'XGBoost'), use_container_width=True)

            st.subheader("Classification Reports")
            r1, r2 = st.columns(2)
            with r1:
                st.markdown("**Logistic Regression**")
                st.dataframe(classification_report_df(y_test, lr_pred), use_container_width=True)
            with r2:
                st.markdown("**XGBoost**")
                st.dataframe(classification_report_df(y_test, xgb_pred), use_container_width=True)
        else:
            st.warning("XGBoost is unavailable; showing only Logistic Regression evaluation.")
            st.plotly_chart(plot_confusion_matrix(y_test, lr_pred, 'Logistic Regression'), use_container_width=True)
            st.subheader("Classification Report")
            st.dataframe(classification_report_df(y_test, lr_pred), use_container_width=True)
    else:
        st.info("Train the models first to see evaluation.")

with tab4:
    if st.session_state.get('trained'):
        st.subheader("Predict Loan Default Risk")
        with st.form("predict_form"):
            p1, p2, p3 = st.columns(3)
            with p1:
                loan_amnt = st.number_input("Loan Amount ($)", 1000, 40000, 15000, 500)
                int_rate = st.number_input("Interest Rate (%)", 5.0, 28.0, 12.0, 0.5)
                term = st.selectbox("Term (months)", [36, 60])
                dti = st.number_input("DTI Ratio", 0.0, 40.0, 15.0, 0.5)
            with p2:
                annual_inc = st.number_input("Annual Income ($)", 10000, 300000, 60000, 1000)
                emp_length = st.slider("Employment Length (yrs)", 0, 10, 5)
                home = st.selectbox("Home Ownership", ['RENT', 'OWN', 'MORTGAGE'])
                purpose = st.selectbox("Loan Purpose",
                    ['debt_consolidation', 'credit_card', 'home_improvement',
                     'major_purchase', 'small_business', 'car', 'medical', 'other'])
            with p3:
                grade = st.selectbox("Grade", list('ABCDEFG'))
                revol_util = st.number_input("Revolving Utilization (%)", 0.0, 100.0, 50.0, 1.0)
                delinq = st.number_input("Delinquencies (2yr)", 0, 10, 0)
                pub_rec = st.number_input("Public Records", 0, 5, 0)
            submitted = st.form_submit_button("Predict Default Risk")

        if submitted:
            installment = (loan_amnt * (int_rate / 1200) /
                           (1 - (1 + int_rate / 1200) ** -term))
            input_df = pd.DataFrame([{
                'loan_amnt': loan_amnt, 'term': term, 'int_rate': int_rate,
                'installment': installment, 'grade': grade,
                'emp_length': emp_length, 'annual_inc': annual_inc,
                'dti': dti, 'home_ownership': home, 'purpose': purpose,
                'open_acc': 11, 'revol_bal': 15000, 'revol_util': revol_util,
                'total_acc': 25, 'delinq_2yrs': delinq, 'pub_rec': pub_rec,
                'mort_acc': 2, 'default': 0
            }])
            from src.model import preprocess
            full_df = pd.concat([df, input_df], ignore_index=True)
            X_all, _, feat_names, _ = preprocess(full_df)
            X_input = X_all[-1:, :]

            lr = st.session_state['results']['lr_model']
            xgb_m = st.session_state['results']['xgb_model']
            xgb_available = st.session_state.get('xgb_available', False)
            lr_prob = lr.predict_proba(X_input)[0][1]

            st.markdown("### Prediction Results")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Logistic Regression", f"{lr_prob:.1%}",
                          "HIGH RISK" if lr_prob > 0.5 else "LOW RISK")
            if xgb_available and xgb_m is not None:
                xgb_prob = xgb_m.predict_proba(X_input)[0][1]
                with c2:
                    st.metric("XGBoost", f"{xgb_prob:.1%}",
                              "HIGH RISK" if xgb_prob > 0.5 else "LOW RISK")
            else:
                with c2:
                    st.info("XGBoost unavailable for prediction.")
    else:
        st.info("Train the models first to make predictions.")

st.markdown("---")
st.caption("Built with Streamlit • Logistic Regression & XGBoost via scikit-learn")
