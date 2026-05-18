import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))
from Credit_Risk_Model.src.generate_data import generate_lending_data, save_data
from Credit_Risk_Model.src.eda import load_data
from Credit_Risk_Model.src.model import run_pipeline, get_feature_importance
from Credit_Risk_Model.src.roc_auc import compute_metrics

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def main():
    csv_path = os.path.join(DATA_DIR, 'lending_club.csv')
    if not os.path.exists(csv_path):
        print("Generating data...")
        save_data()
    df = load_data()
    print(f"Dataset: {len(df)} rows, default rate: {df['default'].mean():.1%}")

    results = run_pipeline(df, test_size=0.2)
    lr = results['lr_model']
    xgb = results['xgb_model']
    X_test = results['X_test']
    y_test = results['y_test']
    feat_names = results['feature_names']

    lr_pred = lr.predict(X_test)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    xgb_pred = xgb.predict(X_test)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]

    lr_metrics = compute_metrics(y_test, lr_pred, lr_proba)
    xgb_metrics = compute_metrics(y_test, xgb_pred, xgb_proba)

    print("\n=== Logistic Regression ===")
    for k, v in lr_metrics.items():
        print(f"  {k}: {v}")

    print("\n=== XGBoost ===")
    for k, v in xgb_metrics.items():
        print(f"  {k}: {v}")

    print("\n=== Top Features (XGBoost) ===")
    fi = get_feature_importance(xgb, feat_names, 'xgboost')
    for _, row in fi.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, 'metrics.json'), 'w') as f:
        json.dump({'logistic_regression': lr_metrics, 'xgboost': xgb_metrics}, f, indent=2)
    print(f"\nSaved metrics to {DATA_DIR}/metrics.json")


if __name__ == '__main__':
    main()
