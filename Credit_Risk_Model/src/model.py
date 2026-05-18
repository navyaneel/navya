import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False

def preprocess(df):
    df = df.copy()
    cat_cols = ['grade', 'home_ownership', 'purpose']
    existing_cats = [c for c in cat_cols if c in df.columns]
    df = pd.get_dummies(df, columns=existing_cats, drop_first=False)
    y = df['default'].values
    X = df.drop(columns=['default'])
    feature_names = X.columns.tolist()
    numeric_cols = X.select_dtypes(include='number').columns.tolist()
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    X_scaled = X.values
    return X_scaled, y, feature_names, scaler

def train_logistic(X_train, y_train):
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    if not XGBOOST_AVAILABLE:
        return None

    neg = int(np.sum(y_train == 0))
    pos = int(np.sum(y_train == 1))
    spw = neg / pos if pos > 0 else 1
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=spw,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def get_feature_importance(model, feature_names, model_type='logistic'):
    if model is None:
        return pd.DataFrame({'feature': feature_names, 'importance': np.zeros(len(feature_names))})
    if model_type == 'logistic':
        importances = np.abs(model.coef_[0])
    else:
        importances = model.feature_importances_
    df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    return df.sort_values('importance', ascending=False).reset_index(drop=True)

def run_pipeline(df, test_size=0.2):
    X, y, feature_names, scaler = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    lr_model = train_logistic(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)
    return {
        'lr_model': lr_model,
        'xgb_model': xgb_model,
        'xgboost_available': XGBOOST_AVAILABLE,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'scaler': scaler
    }
