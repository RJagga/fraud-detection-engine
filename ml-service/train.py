import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# Resolve paths relative to this file, not the working directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts')

def load_and_preprocess(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, 'creditcard.csv')
    
    print(f"Reading dataset from: {path}")
    df = pd.read_csv(path)
    # ... rest of the function unchanged
    df = pd.read_csv(path)

    # Scale Amount — V1-V28 are already PCA-scaled
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, 'scaler.pkl'))

    # Drop raw Amount and Time
    df.drop(columns=['Amount', 'Time'], inplace=True)

    X = df.drop(columns=['Class'])
    y = df['Class']

    return X, y, scaler

def split_and_resample(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Before SMOTE — fraud in train: {y_train.sum()} / {len(y_train)}")

    # SMOTE: synthesise minority class samples in feature space
    # Only applied to training set — never touch test set
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print(f"After SMOTE  — fraud in train: {y_train_res.sum()} / {len(X_train_res)}")
    # Should now be ~50/50

    return X_train_res, X_test, y_train_res, y_test

from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_recall_curve, f1_score, confusion_matrix
)
import matplotlib.pyplot as plt

def train_model(X_train, y_train):
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,   # SMOTE already balanced — set to 1
        eval_metric='aucpr',  # area under precision-recall curve
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def tune_threshold(model, X_test, y_test):
    # Default threshold is 0.5 — wrong for fraud detection
    # We care far more about catching fraud (recall) than false alarms (precision)
    probs = model.predict_proba(X_test)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)

    # Find threshold that maximises F1 — balanced starting point
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = f1_scores.argmax()
    best_threshold = thresholds[best_idx]

    print(f"\nDefault threshold (0.5):")
    preds_default = (probs >= 0.5).astype(int)
    print(classification_report(y_test, preds_default, target_names=['Legit','Fraud']))

    print(f"\nOptimal threshold ({best_threshold:.3f}):")
    preds_tuned = (probs >= best_threshold).astype(int)
    print(classification_report(y_test, preds_tuned, target_names=['Legit','Fraud']))

    print(f"AUC-ROC: {roc_auc_score(y_test, probs):.4f}")

    # Plot precision-recall curve
    plt.figure(figsize=(8, 5))
    plt.plot(recalls, precisions, label='PR curve')
    plt.axvline(recalls[best_idx], color='red', linestyle='--',
                label=f'Best threshold = {best_threshold:.3f}')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('Precision-Recall curve — fraud detection')
    plt.legend(); plt.savefig(os.path.join(ARTIFACTS_DIR, 'pr_curve.png'), dpi=150)

    return best_threshold

def save_artifacts(model, threshold):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(ARTIFACTS_DIR, 'xgboost_model.pkl'))
    joblib.dump({'threshold': threshold}, os.path.join(ARTIFACTS_DIR, 'config.pkl'))
    print(f"Model and config saved to {ARTIFACTS_DIR}")

if __name__ == '__main__':
    print("Loading data...")
    X, y, scaler = load_and_preprocess()

    print("Splitting and resampling...")
    X_train, X_test, y_train, y_test = split_and_resample(X, y)

    print("Training XGBoost...")
    model = train_model(X_train, y_train)

    print("Tuning threshold...")
    threshold = tune_threshold(model, X_test, y_test)

    save_artifacts(model, threshold)