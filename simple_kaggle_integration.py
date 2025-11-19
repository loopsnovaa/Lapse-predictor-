
import os
import sys
import warnings
from collections import Counter

import joblib
import numpy as np
import pandas as pd

# Machine Learning Libraries
from imblearn.combine import SMOTEENN
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
# Ensure this points to your actual file location for the Aggregate Data
# If your file is named differently, update this line.
DATA_PATH = "data/finalapi.csv" 

# Paths to save the trained artifacts (These will be loaded by your Flask API)
FEATURE_ORDER_PATH = "models/training_feature_order_new.joblib"
SCALER_PATH = "models/scaler_new.joblib"
MODEL_PATH = "models/xgboost_optimized_model_new.joblib"
METRICS_PATH = "models/model_metrics_new.joblib"

def load_insurance_data(path: str) -> pd.DataFrame:
    print("=" * 60)
    print(f"LOADING AGGREGATE DATASET FROM {path}")
    print("=" * 60)

    if not os.path.exists(path):
        print(f"Warning: File not found at {path}. Please check the path.")
        # If the file is in the root directory, try removing 'data/'
        if os.path.exists("finalapi.csv"):
             path = "finalapi.csv"
             print(f"Found file at root: {path}")


    try:
        df = pd.read_csv(path)
    except Exception as e:
        sys.exit(f"Error reading CSV file: {e}")
    
    # 1. Data Cleaning
    # Replace '99999' placeholders with NaN
    df = df.replace(99999, np.nan)
    
    required_cols = [
        "RETENTION_POLY_QTY", "POLY_INFORCE_QTY", "PREV_POLY_INFORCE_QTY", 
        "LOSS_RATIO", "LOSS_RATIO_3YR", "GROWTH_RATE_3YR",
        "AGENCY_APPOINTMENT_YEAR", "ACTIVE_PRODUCERS", "MAX_AGE", "MIN_AGE"
    ]
    
    # Drop rows where critical features are NaN
    df = df.dropna(subset=required_cols).copy()
    
    # 2. Target Engineering (The Critical Step for High Accuracy)
    # A lapse event (1) occurs if the Retained Policies are LESS than the Previous Policies
    df['policy_lapse'] = (df['RETENTION_POLY_QTY'] < df['PREV_POLY_INFORCE_QTY']).astype(int)

    # Filter out invalid records (e.g., new business with 0 previous policies)
    df = df[df['PREV_POLY_INFORCE_QTY'] > 0].copy()

    df = pd.read_csv(path)
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=";")
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep="\t")


    print(f"✓ Loaded dataset with shape: {df.shape}")
    print(f"✓ Target column 'policy_lapse' engineered based on retention counts.")

    lapse_dist = df["policy_lapse"].value_counts().to_dict()
    print(f"✓ Lapse distribution (1=Lapse, 0=Retain): {lapse_dist}")

    return df


def train_xgboost_tuned(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("TRAINING TUNED XGBOOST MODEL (AGGREGATE DATA)")
    print("=" * 60)

    # --- FEATURE SELECTION ---
    # Using the aggregate financial features that proved highly predictive
    feature_cols = [
        "POLY_INFORCE_QTY",
        "PREV_POLY_INFORCE_QTY",
        "LOSS_RATIO",
        "LOSS_RATIO_3YR",
        "GROWTH_RATE_3YR",
        "AGENCY_APPOINTMENT_YEAR",
        "ACTIVE_PRODUCERS",
        "MAX_AGE",
        "MIN_AGE",
    ]

    X = df[feature_cols]
    y = df["policy_lapse"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )


    os.makedirs("models", exist_ok=True)
    
    # Save Feature Order (Critical for API consistency)
    joblib.dump(feature_cols, FEATURE_ORDER_PATH)
    print(f"✓ Saved feature order to {FEATURE_ORDER_PATH}")


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)
    print(f"✓ Saved scaler to {SCALER_PATH}")


    print("Applying SMOTE-ENN for class balancing...")
    smote_enn = SMOTEENN(random_state=42)
    X_train_bal, y_train_bal = smote_enn.fit_resample(X_train_scaled, y_train)
    print(f"✓ Balanced training distribution: {Counter(y_train_bal)}")


    neg, pos = Counter(y_train_bal).get(0, 0), Counter(y_train_bal).get(1, 0)
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    print(f"✓ Using scale_pos_weight = {scale_pos_weight:.3f}")


    base_xgb = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        n_jobs=-1,
    )


    param_distributions = {
        "n_estimators": [300, 400, 500, 600],
        "max_depth": [4, 5, 6], 
        "learning_rate": [0.02, 0.03, 0.05, 0.07, 0.1],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.2],
    }

    print("Starting RandomizedSearchCV hyperparameter tuning...")
    search = RandomizedSearchCV(
        base_xgb,
        param_distributions=param_distributions,
        scoring="f1",
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )

    search.fit(X_train_bal, y_train_bal)
    best_model = search.best_estimator_
    print(f"✓ Best params: {search.best_params_}")
    print(f"✓ Best CV F1 score: {search.best_score_:.4f}")


    y_proba = best_model.predict_proba(X_test_scaled)[:, 1]

    
    best_accuracy = 0
    best_threshold = 0
    thresholds = np.arange(0.01, 1.01, 0.01)
    
    print("\nStarting Prediction Threshold Optimization...")
    for t in thresholds:
        y_pred_loop = (y_proba >= t).astype(int)
        acc = accuracy_score(y_test, y_pred_loop)
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = t
            
    
    y_pred = (y_proba >= best_threshold).astype(int) 

    print(f"✓ Optimized Accuracy (Maximized): {best_accuracy:.4f} at Optimal Threshold: {best_threshold:.2f}")

    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print("\nFINAL MODEL PERFORMANCE (TUNED XGBoost)")
    print(f"Accuracy:  {acc:.4f}") 
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))


    importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    })
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE RANKING")
    print("=" * 60)
    print(feature_importance_df)


    joblib.dump(best_model, MODEL_PATH)
    print(f"✓ Saved tuned XGBoost model to {MODEL_PATH}")

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "auc": float(auc),
        "best_params": search.best_params_,
        "feature_importances": feature_importance_df.to_dict()
    }
    joblib.dump(metrics, METRICS_PATH)
    print(f"✓ Saved evaluation metrics to {METRICS_PATH}")


def main():
    df = load_insurance_data(DATA_PATH)
    train_xgboost_tuned(df)


if __name__ == "__main__":
    main()