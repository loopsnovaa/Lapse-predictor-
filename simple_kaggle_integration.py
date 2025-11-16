#!/usr/bin/env python3
"""
Insurance Policy Lapse Model Training (XGBoost-only, Tuned, Using Processed CSV)

Assumes data/Kaggle.csv already has these columns:

    policy_id
    age
    gender
    policy_type_1
    policy_type_2
    policy_amount
    premium_amount
    policy_tenure_years
    policy_tenure_decimal
    channel1
    channel2
    channel3
    substandard_risk
    number_of_advance_premium
    initial_benefit
    policy_lapse
    premium_to_benefit_ratio
    age_squared
    premium_squared
    benefit_squared

This script:

1) Loads that CSV directly (no extra cleaning)
2) Splits into train / test
3) Balances the training data with SMOTE-ENN
4) Runs RandomizedSearchCV to tune XGBoost hyperparameters
5) Evaluates with accuracy, precision, recall, F1, AUC
6) Saves:

   models/training_feature_order.joblib
   models/scaler.joblib
   models/xgboost_optimized_model.joblib
   models/model_metrics.joblib

These are compatible with your existing API + dashboard.
"""

import os
import sys
import warnings
from collections import Counter

import joblib
import numpy as np
import pandas as pd

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

DATA_PATH = "data/Kaggle.csv"

FEATURE_ORDER_PATH = "models/training_feature_order.joblib"
SCALER_PATH = "models/scaler.joblib"
MODEL_PATH = "models/xgboost_optimized_model.joblib"
METRICS_PATH = "models/model_metrics.joblib"


# -------------------------------------------------------------
# 1. Load your ALREADY-PROCESSED insurance dataset
# -------------------------------------------------------------
def load_insurance_data(path: str) -> pd.DataFrame:
    print("=" * 60)
    print("LOADING PROCESSED INSURANCE DATASET FROM Kaggle.csv")
    print("=" * 60)

    if not os.path.exists(path):
        sys.exit(f"Dataset not found at {path}")

    # Try normal CSV, then ;, then tab, in case of weird separators
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=";")
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep="\t")

    print(f"✓ Loaded dataset with shape: {df.shape}")
    print(f"✓ Columns: {list(df.columns)}")

    expected_cols = [
        "policy_id",
        "age",
        "gender",
        "policy_type_1",
        "policy_type_2",
        "policy_amount",
        "premium_amount",
        "policy_tenure_years",
        "policy_tenure_decimal",
        "channel1",
        "channel2",
        "channel3",
        "substandard_risk",
        "number_of_advance_premium",
        "initial_benefit",
        "policy_lapse",
        "premium_to_benefit_ratio",
        "age_squared",
        "premium_squared",
        "benefit_squared",
    ]

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"The following expected columns are missing from Kaggle.csv: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    lapse_dist = df["policy_lapse"].value_counts().to_dict()
    print(f"✓ Lapse distribution: {lapse_dist}")

    return df[expected_cols].copy()


# -------------------------------------------------------------
# 2. Train tuned XGBoost model (balanced)
# -------------------------------------------------------------
def train_xgboost_tuned(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("TRAINING TUNED XGBOOST MODEL (BALANCED)")
    print("=" * 60)

    # Features & target
    feature_cols = [
        "age",
        "gender",
        "policy_type_1",
        "policy_type_2",
        "policy_amount",
        "premium_amount",
        "policy_tenure_years",
        "policy_tenure_decimal",
        "channel1",
        "channel2",
        "channel3",
        "substandard_risk",
        "number_of_advance_premium",
        "initial_benefit",
        "premium_to_benefit_ratio",
        "age_squared",
        "premium_squared",
        "benefit_squared",
    ]

    X = df[feature_cols]
    y = df["policy_lapse"].astype(int)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save feature order for API
    os.makedirs("models", exist_ok=True)
    joblib.dump(feature_cols, FEATURE_ORDER_PATH)
    print(f"✓ Saved feature order to {FEATURE_ORDER_PATH}")

    # Scale features (mainly for stability with squared terms)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)
    print(f"✓ Saved scaler to {SCALER_PATH}")

    # Balance training with SMOTE-ENN
    print("Applying SMOTE-ENN for class balancing...")
    smote_enn = SMOTEENN(random_state=42)
    X_train_bal, y_train_bal = smote_enn.fit_resample(X_train_scaled, y_train)
    print(f"✓ Balanced training distribution: {Counter(y_train_bal)}")

    # Class imbalance weight
    neg, pos = Counter(y_train_bal).get(0, 0), Counter(y_train_bal).get(1, 0)
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    print(f"✓ Using scale_pos_weight = {scale_pos_weight:.3f}")

    # Base XGBoost model
    base_xgb = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        n_jobs=-1,
    )

    # Hyperparameter search space
    param_distributions = {
        "n_estimators": [300, 400, 500, 600],
        "max_depth": [4, 5, 6, 7, 8],
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
        n_iter=20,                # increase to 30+ if you want even better tuning
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

    # Evaluate on test set
    y_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

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

    # Save model & metrics
    joblib.dump(best_model, MODEL_PATH)
    print(f"✓ Saved tuned XGBoost model to {MODEL_PATH}")

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "auc": float(auc),
        "best_params": search.best_params_,
    }
    joblib.dump(metrics, METRICS_PATH)
    print(f"✓ Saved evaluation metrics to {METRICS_PATH}")

    print("\nFINAL XGBOOST METRIC SUMMARY")
    for k, v in metrics.items():
        if k == "best_params":
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v:.4f}")


# -------------------------------------------------------------
# 3. MAIN
# -------------------------------------------------------------
def main():
    df = load_insurance_data(DATA_PATH)
    train_xgboost_tuned(df)


if __name__ == "__main__":
    main()
