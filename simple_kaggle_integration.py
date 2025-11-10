#!/usr/bin/env python3
"""
Simplified Kaggle Dataset Integration Script
Final Version (No Calibration)
------------------------------------------
- Loads, cleans, and processes Kaggle insurance data
- Balances classes with SMOTE-ENN
- Trains Logistic Regression, Random Forest, and Optimized XGBoost (no calibration)
- Saves models, scaler, and processed dataset
"""

import os
import sys
import warnings
from collections import Counter
import joblib
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Load and clean Kaggle dataset
# ----------------------------------------------------------------------
def load_and_clean_kaggle_data(file_path: str) -> pd.DataFrame:
    print("=" * 60)
    print("LOADING AND CLEANING KAGGLE INSURANCE DATASET")
    print("=" * 60)

    df = pd.read_csv(file_path, sep=";", encoding="utf-8")
    print(f"✓ Loaded dataset with shape: {df.shape}")

    df.columns = df.columns.str.strip()
    drop_cols = [c for c in df.columns if "Unnamed" in c]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    df["BENEFIT"] = (
        df["BENEFIT"].astype(str)
        .str.replace(",", "").str.replace(" ", "").str.replace("-", "0")
    )
    df["BENEFIT"] = pd.to_numeric(df["BENEFIT"], errors="coerce").fillna(0)

    df["Premium"] = (
        df["Premium"].astype(str)
        .str.replace(",", "").str.replace(" ", "").str.replace("-", "0")
    )
    df["Premium"] = pd.to_numeric(df["Premium"], errors="coerce").fillna(0)

    df["INITIAL BENEFIT"] = df["INITIAL BENEFIT"].fillna(0)
    df["policy_lapse"] = (df["POLICY STATUS"] == "Lapse").astype(int)

    lapse_counts = df["policy_lapse"].value_counts()
    print(f"✓ Policy lapse distribution: {lapse_counts.to_dict()}")

    df_simple = pd.DataFrame({
        "policy_id": range(1, len(df) + 1),
        "age": df["ENTRY AGE"],
        "gender": df["SEX"].map({"M": 1, "F": 0}),
        "policy_type_1": df["POLICY TYPE 1"],
        "policy_type_2": df["POLICY TYPE 2"],
        "policy_amount": df["BENEFIT"],
        "premium_amount": df["Premium"],
        "policy_tenure_years": df["Policy Year"],
        "policy_tenure_decimal": df["Policy Year (Decimal)"],
        "channel1": df["CHANNEL1"],
        "channel2": df["CHANNEL2"],
        "channel3": df["CHANNEL3"],
        "substandard_risk": df["SUBSTANDARD RISK"],
        "number_of_advance_premium": df["NUMBER OF ADVANCE PREMIUM"],
        "initial_benefit": df["INITIAL BENEFIT"],
        "policy_lapse": df["policy_lapse"]
    })

    df_simple["premium_to_benefit_ratio"] = (
        df_simple["premium_amount"] / (df_simple["policy_amount"] + 1)
    ).replace([np.inf, -np.inf], 0).fillna(0)
    df_simple["age_squared"] = df_simple["age"] ** 2
    df_simple["premium_squared"] = df_simple["premium_amount"] ** 2
    df_simple["benefit_squared"] = df_simple["policy_amount"] ** 2

    print(f"✓ Simplified dataset shape: {df_simple.shape}")
    return df_simple

# ----------------------------------------------------------------------
# Model training and evaluation
# ----------------------------------------------------------------------
def run_simple_prediction(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("RUNNING SIMPLE PREDICTION (No Calibration)")
    print("=" * 60)

    df_sample = df.sample(n=10000, random_state=42) if len(df) > 10000 else df

    X = df_sample.drop(columns=["policy_lapse", "policy_id"])
    y = df_sample["policy_lapse"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    os.makedirs("models", exist_ok=True)
    joblib.dump(list(X.columns), "models/training_feature_order.joblib")
    print("✓ Saved feature order to models/training_feature_order.joblib")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Applying SMOTE-ENN for balancing...")
    smote_enn = SMOTEENN(random_state=42)
    X_train_bal, y_train_bal = smote_enn.fit_resample(X_train_scaled, y_train)
    print(f"✓ Balanced training set: {Counter(y_train_bal)}")

    # Logistic Regression
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_bal, y_train_bal)

    # Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_bal, y_train_bal)

    # XGBoost (No calibration)
    print("\nTraining Optimized XGBoost (No Calibration)...")
    neg, pos = Counter(y_train_bal).get(0, 0), Counter(y_train_bal).get(1, 0)
    scale_pos_weight = neg / pos if pos > 0 else 1
    print(f"→ Using scale_pos_weight = {scale_pos_weight:.3f}")

    xgb_model = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        n_estimators=400,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8
    )
    xgb_model.fit(X_train_bal, y_train_bal)
    print("✓ XGBoost training complete.")

    # Evaluate models
    print("\n" + "=" * 40)
    print("MODEL EVALUATION (TEST SET)")
    print("=" * 40)

    models = {
        "Logistic Regression": lr_model,
        "Random Forest": rf_model,
        "Optimized XGBoost": xgb_model
    }

    results = {}
    for name, model in models.items():
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        auc = roc_auc_score(y_test, y_proba)
        acc = (y_pred == y_test.values).mean()
        print(f"\n{name}:")
        print(f"  AUC = {auc:.4f}, Accuracy = {acc:.4f}")
        print(classification_report(y_test, y_pred))
        results[name] = {"auc": auc, "accuracy": acc}

    # Save models and data
    os.makedirs("data", exist_ok=True)
    df_sample.to_csv("data/kaggle_processed_simple.csv", index=False)
    print("✓ Saved processed dataset to data/kaggle_processed_simple.csv")

    joblib.dump(lr_model, "models/logistic_regression_model.joblib")
    joblib.dump(rf_model, "models/random_forest_model.joblib")
    joblib.dump(xgb_model, "models/xgboost_optimized_model.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    print("✓ Models and scaler saved to models/ directory")

    return results

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("REAL KAGGLE INSURANCE DATASET - FINAL TRAINING (No Calibration)")
    print("=" * 60)

    data_path = "data/Kaggle.csv"
    if not os.path.exists(data_path):
        sys.exit(f"Dataset not found at {data_path}")

    df = load_and_clean_kaggle_data(data_path)
    results = run_simple_prediction(df)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    for name, m in results.items():
        print(f"{name}: AUC={m['auc']:.4f}, Accuracy={m['accuracy']:.4f}")
    print("\n✓ All models updated successfully.")


if __name__ == "__main__":
    main()
