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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# -----------------------------------------------------------
# Load & Clean Dataset
# -----------------------------------------------------------
def load_and_clean_kaggle_data(file_path: str) -> pd.DataFrame:

    df = pd.read_csv(file_path, sep=";", encoding="utf-8")

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

    return df_simple

# -----------------------------------------------------------
# Train Model (ONLY 14 FEATURES)
# -----------------------------------------------------------
def run_training(df: pd.DataFrame):

    df_sample = df.sample(n=10000, random_state=42) if len(df) > 10000 else df

    features = [
        "age", "gender", "policy_type_1", "policy_type_2", "policy_amount",
        "premium_amount", "policy_tenure_years", "policy_tenure_decimal",
        "channel1", "channel2", "channel3", "substandard_risk",
        "number_of_advance_premium", "initial_benefit"
    ]

    X = df_sample[features]
    y = df_sample["policy_lapse"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    os.makedirs("models", exist_ok=True)
    joblib.dump(features, "models/training_feature_order.joblib")

    # ----- Correct SMOTE-ENN Pipeline -----
    smote_enn = SMOTEENN(random_state=42)
    X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    # ----- Train XGBoost -----
    xgb = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6
    )

    xgb.fit(X_train_scaled, y_train_resampled)

    # ----- Evaluate -----
    print("\nEVALUATION")
    y_proba = xgb.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    print(f"AUC = {roc_auc_score(y_test, y_proba):.4f}")
    print(classification_report(y_test, y_pred))

    # ----- Save Model -----
    joblib.dump(xgb, "models/xgboost_optimized_model.joblib")
    joblib.dump(scaler, "models/scaler.joblib")

    print("\n✓ MODEL TRAINED AND SAVED SUCCESSFULLY")

# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
def main():
    df = load_and_clean_kaggle_data("data/Kaggle.csv")
    run_training(df)

if __name__ == "__main__":
    main()
