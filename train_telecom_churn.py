import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA_PATH = "data/Telco-Customer-Churn.csv"
MODEL_PATH = "models/telecom_churn_model.joblib"
METRICS_PATH = "models/model_metrics.joblib"


def load_and_prepare_data(path: str):
    print("=" * 60)
    print("LOADING TELECOM CHURN DATASET")
    print("=" * 60)

    if not os.path.exists(path):
        sys.exit(f"Dataset not found at {path}")

    df = pd.read_csv(path)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    print("✓ Dataset shape:", df.shape)
    print("✓ Churn distribution:", df["Churn"].value_counts().to_dict())

    return df


def train_model(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("TRAINING 20-FEATURE TELECOM CHURN MODEL")
    print("=" * 60)

    feature_cols = [
        "gender", "SeniorCitizen", "Partner", "Dependents",
        "tenure", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges",
    ]

    X = df[feature_cols]
    y = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    categorical = [
        "gender", "Partner", "Dependents",
        "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )

    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = neg / pos
    print(f"✓ class imbalance scale weight: {scale_pos_weight:.3f}")

    model = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist"
    )

    clf = Pipeline([
        ("prep", preprocessor),
        ("model", model),
    ])

    clf.fit(X_train, y_train)
    print("✓ Training complete")

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\nFINAL MODEL METRICS:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(
        {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "auc": float(auc),
        },
        METRICS_PATH,
    )

    print("\n✓ Model saved to:", MODEL_PATH)
    print("✓ Metrics saved to:", METRICS_PATH)


def main():
    df = load_and_prepare_data(DATA_PATH)
    train_model(df)


if __name__ == "__main__":
    main()
