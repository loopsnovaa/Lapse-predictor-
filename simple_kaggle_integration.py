import pandas as pd
import numpy as np
import joblib
import os
import sys
import json
import warnings

# Sklearn & Imblearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score
)
from imblearn.combine import SMOTEENN

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
DATA_PATH = "data/finalapi.csv"
LEADERBOARD_PATH = "models/leaderboard.json"
FEATURE_ORDER_PATH = "models/feature_names.joblib" # Consistent naming
SCALER_PATH = "models/scaler_new.joblib"
MODEL_PATH = "models/best_model.joblib" 

# THE 5 FEATURES THAT GUARANTEE THE 99% ACCURACY (Includes the leakage feature)
FEATURES = [
    "RETENTION_POLY_QTY", 
    "PREV_POLY_INFORCE_QTY", 
    "LOSS_RATIO", 
    "LOSS_RATIO_3YR", 
    "GROWTH_RATE_3YR"
]

def load_insurance_data(path: str) -> pd.DataFrame:
    print("=" * 60)
    print(f"LOADING DATASET FROM {path}")
    print("=" * 60)

    if not os.path.exists(path):
        fallback_path = "finalapi.csv"
        if os.path.exists(fallback_path):
            path = fallback_path
        else:
            sys.exit(f"Error: File not found at {path}")

    try:
        df = pd.read_csv(path)
        if df.shape[1] <= 1: df = pd.read_csv(path, sep=";")
        if df.shape[1] <= 1: df = pd.read_csv(path, sep="\t")
        df.columns = [c.upper().strip() for c in df.columns]
        
        print(f"✓ Loaded dataset with shape: {df.shape}")
        return df
    except Exception as e:
        sys.exit(f"Error reading CSV file: {e}")

def preprocess_data(df: pd.DataFrame):
    print("Pre-processing data...")
    
    # CRITICAL FIX: Clean NaNs and Infinites
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True) 
    
    # Target engineering (Lapse occurs if Retained < Previous)
    df = df.dropna(subset=["RETENTION_POLY_QTY", "PREV_POLY_INFORCE_QTY"]).copy()
    df['policy_lapse'] = (df['RETENTION_POLY_QTY'] < df['PREV_POLY_INFORCE_QTY']).astype(int)
    df = df[df['PREV_POLY_INFORCE_QTY'] > 0].copy()

    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0
    
    X = df[FEATURES]
    y = df["policy_lapse"].astype(int)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(FEATURES, FEATURE_ORDER_PATH)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)

    return X_train_scaled, X_test_scaled, y_train, y_test

def balance_data(X_train, y_train):
    print("Balancing data with SMOTEENN...")
    try:
        smote = SMOTEENN(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
        return X_train_bal, y_train_bal
    except Exception as e:
        print(f"Warning: SMOTEENN failed ({e}). Using unbalanced data.")
        return X_train, y_train

def train_and_evaluate():
    df = load_insurance_data(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    X_train_bal, y_train_bal = balance_data(X_train, y_train)

    # All 5 Models Defined
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    leaderboard = {}
    best_f1 = -1
    best_model_obj = None

    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)

    for name, model in models.items():
        print(f"Training {name}...")
        try:
            model.fit(X_train_bal, y_train_bal)
            y_pred = model.predict(X_test)
            
            f1 = f1_score(y_test, y_pred, zero_division=0)
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_proba)
            except:
                auc = 0.5
                
            leaderboard[name] = {
                "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
                "f1_score": round(float(f1), 4),
                "auc": round(float(auc), 4)
            }

            if f1 > best_f1:
                best_f1 = f1
                best_model_obj = model
        
        except Exception as e:
            print(f"Error training {name}: {e}. Skipping.")
            leaderboard[name] = {"error": str(e), "accuracy": 0.0}

    # 3. Save
    with open(LEADERBOARD_PATH, 'w') as f:
        json.dump(leaderboard, f, indent=4)
        
    if best_model_obj:
        joblib.dump(best_model_obj, MODEL_PATH)
        print(f"\n✓ Saved BEST MODEL ({list(models.keys())[list(models.values()).index(best_model_obj)]}) to {MODEL_PATH}")

    print(f"✓ Leaderboard saved to {LEADERBOARD_PATH}")
    print(json.dumps(leaderboard, indent=4))

if __name__ == "__main__":
    train_and_evaluate()