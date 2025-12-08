import pandas as pd
import numpy as np
import joblib
import os
import sys
import json
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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
FEATURE_ORDER_PATH = "models/training_feature_order_new.joblib"
SCALER_PATH = "models/scaler_new.joblib"
MODEL_PATH = "models/xgboost_optimized_model_new.joblib"

def load_insurance_data(path: str) -> pd.DataFrame:
    print("=" * 60)
    print(f"LOADING AGGREGATE DATASET FROM {path}")
    print("=" * 60)

    if not os.path.exists(path):
        print(f"Warning: File not found at {path}. Please check the path.")
        if os.path.exists("finalapi.csv"):
             path = "finalapi.csv"
             print(f"Found file at root: {path}")

    try:
        df = pd.read_csv(path)
        # Handle separators just in case
        if df.shape[1] <= 1: df = pd.read_csv(path, sep=";")
        if df.shape[1] <= 1: df = pd.read_csv(path, sep="\t")
        return df
    except Exception as e:
        sys.exit(f"Error reading CSV file: {e}")

def train_and_evaluate():
    print("="*60)
    print("TRAINING ALL MODELS FOR LEADERBOARD")
    print("="*60)
    
    # 1. Load Data
    df = load_insurance_data(DATA_PATH)
    
    # --- CRITICAL FIX: REPLACE INFINITY AND NANS ---
    # Replace explicit "infinity" values (common in division errors) with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # -----------------------------------------------

    # 2. Feature Selection
    features = [
        "RETENTION_POLY_QTY", 
        "PREV_POLY_INFORCE_QTY", 
        "LOSS_RATIO", 
        "LOSS_RATIO_3YR", 
        "GROWTH_RATE_3YR"
    ]
    
    # Ensure columns exist, fill missing with 0
    for col in features:
        if col not in df.columns:
            print(f"Warning: {col} not found. Filling with 0.")
            df[col] = 0
            
    # Filter Data (Only drop rows where we can't calculate the target)
    if "RETENTION_POLY_QTY" not in df.columns or "PREV_POLY_INFORCE_QTY" not in df.columns:
         sys.exit("CRITICAL ERROR: 'RETENTION_POLY_QTY' or 'PREV_POLY_INFORCE_QTY' missing.")
         
    df = df.dropna(subset=["RETENTION_POLY_QTY", "PREV_POLY_INFORCE_QTY"]).copy()
    
    # Target Engineering
    df['policy_lapse'] = (df['RETENTION_POLY_QTY'] < df['PREV_POLY_INFORCE_QTY']).astype(int)
    df = df[df['PREV_POLY_INFORCE_QTY'] > 0].copy()

    print(f"✓ Loaded dataset with shape: {df.shape}")
    print(f"✓ Target column 'policy_lapse' engineered.")

    # Prepare Training Data
    X = df[features].copy()
    y = df["policy_lapse"].astype(int)
    
    # --- FINAL SAFETY NET: FILL ALL REMAINING NANS WITH 0 ---
    # This prevents the "Input X contains NaN" error
    X = X.fillna(0)
    # --------------------------------------------------------

    # Save Feature Order
    os.makedirs("models", exist_ok=True)
    joblib.dump(features, FEATURE_ORDER_PATH)

    # Split & Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)
    
    # Balance
    print("Balancing data...")
    try:
        smote = SMOTEENN(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
    except Exception as e:
        print(f"SMOTEENN skipped due to error: {e}. Using unbalanced data.")
        X_train_bal, y_train_bal = X_train_scaled, y_train
        
    # Define Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    leaderboard = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        try:
            model.fit(X_train_bal, y_train_bal)
            
            y_pred = model.predict(X_test_scaled)
            try:
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
                auc = roc_auc_score(y_test, y_proba)
            except:
                auc = 0.5
                
            leaderboard[name] = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
                "auc": float(auc)
            }
        except Exception as e:
            print(f"Error training {name}: {e}")
            leaderboard[name] = {"accuracy": 0, "error": str(e)}
        
    # Save Leaderboard
    with open(LEADERBOARD_PATH, 'w') as f:
        json.dump(leaderboard, f, indent=4)
        
    # Save XGBoost as the main model
    if "XGBoost" in models:
        joblib.dump(models["XGBoost"], MODEL_PATH)

    print(f"✓ Leaderboard saved to {LEADERBOARD_PATH}")
    print(json.dumps(leaderboard, indent=4))

if __name__ == "__main__":
    train_and_evaluate()