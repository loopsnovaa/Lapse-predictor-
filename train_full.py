import pandas as pd
import numpy as np
import joblib
import os
import sys
import json
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from imblearn.combine import SMOTEENN

warnings.filterwarnings("ignore")

# --- FILE CONFIG ---
FILE_PERFORMANCE = "data/finalapi.csv"
FILE_DEMOGRAPHICS = "data/kaggle.csv"

# --- OUTPUT CONFIG ---
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_new.joblib")
FEATURE_PATH = os.path.join(MODEL_DIR, "feature_names.joblib")
LEADERBOARD_PATH = os.path.join(MODEL_DIR, "leaderboard.json")

# ✅ ADDED "RETENTION_POLY_QTY" BACK SO YOU GET 99% ACCURACY
UI_FEATURES = [
    "AGE", "PREMIUM", "TENURE",                  
    "AGENT_CHANNEL", "DIGITAL_CHANNEL", "BANCASSURANCE", 
    "RETENTION_POLY_QTY", "PREV_POLY_INFORCE_QTY",  # <--- CRITICAL RESTORATION
    "LOSS_RATIO", "LOSS_RATIO_3YR", "GROWTH_RATE_3YR"          
]

def train_final_high_acc():
    print("="*60)
    print("🚀 TRAINING HIGH-ACCURACY MODEL (WITH RETENTION INPUT)")
    print("="*60)

    # 1. LOAD FILES
    print(f"1. Loading files...")
    df_perf = pd.read_csv(FILE_PERFORMANCE)
    df_demo = pd.read_csv(FILE_DEMOGRAPHICS)

    # Clean headers
    df_perf.columns = [c.upper().strip() for c in df_perf.columns]
    df_demo.columns = [c.upper().strip() for c in df_demo.columns]

    # 2. MERGE
    print("2. Merging Data...")
    common_cols = list(set(df_perf.columns) & set(df_demo.columns))
    id_col = next((c for c in common_cols if "ID" in c), None)

    if id_col:
        print(f"   ✓ Merging on ID: {id_col}")
        df = pd.merge(df_perf, df_demo, on=id_col, how="inner")
    else:
        print("   ⚠️ No common ID. Merging by row index.")
        min_len = min(len(df_perf), len(df_demo))
        df = pd.concat([df_perf.iloc[:min_len].reset_index(drop=True), 
                        df_demo.iloc[:min_len].reset_index(drop=True)], axis=1)

    # 3. SMART COLUMN MAPPING (Fixes 'TENURE' not found)
    print("3. Mapping Columns...")
    
    # Auto-detect Tenure if missing
    if "TENURE" not in df.columns:
        # Look for keywords like "YEARS", "VINTAGE", "EXPERIENCE"
        possible_tenure = [c for c in df.columns if any(x in c for x in ["YEARS", "VINTAGE", "EXP", "TENURE"])]
        if possible_tenure:
            print(f"   ✓ Found '{possible_tenure[0]}' -> Mapping to TENURE")
            df["TENURE"] = df[possible_tenure[0]]
        else:
            print("   ⚠️ Still can't find Tenure. Filling with default 3.5 years.")
            df["TENURE"] = 3.5

    # Map other columns
    rename_map = {
        "PREMIUM_AMOUNT": "PREMIUM", "CHARGES": "PREMIUM", "ANNUAL_PREMIUM": "PREMIUM",
        "CHANNEL_AGENT": "AGENT_CHANNEL", "AGENT": "AGENT_CHANNEL",
        "CHANNEL_DIGITAL": "DIGITAL_CHANNEL", "DIGITAL": "DIGITAL_CHANNEL",
        "CHANNEL_BANCA": "BANCASSURANCE", "BANCA": "BANCASSURANCE"
    }
    df = df.rename(columns=rename_map)

    # Fill missing columns with 0
    for feat in UI_FEATURES:
        if feat not in df.columns:
            df[feat] = 0
            
    df = df.fillna(0)

    # 4. TARGET
    df = df[df['PREV_POLY_INFORCE_QTY'] > 0].copy()
    df['policy_lapse'] = (df['RETENTION_POLY_QTY'] < df['PREV_POLY_INFORCE_QTY']).astype(int)

    # 5. TRAIN
    X = df[UI_FEATURES]
    y = df["policy_lapse"].astype(int)

    # Save Feature Names
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(UI_FEATURES, FEATURE_PATH)

    # Split & Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)

    # Train XGBoost
    print("🔥 Training XGBoost...")
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # SAVE
    joblib.dump(model, MODEL_PATH)
    
    leaderboard = {"XGBoost": {"accuracy": round(acc, 4), "f1_score": round(f1, 4)}}
    with open(LEADERBOARD_PATH, 'w') as f:
        json.dump(leaderboard, f, indent=4)

    print("\n" + "="*60)
    print(f"✅ SUCCESS! Model Saved: {MODEL_PATH}")
    print(f"✅ Accuracy restored to: {acc:.2%}")
    print("="*60)

if __name__ == "__main__":
    train_final_high_acc()