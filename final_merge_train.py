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

# =========================================================
# 🔴 STEP 1: ENTER YOUR FILES AND JOIN COLUMN HERE
# =========================================================
FILE_1 = "finalapi.csv"   # <--- Replace with your 1st file name
FILE_2 = "kaggle.csv"   # <--- Replace with your 2nd file name
JOIN_ID = "Agent_ID"               # <--- The column common to both files

# =========================================================
# 🔴 STEP 2: CHECK THESE COLUMN MAPPINGS
# The Left side is what the model needs.
# The Right side is the column name in YOUR CSV.
# Update the Right side if your columns are named differently.
# =========================================================
COLUMN_MAP = {
    # DEMOGRAPHICS (Likely in File 2)
    "AGE": "Age",
    "PREMIUM": "Premium",
    "TENURE": "Tenure",
    
    # CHANNELS (Likely in File 2)
    "AGENT_CHANNEL": "Agent_Channel",
    "DIGITAL_CHANNEL": "Digital_Channel",
    "BANCASSURANCE": "Bancassurance",
    
    # PERFORMANCE (Likely in File 1)
    "RETENTION_POLY_QTY": "Retention_Poly_Qty",     # Needed for Target
    "PREV_POLY_INFORCE_QTY": "Prev_Poly_Inforce_Qty", # Needed for Target
    "LOSS_RATIO": "Loss_Ratio",
    "LOSS_RATIO_3YR": "Loss_Ratio_3Yr",
    "GROWTH_RATE_3YR": "Growth_Rate_3Yr"
}

# --- SYSTEM CONFIG ---
MODEL_DIR = "models"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_new.joblib")
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names.joblib")
LEADERBOARD_PATH = os.path.join(MODEL_DIR, "leaderboard.json")

def train_final_model():
    print("="*60)
    print("🚀 STARTING EMERGENCY TRAINING RUN")
    print("="*60)

    # 1. LOAD FILES
    print(f"Reading {FILE_1} and {FILE_2}...")
    try:
        df1 = pd.read_csv(FILE_1)
        df2 = pd.read_csv(FILE_2)
    except Exception as e:
        sys.exit(f"❌ Error reading files: {e}\nCheck the filenames in lines 18-19.")

    # 2. MERGE FILES
    print(f"Merging files on '{JOIN_ID}'...")
    try:
        # Standardize join column to string to avoid type mismatches
        df1[JOIN_ID] = df1[JOIN_ID].astype(str)
        df2[JOIN_ID] = df2[JOIN_ID].astype(str)
        
        df = pd.merge(df1, df2, on=JOIN_ID, how="inner")
        print(f"✓ Merged Data Shape: {df.shape}")
    except KeyError:
        sys.exit(f"❌ Error: The column '{JOIN_ID}' was not found in one of the files.")

    # 3. RENAME COLUMNS (Standardizing)
    # We create a reverse map to rename your CSV columns to what the code needs
    print("Mapping columns...")
    
    # Clean headers (strip spaces)
    df.columns = [c.strip() for c in df.columns]
    
    # Flexible renaming: Try to find columns even if case doesn't match
    lower_map = {k.lower(): v for k, v in COLUMN_MAP.items()}
    actual_rename = {}
    
    for col in df.columns:
        if col in COLUMN_MAP.values(): # Exact match
            target = [k for k, v in COLUMN_MAP.items() if v == col][0]
            actual_rename[col] = target
        elif col.lower() in [v.lower() for v in COLUMN_MAP.values()]: # Case-insensitive match
            # Find which target this maps to
            for key, val in COLUMN_MAP.items():
                if val.lower() == col.lower():
                    actual_rename[col] = key

    df = df.rename(columns=actual_rename)

    # 4. VERIFY DATA EXISTENCE
    print("Checking for missing data...")
    required_features = list(COLUMN_MAP.keys())
    missing = [col for col in required_features if col not in df.columns]
    
    if missing:
        print(f"⚠️ WARNING: The following columns were NOT found after merging: {missing}")
        print("   Filling them with 0 to allow training to proceed.")
        for m in missing:
            df[m] = 0
            
    # 5. TARGET ENGINEERING
    print("Creating Target Variable (Lapse Risk)...")
    if "RETENTION_POLY_QTY" in df.columns and "PREV_POLY_INFORCE_QTY" in df.columns:
        df = df.dropna(subset=["RETENTION_POLY_QTY", "PREV_POLY_INFORCE_QTY"])
        df['policy_lapse'] = (df['RETENTION_POLY_QTY'] < df['PREV_POLY_INFORCE_QTY']).astype(int)
        df = df[df['PREV_POLY_INFORCE_QTY'] > 0]
    else:
        sys.exit("❌ CRITICAL ERROR: Could not calculate Policy Lapse. \nEnsure 'RETENTION_POLY_QTY' and 'PREV_POLY_INFORCE_QTY' are mapped correctly.")

    # Select Features for Training (exclude targets)
    TRAIN_FEATURES = [
        "AGE", "PREMIUM", "TENURE", 
        "AGENT_CHANNEL", "DIGITAL_CHANNEL", "BANCASSURANCE", 
        "PREV_POLY_INFORCE_QTY", "LOSS_RATIO", "LOSS_RATIO_3YR", "GROWTH_RATE_3YR"
    ]
    
    X = df[TRAIN_FEATURES].fillna(0)
    y = df['policy_lapse'].astype(int)

    # 6. TRAIN & SAVE
    print("Training Model...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save Feature Names (Vital for UI)
    joblib.dump(TRAIN_FEATURES, FEATURE_NAMES_PATH)

    # Split & Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)

    # Balance
    try:
        smote = SMOTEENN(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
    except:
        X_train_bal, y_train_bal = X_train_scaled, y_train

    # XGBoost
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train_bal, y_train_bal)
    
    # Metrics
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Save Best Model
    joblib.dump(model, BEST_MODEL_PATH)
    
    # Save Leaderboard
    leaderboard = {"XGBoost": {"accuracy": acc, "f1_score": f1}}
    with open(LEADERBOARD_PATH, 'w') as f:
        json.dump(leaderboard, f, indent=4)

    print("\n" + "="*60)
    print(f"✅ SUCCESS! Project Saved.")
    print(f"   Model file: {BEST_MODEL_PATH}")
    print(f"   Accuracy: {acc:.2%}")
    print("="*60)

if __name__ == "__main__":
    train_final_model()