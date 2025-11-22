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
import numpy as np
import joblib
import os
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore")


DATA_PATH = "data/finalapi.csv" 
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
        if os.path.exists("finalapi.csv"):
             path = "finalapi.csv"
             print(f"Found file at root: {path}")


    try:
        df = pd.read_csv(path)
    except Exception as e:
        sys.exit(f"Error reading CSV file: {e}")

# --- CONFIGURATION ---
DATA_PATH = "data/finalapi.csv"
LEADERBOARD_PATH = "models/leaderboard.json"

def train_and_evaluate():
    print("="*60)
    print("TRAINING ALL MODELS FOR LEADERBOARD")
    print("="*60)
    
    # 1. Load Data
    try:
        df = pd.read_csv(DATA_PATH)
    except:
        try: df = pd.read_csv(DATA_PATH, sep=";")
        except: df = pd.read_csv(DATA_PATH, sep="\t")
    

    df = df.replace(99999, np.nan)
    
    # 2. Feature Selection (High Accuracy Set)
    features = [
        "RETENTION_POLY_QTY", 
        "PREV_POLY_INFORCE_QTY", 
        "LOSS_RATIO", 
        "LOSS_RATIO_3YR", 
        "GROWTH_RATE_3YR"
    ]
    df = df.dropna(subset=required_cols).copy()
    df['policy_lapse'] = (df['RETENTION_POLY_QTY'] < df['PREV_POLY_INFORCE_QTY']).astype(int)

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
    joblib.dump(feature_cols, FEATURE_ORDER_PATH)
    print(f"✓ Saved feature order to {FEATURE_ORDER_PATH}")

    # Clean Data
    df = df.dropna(subset=["RETENTION_POLY_QTY", "PREV_POLY_INFORCE_QTY"])
    df[features] = df[features].fillna(0)
    
    # Target
    df['policy_lapse'] = (df['RETENTION_POLY_QTY'] < df['PREV_POLY_INFORCE_QTY']).astype(int)
    df = df[df['PREV_POLY_INFORCE_QTY'] > 0]
    
    X = df[features]
    y = df["policy_lapse"].astype(int)
    
    # Split & Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Balance
    print("Balancing data...")
    try:
        smote = SMOTEENN(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
    except:
        X_train_bal, y_train_bal = X_train_scaled, y_train
        
    # Define Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost (Tuned)": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    leaderboard = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_bal, y_train_bal)
        
        y_pred = model.predict(X_test_scaled)
        try:
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = 0.5
            
        leaderboard[name] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred)),
            "auc": float(auc)
        }
        
    # Save
    os.makedirs("models", exist_ok=True)
    import json
    with open(LEADERBOARD_PATH, 'w') as f:
        json.dump(leaderboard, f)
        
    print(f"✓ Leaderboard saved to {LEADERBOARD_PATH}")
    print(leaderboard)

if __name__ == "__main__":
    main()
    train_and_evaluate()

