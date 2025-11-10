# debug_full_check.py
import joblib, os, time, numpy as np, pandas as pd, sys
from pathlib import Path

MODEL_PATH = "models/xgboost_optimized_model.joblib"
SCALER_PATH = "models/scaler.joblib"
FEATURE_ORDER_PATH = "models/training_feature_order.joblib"
PROC_CSV = "data/kaggle_processed_simple.csv"

def fmt_ts(p):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p)))

print("== Files and timestamps ==")
for p in [MODEL_PATH, SCALER_PATH, FEATURE_ORDER_PATH, PROC_CSV]:
    if os.path.exists(p):
        print(f"{p}: exists, modified: {fmt_ts(p)}, size: {os.path.getsize(p):,} bytes")
    else:
        print(f"{p}: MISSING")

print("\n== Load model and scaler ==")
try:
    model = joblib.load(MODEL_PATH)
    print("Loaded model object type:", type(model))
    # If it's a wrapper like CalibratedClassifierCV, show underlying estimator type
    try:
        base = getattr(model, "base_estimator_", getattr(model, "estimator", None))
        if base is not None:
            print("Underlying estimator type:", type(base))
    except Exception:
        pass
    print("model.classes_:", getattr(model, "classes_", None))
except Exception as e:
    print("FAILED loading model:", e)
    sys.exit(1)

try:
    scaler = joblib.load(SCALER_PATH)
    print("Loaded scaler type:", type(scaler))
    print("Scaler mean length:", len(getattr(scaler, "mean_", [])))
    print("Scaler mean (first 8):", getattr(scaler, "mean_", [])[:8])
    print("Scaler var (first 8):", getattr(scaler, "var_", [])[:8])
except Exception as e:
    print("FAILED loading scaler:", e)
    sys.exit(1)

print("\n== Feature order ==")
if os.path.exists(FEATURE_ORDER_PATH):
    feat = joblib.load(FEATURE_ORDER_PATH)
    print("FEATURE count:", len(feat))
    print("FEATURES (first 20):", feat[:20])
else:
    print("No training_feature_order.joblib found")

print("\n== Quick sample predictions (3 cases) ==")
samples = [
    {"age":22,"gender":1,"policy_type_1":3,"policy_type_2":4,"policy_amount":500000.0,"premium_amount":100.0,"policy_tenure_years":0.0,"policy_tenure_decimal":0.05,"channel1":6,"channel2":3,"channel3":0,"substandard_risk":1.0,"number_of_advance_premium":0,"initial_benefit":0.0},
    {"age":55,"gender":0,"policy_type_1":1,"policy_type_2":1,"policy_amount":200000.0,"premium_amount":5000.0,"policy_tenure_years":12.0,"policy_tenure_decimal":0.95,"channel1":2,"channel2":1,"channel3":1,"substandard_risk":0.0,"number_of_advance_premium":3,"initial_benefit":100000.0},
    {"age":30,"gender":1,"policy_type_1":4,"policy_type_2":4,"policy_amount":10000.0,"premium_amount":50.0,"policy_tenure_years":0.5,"policy_tenure_decimal":0.4,"channel1":6,"channel2":4,"channel3":0,"substandard_risk":1.0,"number_of_advance_premium":0,"initial_benefit":0.0}
]

# Build DataFrame with engineered features using same logic as API
df = pd.DataFrame(samples)
df["premium_to_benefit_ratio"] = df["premium_amount"] / (df["policy_amount"] + 1)
df["age_squared"] = df["age"] ** 2
df["premium_squared"] = df["premium_amount"] ** 2
df["benefit_squared"] = df["policy_amount"] ** 2

# Align columns
if os.path.exists(FEATURE_ORDER_PATH):
    X = df[feat]
else:
    # fallback: use API default order if feature list missing
    fallback = ['age','gender','policy_type_1','policy_type_2','policy_amount','premium_amount','policy_tenure_years','policy_tenure_decimal','channel1','channel2','channel3','substandard_risk','number_of_advance_premium','initial_benefit','premium_to_benefit_ratio','age_squared','premium_squared','benefit_squared']
    X = df[fallback]

print("X raw (first row):")
print(X.iloc[0].to_dict())

# Scale
X_scaled = scaler.transform(X.astype(float))
print("\nX_scaled (first row, first 12 values):", X_scaled[0][:12].tolist())

# Predict probabilities
probas = model.predict_proba(X_scaled)[:, 1]
print("\nPredicted probabilities for 3 cases:", [float(round(p,6)) for p in probas])

# Check global distribution on processed CSV (if exists)
if os.path.exists(PROC_CSV):
    dproc = pd.read_csv(PROC_CSV)
    feature_cols = [c for c in dproc.columns if c not in ("policy_id","policy_lapse")]
    Xall = dproc[feature_cols].astype(float)
    Xall_scaled = scaler.transform(Xall)
    all_probas = model.predict_proba(Xall_scaled)[:, 1]
    print("\nAll_probas stats: min, mean, max:", float(all_probas.min()), float(all_probas.mean()), float(all_probas.max()))
    print("Percentiles (5,25,50,75,95):", np.percentile(all_probas, [5,25,50,75,95]).tolist())
else:
    print("\nProcessed CSV not found; skipping distribution check")
