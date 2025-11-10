# debug_predict.py (robust)
import joblib, numpy as np, pandas as pd, os, json

MODEL_PATH = "models/xgboost_optimized_model.joblib"
SCALER_PATH = "models/scaler.joblib"
FEATURE_ORDER_PATH = "models/training_feature_order.joblib"
PROC_CSV = "data/kaggle_processed_simple.csv"

print("Loading model and scaler...")
m = joblib.load(MODEL_PATH)
s = joblib.load(SCALER_PATH)

print("Model n_features_in_:", getattr(m, "n_features_in_", None))
print("Scaler mean length:", len(getattr(s, "mean_", [])))

# load feature order if exists
if os.path.exists(FEATURE_ORDER_PATH):
    feat_order = joblib.load(FEATURE_ORDER_PATH)
    print("Loaded FEATURE_NAMES from training:", feat_order)
else:
    feat_order = [
        'age','gender','policy_type_1','policy_type_2','policy_amount',
        'premium_amount','policy_tenure_years','policy_tenure_decimal',
        'channel1','channel2','channel3','substandard_risk',
        'number_of_advance_premium','initial_benefit',
        'premium_to_benefit_ratio','age_squared','premium_squared','benefit_squared'
    ]
    print("No training_feature_order.joblib found; using fallback FEATURE_NAMES")

print("Feature count used:", len(feat_order))

# Construct the "high-risk" sample you tested
sample = {
    "age": 22,
    "gender": "M",   # string allowed; script will normalize
    "policy_type_1": 3,
    "policy_type_2": 4,
    "policy_amount": 500000.0,
    "premium_amount": 100.0,
    "policy_tenure_years": 0.0,
    "policy_tenure_decimal": 0.05,
    "channel1": 6,
    "channel2": 3,
    "channel3": 0,
    "substandard_risk": 1.0,
    "number_of_advance_premium": 0,
    "initial_benefit": 0.0
}

# normalize gender if string
g = sample.get("gender")
if isinstance(g, str):
    g2 = g.strip().lower()
    if g2 in ("m","male"):
        sample["gender"] = 1.0
    elif g2 in ("f","female"):
        sample["gender"] = 0.0
    else:
        try:
            sample["gender"] = float(g)
        except:
            sample["gender"] = 0.0

# create df and engineered features same as API
df = pd.DataFrame([sample])
df['premium_to_benefit_ratio'] = df['premium_amount'] / (df['policy_amount'] + 1.0)
df['age_squared'] = df['age'] ** 2
df['premium_squared'] = df['premium_amount'] ** 2
df['benefit_squared'] = df['policy_amount'] ** 2

# Ensure columns in same order
X = df[feat_order].astype(float)
print("\nX (raw) columns and values:")
for c, v in zip(X.columns, X.iloc[0].tolist()):
    print(f"  {c:30}: {v}")

# scale
X_scaled = s.transform(X)
print("\nX_scaled (first 12 values):", X_scaled[0][:12].tolist())

# model prediction
proba = float(m.predict_proba(X_scaled)[:,1][0])
pred = m.predict(X_scaled)[0]
print("\nModel predict_proba positive class:", proba)
print("Model predict (class):", int(pred))

# show top feature importances from model
try:
    importances = getattr(m, "feature_importances_", None)
    if importances is not None:
        fi = sorted(zip(feat_order, importances), key=lambda x: x[1], reverse=True)
        print("\nTop 10 feature importances (model):")
        for name, imp in fi[:10]:
            print(f"  {name:30}: {imp:.6f}")
    else:
        print("\nModel has no feature_importances_ attribute")
except Exception as e:
    print("Could not read feature importances:", e)

# show training label distribution if processed CSV exists
if os.path.exists(PROC_CSV):
    dfp = pd.read_csv(PROC_CSV)
    if 'policy_lapse' in dfp.columns:
        print("\nTraining file label distribution (from data/kaggle_processed_simple.csv):")
        print(dfp['policy_lapse'].value_counts().to_string())
    else:
        print("\nProcessed CSV present but no policy_lapse column found.")
else:
    print("\nProcessed CSV not found at:", PROC_CSV)
