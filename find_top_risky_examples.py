# find_top_risky_examples.py
import joblib, pandas as pd, numpy as np, os, json

MODEL_PATH = "models/xgboost_optimized_model.joblib"
SCALER_PATH = "models/scaler.joblib"
PROC_CSV = "data/kaggle_processed_simple.csv"
FEATURE_ORDER_PATH = "models/training_feature_order.joblib"

print("Loading model, scaler and processed data...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
df = pd.read_csv(PROC_CSV)

# load feature order or infer
if os.path.exists(FEATURE_ORDER_PATH):
    feat_order = joblib.load(FEATURE_ORDER_PATH)
else:
    feat_order = [c for c in df.columns if c not in ("policy_id","policy_lapse")]

print(f"Using {len(feat_order)} features: {feat_order[:20]}...\n")

# Prepare features and engineered columns if not present
# (processed csv should already have engineered; ensure they exist)
for col in ["premium_to_benefit_ratio","age_squared","premium_squared","benefit_squared"]:
    if col not in df.columns:
        df[col] = 0

X = df[feat_order].astype(float)
X_scaled = scaler.transform(X)
probas = model.predict_proba(X_scaled)[:, 1]

df_out = df.copy()
df_out["pred_proba"] = probas

# Show global stats
print("Global probability stats (min, 5%, 25%, 50%, 75%, 95%, max):")
percentiles = np.percentile(probas, [0,5,25,50,75,95,100])
print([float(round(x,6)) for x in percentiles])
print("Mean:", float(round(probas.mean(),6)))
print("Positive label rate in processed csv:", int(df_out['policy_lapse'].sum()), "/", len(df_out))
print()

# Top 10 highest-risk rows (by model's prob)
topN = 10
top = df_out.sort_values("pred_proba", ascending=False).head(topN)
pd.set_option("display.max_columns", 999)
print(f"TOP {topN} highest-risk records (model prediction):\n")
print(top[["policy_id","policy_lapse","pred_proba"] + feat_order].head(topN).to_string(index=False))
print("\n\nFor convenience, I will now output JSON payloads (one per top-3) you can POST to the API exactly as shown.\n")

top3 = top.head(3)
for i, row in top3.iterrows():
    payload = {k: float(row[k]) if k in feat_order else None for k in feat_order}
    # convert any NaN to 0
    payload = {k:(0.0 if (pd.isna(v) or v is None) else v) for k,v in payload.items()}
    print(f"--- Example {i} | policy_id={int(row['policy_id'])} | proba={row['pred_proba']:.4f} | actual_lapse={int(row['policy_lapse'])} ---")
    print(json.dumps(payload, indent=2))
    print()
