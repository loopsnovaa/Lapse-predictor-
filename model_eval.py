# model_eval.py
import joblib, pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score, classification_report

MODEL_PATH = "models/xgboost_optimized_model.joblib"
SCALER_PATH = "models/scaler.joblib"
FEAT_ORDER_PATH = "models/training_feature_order.joblib"
PROC_CSV = "data/kaggle_processed_simple.csv"

m = joblib.load(MODEL_PATH)
s = joblib.load(SCALER_PATH)

# Load processed data
df = pd.read_csv(PROC_CSV)
if 'policy_lapse' not in df.columns:
    raise SystemExit("policy_lapse column missing in processed CSV")

# Load feature order if present, else infer from df (drop id/target)
try:
    feat_order = joblib.load(FEAT_ORDER_PATH)
except:
    feat_order = [c for c in df.columns if c not in ('policy_lapse','policy_id')]

X = df[feat_order].astype(float)
y = df['policy_lapse'].astype(int).values

# Scale
X_scaled = s.transform(X)

# Predict
probas = m.predict_proba(X_scaled)[:, 1]
preds = (probas > 0.5).astype(int)

# Metrics
auc = roc_auc_score(y, probas)
report = classification_report(y, preds, digits=4)
mean_pos = probas[y == 1].mean() if sum(y==1)>0 else float('nan')
mean_neg = probas[y == 0].mean() if sum(y==0)>0 else float('nan')

print("Model classes_:", getattr(m, "classes_", None))
print(f"Dataset size: {len(df):,}")
print(f"Positive (lapse) count: {sum(y==1):,}, Negative count: {sum(y==0):,}")
print(f"AUC: {auc:.4f}")
print(f"Mean predicted prob (y==1): {mean_pos:.6f}")
print(f"Mean predicted prob (y==0): {mean_neg:.6f}")
print("\nClassification report (threshold 0.5):")
print(report)
