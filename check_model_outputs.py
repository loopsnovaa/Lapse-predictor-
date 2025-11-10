import joblib
import numpy as np
import pandas as pd

# Load model and scaler
MODEL = joblib.load("models/xgboost_optimized_model.joblib")
SCALER = joblib.load("models/scaler.joblib")
FEATURES = joblib.load("models/training_feature_order.joblib")

# Create some sample cases
samples = [
    {"age":22,"gender":1,"policy_type_1":3,"policy_type_2":4,"policy_amount":500000,"premium_amount":100,
     "policy_tenure_years":0,"policy_tenure_decimal":0.05,"channel1":6,"channel2":3,"channel3":0,
     "substandard_risk":1,"number_of_advance_premium":0,"initial_benefit":0},

    {"age":55,"gender":0,"policy_type_1":1,"policy_type_2":1,"policy_amount":200000,"premium_amount":5000,
     "policy_tenure_years":12,"policy_tenure_decimal":0.95,"channel1":2,"channel2":1,"channel3":1,
     "substandard_risk":0,"number_of_advance_premium":3,"initial_benefit":100000},

    {"age":30,"gender":1,"policy_type_1":4,"policy_type_2":4,"policy_amount":10000,"premium_amount":50,
     "policy_tenure_years":0.5,"policy_tenure_decimal":0.4,"channel1":6,"channel2":4,"channel3":0,
     "substandard_risk":1,"number_of_advance_premium":0,"initial_benefit":0}
]

df = pd.DataFrame(samples)
df["premium_to_benefit_ratio"] = df["premium_amount"] / (df["policy_amount"] + 1)
df["age_squared"] = df["age"] ** 2
df["premium_squared"] = df["premium_amount"] ** 2
df["benefit_squared"] = df["policy_amount"] ** 2

X = df[FEATURES]
X_scaled = SCALER.transform(X)

probas = MODEL.predict_proba(X_scaled)[:, 1]
print("\nPredicted probabilities:")
for i, p in enumerate(probas):
    print(f"Case {i+1}: {p:.4f}")
