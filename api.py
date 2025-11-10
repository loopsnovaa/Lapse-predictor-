import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1️⃣ LOAD MODEL + SCALER
# ============================================================

app = Flask(__name__)
CORS(app)

try:
    MODEL = joblib.load("models/xgboost_optimized_model.joblib")
    SCALER = joblib.load("models/scaler.joblib")

    try:
        FEATURE_NAMES = joblib.load("models/training_feature_order.joblib")
        print(f"✓ Loaded feature order from training_feature_order.joblib ({len(FEATURE_NAMES)} features)")
    except:
        FEATURE_NAMES = [
            'age', 'gender', 'policy_type_1', 'policy_type_2',
            'policy_amount', 'premium_amount', 'policy_tenure_years',
            'policy_tenure_decimal', 'channel1', 'channel2', 'channel3',
            'substandard_risk', 'number_of_advance_premium', 'initial_benefit',
            'premium_to_benefit_ratio', 'age_squared', 'premium_squared', 'benefit_squared'
        ]
        print("⚠️ No training_feature_order.joblib found — using default feature order")

    print("✓ Model and Scaler loaded successfully.")

except Exception as e:
    print(f"❌ ERROR loading model/scaler: {e}")
    sys.exit(1)

# ============================================================
# 2️⃣ PREPROCESS FUNCTION
# ============================================================

def preprocess_input(data_dict):
    """
    Apply the same preprocessing and scaling as training.
    """
    df = pd.DataFrame([data_dict])

    # Feature engineering (must match training)
    df["premium_to_benefit_ratio"] = df["premium_amount"] / (df["policy_amount"] + 1)
    df["age_squared"] = df["age"] ** 2
    df["premium_squared"] = df["premium_amount"] ** 2
    df["benefit_squared"] = df["policy_amount"] ** 2

    # Align order
    X = df[FEATURE_NAMES]

    # Scale
    X_scaled = SCALER.transform(X)
    return X_scaled


# ============================================================
# 3️⃣ ROUTES
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": True})


@app.route("/predict_lapse", methods=["POST"])
def predict_lapse():
    try:
        data = request.get_json(force=True)

        # Preprocess
        X_scaled = preprocess_input(data)

        # Predict probability
        prob = float(MODEL.predict_proba(X_scaled)[:, 1][0])

        # Risk classification
        if prob < 0.3:
            risk_level = "Low"
        elif prob < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"

        print(f"[PREDICT] Probability={prob:.4f} → Risk={risk_level}")

        return jsonify({
            "status": "success",
            "policy_risk_score": round(prob, 4),
            "lapse_probability_percent": round(prob * 100, 2),
            "risk_level": risk_level,
            "model_used": "Optimized XGBoost"
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Prediction failed: {e}"
        }), 400


# ============================================================
# 4️⃣ RUN
# ============================================================

if __name__ == "__main__":
    print("API starting on http://127.0.0.1:5000")
    app.run(debug=False)
