import json
import os
from datetime import datetime
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from flask_cors import CORS
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# LOAD MODEL + SCALER + FEATURE ORDER + METRICS
# ============================================================

MODEL_PATH = "models/xgboost_optimized_model.joblib"
SCALER_PATH = "models/scaler.joblib"
FEATURE_ORDER_PATH = "models/training_feature_order.joblib"
METRICS_PATH = "models/model_metrics.joblib"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_order = joblib.load(FEATURE_ORDER_PATH)
MODEL_METRICS = joblib.load(METRICS_PATH)

# ============================================================
# LOGGING SETUP
# ============================================================

os.makedirs("logs", exist_ok=True)
PREDICTION_LOG = "logs/predictions.log"
FEEDBACK_LOG = "logs/feedback.log"

def log_prediction(data):
    with open(PREDICTION_LOG, "a") as f:
        f.write(json.dumps(data) + "\n")

def log_feedback(data):
    with open(FEEDBACK_LOG, "a") as f:
        f.write(json.dumps(data) + "\n")

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)
CORS(app)

# ============================================================
# PREPROCESS — SAME AS TRAINING
# ============================================================

def preprocess_input(data_dict):
    df = pd.DataFrame([data_dict])

    df["premium_to_benefit_ratio"] = df["premium_amount"] / (df["policy_amount"] + 1)
    df["age_squared"] = df["age"] ** 2
    df["premium_squared"] = df["premium_amount"] ** 2
    df["benefit_squared"] = df["policy_amount"] ** 2

    X = df[feature_order]
    X_scaled = scaler.transform(X)
    return X_scaled

# ============================================================
# ROUTES
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": True})


# ------------------------------------------------------------
# PREDICTION ENDPOINT
# ------------------------------------------------------------
@app.route("/predict_lapse", methods=["POST"])
def predict_lapse():
    try:
        data = request.get_json(force=True)

        # Preprocess
        X_scaled = preprocess_input(data)

        # Predict probability
        proba = float(model.predict_proba(X_scaled)[0][1])

        # Risk classification
        if proba < 0.30:
            risk = "Low"
        elif proba < 0.70:
            risk = "Medium"
        else:
            risk = "High"

        # Log the prediction
        record = {
            "timestamp": datetime.now().isoformat(),
            "input": data,
            "predicted_probability": proba,
            "risk_level": risk
        }
        log_prediction(record)

        return jsonify({
            "status": "success",
            "model_used": "Optimized XGBoost",
            "policy_risk_score": round(proba, 4),
            "lapse_probability_percent": round(proba * 100, 2),
            "risk_level": risk
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# ------------------------------------------------------------
# FEEDBACK ENDPOINT
# ------------------------------------------------------------
@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json()
    data["timestamp"] = datetime.now().isoformat()
    log_feedback(data)
    return jsonify({"status": "success", "message": "Feedback stored"})


# ------------------------------------------------------------
# MODEL STATS (FIXED WITH ACCURACY + METRICS)
# ------------------------------------------------------------
@app.route("/model_stats", methods=["GET"])
def model_stats():

    if not os.path.exists(PREDICTION_LOG):
        return jsonify({"status": "error", "message": "No logs found"}), 404

    records = []
    with open(PREDICTION_LOG, "r") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except:
                pass

    if not records:
        return jsonify({"status": "error", "message": "Log file empty"}), 404

    probs = [r["predicted_probability"] for r in records]
    risks = [r["risk_level"] for r in records]

    stats = {
        "total_predictions": len(probs),
        "average_predicted_risk": float(np.mean(probs)),
        "low_risk_count": risks.count("Low"),
        "medium_risk_count": risks.count("Medium"),
        "high_risk_count": risks.count("High"),
        "saved_customers_estimate": risks.count("Medium") + risks.count("High"),

        # 🔥 FIXED — NOW EXPOSE MODEL METRICS TO DASHBOARD
        "accuracy": float(MODEL_METRICS.get("accuracy", 0)),
        "precision": float(MODEL_METRICS.get("precision", 0)),
        "recall": float(MODEL_METRICS.get("recall", 0)),
        "f1_score": float(MODEL_METRICS.get("f1_score", 0)),
        "auc": float(MODEL_METRICS.get("auc", 0)),
    }

    return jsonify(stats)


# ============================================================
# RUN FLASK
# ============================================================
if __name__ == "__main__":
    print("API running at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000)
