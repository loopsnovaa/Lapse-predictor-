import json
import os
from datetime import datetime
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Load Model, Scaler, and Feature Order
# ------------------------------------------------------------
MODEL_PATH = "models/xgboost_optimized_model.joblib"
SCALER_PATH = "models/scaler.joblib"
FEATURE_ORDER_PATH = "models/training_feature_order.joblib"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_order = joblib.load(FEATURE_ORDER_PATH)

# ------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
PREDICTION_LOG = "logs/predictions.log"
FEEDBACK_LOG = "logs/feedback.log"

def log_prediction(data):
    with open(PREDICTION_LOG, "a") as f:
        f.write(json.dumps(data) + "\n")

def log_feedback(data):
    with open(FEEDBACK_LOG, "a") as f:
        f.write(json.dumps(data) + "\n")

# ------------------------------------------------------------
# Flask App
# ------------------------------------------------------------
app = Flask(__name__)

# ------------------------------------------------------------
# Health Check
# ------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "model_type": "XGBoost (14 feature model)"
    })

# ------------------------------------------------------------
# Prediction Endpoint
# ------------------------------------------------------------
@app.route("/predict_lapse", methods=["POST"])
def predict_lapse():
    try:
        user_input = request.get_json()

        # ensure correct feature order
        df = pd.DataFrame([user_input], columns=feature_order)

        # scale input
        scaled = scaler.transform(df)

        # model prediction
        proba = float(model.predict_proba(scaled)[0][1])   # probability of lapse

        # classify
        if proba < 0.30:
            risk = "Low"
        elif proba < 0.70:
            risk = "Medium"
        else:
            risk = "High"

        # log the prediction
        record = {
            "timestamp": datetime.now().isoformat(),
            "input": user_input,
            "predicted_probability": proba,
            "risk_level": risk
        }
        log_prediction(record)

        return jsonify({
            "status": "success",
            "model_used": "Optimized XGBoost (14F)",
            "policy_risk_score": round(proba, 4),
            "lapse_probability_percent": round(proba * 100, 2),
            "risk_level": risk
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ------------------------------------------------------------
# Feedback Endpoint
# ------------------------------------------------------------
@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json()
    data["timestamp"] = datetime.now().isoformat()
    log_feedback(data)
    return jsonify({"status": "success", "message": "Feedback stored"})

# ------------------------------------------------------------
# Model Stats Endpoint
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
                continue

    if not records:
        return jsonify({"status": "error", "message": "Log file empty"}), 404

    probabilities = [r["predicted_probability"] for r in records]
    risk_levels = [r["risk_level"] for r in records]

    stats = {
        "total_predictions": len(probabilities),
        "average_predicted_risk": float(np.mean(probabilities)),
        "high_risk_count": risk_levels.count("High"),
        "medium_risk_count": risk_levels.count("Medium"),
        "low_risk_count": risk_levels.count("Low"),
        "saved_customers_estimate": risk_levels.count("High") + risk_levels.count("Medium")
    }

    return jsonify(stats)

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    print("API running at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000)
