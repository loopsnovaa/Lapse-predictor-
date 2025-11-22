import pandas as pd
import numpy as np
import joblib
import json
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = "models/xgboost_optimized_model_new.joblib"
SCALER_PATH = "models/scaler_new.joblib"
FEATURE_ORDER_PATH = "models/training_feature_order_new.joblib"
METRICS_PATH = "models/model_metrics_new.joblib"
LEADERBOARD_PATH = "models/leaderboard.json"

# Global variables
model = None
scaler = None
feature_order = None
metrics = None
leaderboard_data = None

def load_artifacts():
    global model, scaler, feature_order, metrics, leaderboard_data
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"✓ Model loaded")
        
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            print(f"✓ Scaler loaded")

        if os.path.exists(FEATURE_ORDER_PATH):
            feature_order = joblib.load(FEATURE_ORDER_PATH)
            print(f"✓ Feature order loaded")
            
        if os.path.exists(METRICS_PATH):
            metrics = joblib.load(METRICS_PATH)
            
        if os.path.exists(LEADERBOARD_PATH):
            with open(LEADERBOARD_PATH, 'r') as f:
                leaderboard_data = json.load(f)
            print(f"✓ Leaderboard data loaded")

    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")

load_artifacts()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy" if model else "degraded",
        "model_loaded": model is not None,
        "training_metrics": metrics
    })

@app.route('/leaderboard', methods=['GET'])
def get_leaderboard():
    """Returns metrics for all 4 models"""
    if leaderboard_data:
        return jsonify(leaderboard_data)
    else:
        return jsonify({"error": "Leaderboard data not generated yet."}), 404

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler or not feature_order:
        return jsonify({"error": "Model artifacts not fully loaded."}), 500

    try:
        data = request.get_json()
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)

        # Validation & fill missing
        for col in feature_order:
            if col not in df.columns:
                df[col] = 0

        df_sorted = df[feature_order]
        X_scaled = scaler.transform(df_sorted)
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]

        results = []
        for i in range(len(predictions)):
            is_lapse = int(predictions[i])
            prob = float(probabilities[i])
            
            # Explanation
            retention = df_sorted.iloc[i]['RETENTION_POLY_QTY']
            prev = df_sorted.iloc[i]['PREV_POLY_INFORCE_QTY']
            
            reason = "Stable retention metrics."
            if is_lapse == 1:
                reason = f"Retention Qty ({retention}) < Previous Qty ({prev})."
                
            results.append({
                "prediction": "LAPSE" if is_lapse == 1 else "RETAIN",
                "confidence_score": round(prob, 4),
                "primary_driver": reason
            })

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)