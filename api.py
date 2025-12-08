import pandas as pd
import numpy as np
import joblib
import json
import os
import logging
from flask import Flask, request, jsonify

# --- CONFIGURATION ---
app = Flask(__name__)

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_optimized_model_new.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_new.joblib")
FEATURE_ORDER_PATH = os.path.join(MODEL_DIR, "training_feature_order_new.joblib")
LEADERBOARD_PATH = os.path.join(MODEL_DIR, "leaderboard.json")

# Global Artifacts
artifacts = {
    "model": None,
    "scaler": None,
    "features": None,
    "leaderboard": None
}

def load_artifacts():
    """Loads ML artifacts into memory on startup."""
    global artifacts
    logger.info("Loading model artifacts...")
    
    try:
        if os.path.exists(MODEL_PATH):
            artifacts["model"] = joblib.load(MODEL_PATH)
            logger.info(f"✓ Model loaded from {MODEL_PATH}")
        else:
            logger.error(f"❌ Model not found at {MODEL_PATH}")

        if os.path.exists(SCALER_PATH):
            artifacts["scaler"] = joblib.load(SCALER_PATH)
            logger.info(f"✓ Scaler loaded from {SCALER_PATH}")

        if os.path.exists(FEATURE_ORDER_PATH):
            artifacts["features"] = joblib.load(FEATURE_ORDER_PATH)
            logger.info(f"✓ Feature order loaded from {FEATURE_ORDER_PATH}")
            
        if os.path.exists(LEADERBOARD_PATH):
            with open(LEADERBOARD_PATH, 'r') as f:
                artifacts["leaderboard"] = json.load(f)
            logger.info(f"✓ Leaderboard loaded from {LEADERBOARD_PATH}")

    except Exception as e:
        logger.error(f"❌ Critical Error loading artifacts: {e}")

# Load immediately
load_artifacts()

# --- ROUTES ---

@app.route('/', methods=['GET'])
def index():
    """Root endpoint to verify API is running."""
    return jsonify({
        "service": "ChurnAlyse AI API",
        "status": "active",
        "version": "1.1.0"
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check for container orchestration."""
    model_status = artifacts["model"] is not None
    return jsonify({
        "status": "healthy" if model_status else "degraded",
        "artifacts_loaded": {
            "model": model_status,
            "scaler": artifacts["scaler"] is not None,
            "features": artifacts["features"] is not None
        }
    }), 200 if model_status else 503

@app.route('/leaderboard', methods=['GET'])
def get_leaderboard():
    """Returns metrics for all trained models."""
    if artifacts["leaderboard"]:
        return jsonify(artifacts["leaderboard"])
    return jsonify({"error": "Leaderboard data not available. Run training script first."}), 404

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main inference endpoint.
    Expects JSON payload: Single dict or List of dicts.
    """
    # 1. Validation
    if not artifacts["model"] or not artifacts["scaler"] or not artifacts["features"]:
        return jsonify({"error": "Model artifacts not fully loaded on server."}), 503

    try:
        # 2. Parse Input
        data = request.get_json()
        if not data:
            return jsonify({"error": "Empty payload."}), 400
            
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)

        # 3. Preprocessing (Align with Training Schema)
        # Ensure all training features exist, fill missing with 0
        for col in artifacts["features"]:
            if col not in df.columns:
                df[col] = 0

        # Sort columns strictly by training order
        df_sorted = df[artifacts["features"]]
        
        # Scale
        X_scaled = artifacts["scaler"].transform(df_sorted)
        
        # 4. Inference
        predictions = artifacts["model"].predict(X_scaled)
        
        # Handle probability (check if model supports it)
        try:
            probabilities = artifacts["model"].predict_proba(X_scaled)[:, 1]
        except:
            # Fallback for models without predict_proba (e.g. SVM/Ridge)
            probabilities = [1.0 if p == 1 else 0.0 for p in predictions]

        # 5. Result Formatting
        results = []
        for i in range(len(predictions)):
            is_lapse = int(predictions[i])
            prob = float(probabilities[i])
            
            # Dynamic Explanation Logic
            # We check the specific row for the logical driver
            retention = df_sorted.iloc[i].get('RETENTION_POLY_QTY', 0)
            prev = df_sorted.iloc[i].get('PREV_POLY_INFORCE_QTY', 0)
            
            reason = "Stable metrics. No immediate risk detected."
            if is_lapse == 1:
                if retention < prev:
                    reason = f"Retention Qty ({retention}) < Previous Qty ({prev})"
                else:
                    reason = "High predicted probability based on agency metrics."

            results.append({
                "prediction": "LAPSE" if is_lapse == 1 else "RETAIN",
                "risk_probability": round(prob, 4),
                "primary_driver": reason
            })

        return jsonify({"results": results})

    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run on port 5000 (Default for Flask)
    print("="*40)
    print("🚀 API STARTED ON http://localhost:5000")
    print("="*40)
    app.run(host='0.0.0.0', port=5000, debug=True)