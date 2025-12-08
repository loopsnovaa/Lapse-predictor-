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

# --- CRITICAL: CORRECT FILE PATHS (MATCHING TRAIN_FULL.PY) ---
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")      # Updated
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_new.joblib")
FEATURE_PATH = os.path.join(MODEL_DIR, "feature_names.joblib") # Updated
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
        # 1. Load Model
        if os.path.exists(MODEL_PATH):
            artifacts["model"] = joblib.load(MODEL_PATH)
            logger.info(f"✓ Model loaded from {MODEL_PATH}")
        else:
            logger.error(f"❌ Model not found at {MODEL_PATH}. Run train_full.py first!")

        # 2. Load Scaler
        if os.path.exists(SCALER_PATH):
            artifacts["scaler"] = joblib.load(SCALER_PATH)
            logger.info(f"✓ Scaler loaded from {SCALER_PATH}")

        # 3. Load Feature Names
        if os.path.exists(FEATURE_PATH):
            artifacts["features"] = joblib.load(FEATURE_PATH)
            logger.info(f"✓ Feature names loaded from {FEATURE_PATH}")
            
        # 4. Load Leaderboard
        if os.path.exists(LEADERBOARD_PATH):
            with open(LEADERBOARD_PATH, 'r') as f:
                artifacts["leaderboard"] = json.load(f)
            logger.info(f"✓ Leaderboard loaded")

    except Exception as e:
        logger.error(f"❌ Critical Error loading artifacts: {e}")

# Load immediately on start
load_artifacts()

# --- ROUTES ---

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "service": "ChurnAlyse AI API",
        "status": "active",
        "version": "2.0.0"
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Inference Endpoint.
    Expects JSON with: Age, Premium, Tenure, Channels, Prev Qty, Retained Qty, etc.
    """
    # 1. Validation
    if not artifacts["model"] or not artifacts["scaler"] or not artifacts["features"]:
        return jsonify({"error": "Model artifacts not loaded. Check server logs."}), 503

    try:
        # 2. Parse Input
        data = request.get_json()
        if not data:
            return jsonify({"error": "Empty payload."}), 400
            
        # Handle dict vs list
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)

        # 3. Preprocessing (Align with Training Schema)
        # Ensure all training features exist, fill missing with 0
        model_features = artifacts["features"]
        
        for col in model_features:
            if col not in df.columns:
                df[col] = 0

        # Sort columns strictly by training order
        df_sorted = df[model_features]
        
        # Scale
        X_scaled = artifacts["scaler"].transform(df_sorted)
        
        # 4. Inference
        predictions = artifacts["model"].predict(X_scaled)
        
        # Get Probabilities
        try:
            probabilities = artifacts["model"].predict_proba(X_scaled)[:, 1]
        except:
            probabilities = [1.0 if p == 1 else 0.0 for p in predictions]

        # 5. Result Formatting
        results = []
        for i in range(len(predictions)):
            is_lapse = int(predictions[i])
            prob = float(probabilities[i])
            
            # Logic Explanation
            retention = df_sorted.iloc[i].get('RETENTION_POLY_QTY', 0)
            prev = df_sorted.iloc[i].get('PREV_POLY_INFORCE_QTY', 0)
            
            reason = "Stable metrics."
            if is_lapse == 1:
                if retention < prev:
                    reason = f"Retention Gap: {prev - retention} policies lost."
                else:
                    reason = "High Lapse Probability detected by AI."

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
    print("="*40)
    print("🚀 API STARTED ON http://localhost:5000")
    print("="*40)
    app.run(host='0.0.0.0', port=5000, debug=True)