import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
import logging
from flask import Flask, request, jsonify

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")      
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_new.joblib")
FEATURE_PATH = os.path.join(MODEL_DIR, "feature_names.joblib")
LEADERBOARD_PATH = os.path.join(MODEL_DIR, "leaderboard.json")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(ROOT_DIR, 'src'))

MODEL_PATH = os.path.join(ROOT_DIR, "models", "kaggle_ensemble_model.joblib")
PREPROCESSOR_PATH = os.path.join(ROOT_DIR, "models", "kaggle_preprocessor.joblib")
LEADERBOARD_PATH = os.path.join(ROOT_DIR, "models", "leaderboard.json")

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

        try:
            from src.models.ensemble import ChurnEnsembleModel
        except ImportError:
            pass 


        if os.path.exists(MODEL_PATH):
            artifacts["model"] = joblib.load(MODEL_PATH)
            logger.info(f"✓ Model loaded from {MODEL_PATH}")
        else:
            logger.error(f"❌ Model not found at {MODEL_PATH}. Run train_full.py first!")

        if os.path.exists(SCALER_PATH):
            artifacts["scaler"] = joblib.load(SCALER_PATH)
            logger.info(f"✓ Scaler loaded from {SCALER_PATH}")

        if os.path.exists(FEATURE_PATH):
            artifacts["features"] = joblib.load(FEATURE_PATH)
            logger.info(f"✓ Feature names loaded from {FEATURE_PATH}")
            
        if os.path.exists(LEADERBOARD_PATH):
            with open(LEADERBOARD_PATH, 'r') as f:
                artifacts["leaderboard"] = json.load(f)
            logger.info(f"✓ Leaderboard loaded")
        if os.path.exists(PREPROCESSOR_PATH):
            preprocessor_data = joblib.load(PREPROCESSOR_PATH)
            
            if isinstance(preprocessor_data, dict):
                artifacts["scaler"] = preprocessor_data.get('scaler')
                artifacts["features"] = preprocessor_data.get('feature_names', [])
            else:
                artifacts["scaler"] = preprocessor_data.scaler
                artifacts["features"] = preprocessor_data.feature_names
                
            logger.info(f"✓ Preprocessor loaded (Features: {len(artifacts['features'])})")
        else:
            logger.error(f"❌ Preprocessor not found at {PREPROCESSOR_PATH}")

        if os.path.exists(LEADERBOARD_PATH):
            with open(LEADERBOARD_PATH, 'r') as f:
                artifacts["leaderboard"] = json.load(f)
            logger.info("✓ Leaderboard loaded")


    except Exception as e:
        logger.error(f"❌ Critical Error loading artifacts: {e}")

load_artifacts()


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
    if not artifacts["model"] or not artifacts["scaler"] or not artifacts["features"]:
        return jsonify({"error": "Model artifacts not loaded. Check server logs."}), 503

    """Root endpoint."""
    return jsonify({
        "service": "ChurnAlyse AI API",
        "status": "active",
        "model_loaded": artifacts["model"] is not None
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check."""
    model_status = artifacts["model"] is not None
    return jsonify({
        "status": "healthy" if model_status else "degraded",
        "artifacts": {
            "model": model_status,
            "preprocessor": artifacts["scaler"] is not None
        }
    }), 200 if model_status else 503

@app.route('/leaderboard', methods=['GET'])
def get_leaderboard():
    """Returns metrics."""
    if artifacts["leaderboard"]:
        return jsonify(artifacts["leaderboard"])
    return jsonify({"error": "Leaderboard data not available."}), 404

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main inference endpoint.
    """
    if not artifacts["model"] or not artifacts["scaler"]:
        return jsonify({"error": "Model artifacts not loaded."}), 503


    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Empty payload."}), 400
            
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)

        model_features = artifacts["features"]
        
        for col in model_features:
            if col not in df.columns:
                df[col] = 0

        df_sorted = df[model_features]
        

        final_df = pd.DataFrame()
        for col in artifacts["features"]:
            if col in df.columns:
                final_df[col] = df[col]
            else:
                final_df[col] = 0.0 

        X_scaled = artifacts["scaler"].transform(final_df)
        
        predictions = artifacts["model"].predict(X_scaled)
        
        try:
            probabilities = artifacts["model"].predict_proba(X_scaled)[:, 1]
        except:
            probabilities = [1.0 if p == 1 else 0.0 for p in predictions]

        results = []
        for i in range(len(predictions)):
            is_lapse = int(predictions[i])
            prob = float(probabilities[i])
            
            retention = df_sorted.iloc[i].get('RETENTION_POLY_QTY', 0)
            prev = df_sorted.iloc[i].get('PREV_POLY_INFORCE_QTY', 0)
            
            reason = "Stable metrics."
            if is_lapse == 1:
                if retention < prev:
                    reason = f"Retention Gap: {prev - retention} policies lost."
                else:
                    reason = "High Lapse Probability detected by AI."

            risk_level = "High" if prob > 0.5 else "Low"
            
            reason = "Stable metrics."
            if is_lapse == 1:
                reason = "High probability of lapse based on historical patterns."


            results.append({
                "prediction": "LAPSE" if is_lapse == 1 else "RETAIN",
                "risk_probability": round(prob, 4),
                "risk_level": risk_level,
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
