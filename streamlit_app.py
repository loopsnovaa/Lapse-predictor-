import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
import os
import sys
import json

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="ChurnAlyse", 
    layout="wide", 
    page_icon="📉",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. PATHS (Absolute to prevent errors)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'src'))

MODEL_PATH = os.path.join(BASE_DIR, "models", "kaggle_ensemble_model.joblib")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "kaggle_preprocessor.joblib")

# ---------------------------------------------------------
# 3. CSS & ANIMATION
# ---------------------------------------------------------
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at center, #0e2a47 0%, #000000 100%);
        color: white;
    }
    .stButton>button {
        background-color: #2ecc71 !important;
        color: white !important;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05); 
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid rgba(255,255,255,0.1); 
        margin-bottom: 15px;
    }
    .circles { position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: 0; pointer-events: none; }
    .circles li {
        position: absolute; display: block; list-style: none; width: 20px; height: 20px;
        background: rgba(46, 204, 113, 0.2); animation: animate 25s linear infinite; bottom: -150px; border-radius: 50%;
    }
    @keyframes animate {
        0% { transform: translateY(0) rotate(0deg); opacity: 1; }
        100% { transform: translateY(-1000px) rotate(720deg); opacity: 0; }
    }
    .block-container { z-index: 10; position: relative; }
</style>
<ul class="circles"><li></li><li></li><li></li><li></li><li></li><li></li><li></li><li></li><li></li><li></li></ul>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 4. SAFE DATA LOADING (NO TRAINING HERE)
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    # Attempt to load custom class if needed
    try: from models.ensemble import ChurnEnsembleModel
    except: pass

    if not os.path.exists(MODEL_PATH):
        return None, None, "File not found: " + MODEL_PATH
    
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor_data = joblib.load(PREPROCESSOR_PATH)
        
        if isinstance(preprocessor_data, dict):
            scaler = preprocessor_data.get('scaler')
            feature_order = preprocessor_data.get('feature_names', [])
        else:
            scaler = preprocessor_data.scaler
            feature_order = preprocessor_data.feature_names
            
        return model, scaler, feature_order
    except Exception as e:
        return None, None, str(e)

model, scaler, feature_order = load_artifacts()

# ---------------------------------------------------------
# 5. PREDICTION LOGIC
# ---------------------------------------------------------
def make_prediction(payload):
    if isinstance(feature_order, str): return None # Error state
    
    try:
        df = pd.DataFrame([payload])
        final_df = pd.DataFrame()
        for col in feature_order:
            final_df[col] = df.get(col, 0.0)
            
        X_scaled = scaler.transform(final_df)
        pred = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0][1]
        
        return {"risk": "High" if pred==1 else "Low", "score": prob}
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        return None

# ---------------------------------------------------------
# 6. UI
# ---------------------------------------------------------
st.title("ChurnAlyse")

if isinstance(feature_order, str): # Error returned as string
    st.error("🔴 System Offline")
    st.warning(f"Error loading models: {feature_order}")
    st.info("Check that 'models/kaggle_ensemble_model.joblib' exists and is compatible.")
elif model is None:
    st.error("🔴 Models Missing")
else:
    st.success("🟢 System Online")
    
    with st.form("calc"):
        c1, c2 = st.columns(2)
        p_amt = c1.number_input("Policy Amount", 0, 1000000, 50000)
        prem = c2.number_input("Premium", 0, 50000, 1000)
        tenure = c1.number_input("Tenure (Months)", 0, 360, 24)
        credit = c2.number_input("Credit Score", 300, 850, 700)
        
        if st.form_submit_button("Predict"):
            res = make_prediction({
                "policy_amount": p_amt, "premium_amount": prem,
                "policy_tenure_months": tenure, "credit_score": credit,
                "income": p_amt*0.1, "premium_to_tenure_ratio": prem/(tenure+1)
            })
            if res:
                color = "#ef4444" if res['risk'] == "High" else "#2ecc71"
                st.markdown(f"""
                <div class="metric-card" style="border-left: 5px solid {color}">
                    <h3 style="margin:0; color:{color}">{res['risk']} Risk</h3>
                    <h1>{res['score']:.1%} Probability</h1>
                </div>
                """, unsafe_allow_html=True)
