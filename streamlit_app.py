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
# 2. PATHS & SETUP
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'src'))

# Try to find whichever model exists
MODEL_PATH = os.path.join(BASE_DIR, "models", "kaggle_ensemble_model.joblib")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_optimized_model_new.joblib")

PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "kaggle_preprocessor.joblib")
if not os.path.exists(PREPROCESSOR_PATH):
    PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "scaler_new.joblib")

# ---------------------------------------------------------
# 3. CSS (YOUR GREEN THEME + ANIMATION)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* APP BACKGROUND */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at center, #0e2a47 0%, #000000 100%);
        color: white;
        overflow-x: hidden;
    }
    [data-testid="stSidebar"] {
        background-color: #0b1e33;
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    /* FLOATING CIRCLES */
    .circles {
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        overflow: hidden; z-index: 0; pointer-events: none;
    }
    .circles li {
        position: absolute; display: block; list-style: none;
        width: 20px; height: 20px; background: rgba(46, 204, 113, 0.2);
        animation: animate 25s linear infinite; bottom: -150px; border-radius: 50%;
    }
    .circles li:nth-child(1) { left: 25%; width: 80px; height: 80px; animation-delay: 0s; }
    .circles li:nth-child(2) { left: 10%; width: 20px; height: 20px; animation-delay: 2s; animation-duration: 12s; }
    .circles li:nth-child(3) { left: 70%; width: 20px; height: 20px; animation-delay: 4s; }
    .circles li:nth-child(4) { left: 40%; width: 60px; height: 60px; animation-delay: 0s; animation-duration: 18s; }
    .circles li:nth-child(5) { left: 65%; width: 20px; height: 20px; animation-delay: 0s; }

    @keyframes animate {
        0% { transform: translateY(0) rotate(0deg); opacity: 1; border-radius: 50%; }
        100% { transform: translateY(-1000px) rotate(720deg); opacity: 0; border-radius: 50%; }
    }

    /* UI ELEMENTS */
    .block-container { z-index: 10; position: relative; }
    
    .stButton>button {
        background-color: #2ecc71 !important; color: white !important;
        border-radius: 8px; border: none; padding: 10px 24px; font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover { transform: scale(1.02); }

    .metric-card {
        background-color: rgba(255, 255, 255, 0.05); padding: 20px;
        border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px); margin-bottom: 15px;
    }
    
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background-color: #0b1e33 !important; color: white !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<ul class="circles"><li></li><li></li><li></li><li></li><li></li><li></li><li></li><li></li><li></li><li></li></ul>', unsafe_allow_html=True)

# ---------------------------------------------------------
# 4. LOAD ARTIFACTS
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        try: from models.ensemble import ChurnEnsembleModel
        except ImportError: pass

        if not os.path.exists(MODEL_PATH): return None, None, f"Missing: {MODEL_PATH}"
    
        model = joblib.load(MODEL_PATH)
        preprocessor_data = joblib.load(PREPROCESSOR_PATH)
        
        feature_order = []
        scaler = None

        # Handle different formats (Dict vs Object)
        if isinstance(preprocessor_data, dict):
            scaler = preprocessor_data.get('scaler')
            feature_order = preprocessor_data.get('feature_names', [])
        elif hasattr(preprocessor_data, 'feature_names'):
            scaler = preprocessor_data.scaler
            feature_order = preprocessor_data.feature_names
        elif hasattr(preprocessor_data, 'transform'): # It's just a scaler
            scaler = preprocessor_data
            # If we loaded just a scaler, we try to load feature list from separate file
            feat_path = os.path.join(BASE_DIR, "models", "training_feature_order_new.joblib")
            if os.path.exists(feat_path):
                feature_order = joblib.load(feat_path)

        return model, scaler, feature_order
    except Exception as e:
        return None, None, str(e)

model, scaler, feature_order = load_artifacts()

# ---------------------------------------------------------
# 5. PREDICTION LOGIC (FIXED FOR SHAPE MISMATCH)
# ---------------------------------------------------------
def make_prediction(payload):
    if model is None: return None
        
    try:
        # 1. Create DataFrame
        df = pd.DataFrame([payload])
        
        # 2. FORCE FEATURE ALIGNMENT (Prevents "Expected 6, got 5")
        final_df = pd.DataFrame()
        
        # If we have a known feature order from training, USE IT
        if feature_order and len(feature_order) > 0:
            for col in feature_order:
                if col in df.columns:
                    final_df[col] = df[col]
                else:
                    final_df[col] = 0.0
        else:
            # Fallback if feature order missing: Pass raw DF (risky but handles some edge cases)
            final_df = df
        
        # 3. Scale
        if scaler:
            X_scaled = scaler.transform(final_df)
        else:
            X_scaled = final_df.values

        # 4. Predict
        pred = model.predict(X_scaled)[0]
        try: prob = model.predict_proba(X_scaled)[0][1]
        except: prob = 1.0 if pred == 1 else 0.0
        
        reason = "Stable metrics"
        # Simple logic based on the inputs available in the UI
        if payload.get('RETENTION_POLY_QTY', 0) < payload.get('PREV_POLY_INFORCE_QTY', 0):
            reason = f"Retention ({payload.get('RETENTION_POLY_QTY')}) < Previous ({payload.get('PREV_POLY_INFORCE_QTY')})"
            
        return {"risk": "High" if pred==1 else "Low", "score": prob, "driver": reason}
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        return None

# ---------------------------------------------------------
# 6. PAGES
# ---------------------------------------------------------
if "page" not in st.session_state: st.session_state.page = "home"
def go_to(p): st.session_state.page = p

def home_page():
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='font-size: 72px; margin-bottom: 10px;'>ChurnAlyse</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='opacity: 0.9; font-weight: 300;'>Predict churn, monitor risk, and save customers proactively.</h3>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if model:
        st.markdown(f"""
        <div style="background: rgba(46, 204, 113, 0.2); border: 1px solid #2ecc71; padding: 12px 25px; border-radius: 50px; display: inline-block;">
            <span style="color: #2ecc71; font-weight: bold; font-size: 16px;">● ML Engine Loaded (Embedded)</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error(f"🔴 Model Error: {feature_order}")

    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, _ = st.columns([1, 4])
    with col1:
        if st.button("Start Now"): go_to("predict")

def predict_page():
    st.sidebar.title("Navigation")
    st.sidebar.radio("Go to:", ["Predict", "Performance"], key="nav_pred", on_change=lambda: go_to(st.session_state.nav_pred.lower()))
    st.title("Predict Policy Lapse Risk")

    c1, c2 = st.columns([1, 1.3])
    with c1:
        with st.form("risk_form"):
            # --- RESTORED INPUTS FROM YOUR SCREENSHOT ---
            st.markdown("### 1. Customer")
            age = st.number_input("Age", 18, 100, 30)
            prem = st.number_input("Premium", 0, 100000, 3500)
            tenure = st.number_input("Tenure (Yrs)", 0.0, 50.0, 1.5)
            
            # Channels
            ch1 = st.number_input("Agent Channel", 0, 1, 0)
            ch2 = st.number_input("Digital Channel", 0, 1, 1)
            ch3 = st.number_input("Bancassurance", 0, 1, 0)
            
            st.markdown("### 2. Agency Metrics")
            ret = st.number_input("Retained Qty", 0, 5000, 90)
            prev = st.number_input("Prev. Qty", 0, 5000, 100)
            curr = st.number_input("Curr. Qty", 0, 5000, 90)
            loss = st.number_input("Loss Ratio", 0.0, 500.0, 65.0)
            loss3 = st.number_input("3-Yr Loss Ratio", 0.0, 500.0, 60.0)
            growth = st.number_input("Growth %", -100.0, 100.0, 2.5)
            
            submitted = st.form_submit_button("Predict")
            
    if submitted:
        # Map inputs to ALL possible features to avoid mismatch
        payload = {
            "RETENTION_POLY_QTY": ret,
            "PREV_POLY_INFORCE_QTY": prev,
            "POLY_INFORCE_QTY": curr,
            "LOSS_RATIO": loss,
            "LOSS_RATIO_3YR": loss3,
            "GROWTH_RATE_3YR": growth,
            "age": age,
            "premium_amount": prem,
            "policy_tenure_months": tenure * 12, # Convert yrs to months for Kaggle model
            "channel1": ch1,
            "channel2": ch2,
            "channel3": ch3,
            "policy_amount": prem * 10, # Estimate
            "credit_score": 700, # Default
            "income": prem * 5   # Default
        }
        
        res = make_prediction(payload)
        
        with c2:
            if res:
                color = "#ef4444" if res['risk'] == "High" else "#2ecc71"
                st.markdown(f"""
                <div class="metric-card" style="border-left: 8px solid {color}; text-align: left;">
                    <h3 style="color:{color}; margin:0;">Risk Level: {res['risk']}</h3>
                    <h1 style="font-size: 4rem; margin: 10px 0;">{res['score']:.1%} <span style="font-size: 20px; color: white;">Probability</span></h1>
                    <p style="opacity: 0.8;">{res['driver']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Strategies (Static for now based on UI)
                st.markdown("### Analysis")
                if ret < prev: st.warning("⚠️ Portfolio Shrinkage detected (Retention < Previous).")
                if loss > 100: st.error("⚠️ Critical Loss Ratio (>100%).")
                
                st.markdown("### Strategy")
                st.info("Offer premium reminders")
                st.info("Personalized agent follow-up")

def performance_page():
    st.sidebar.title("Navigation")
    st.sidebar.radio("Go to:", ["Predict", "Performance"], key="nav_perf", on_change=lambda: go_to(st.session_state.nav_perf.lower()))
    st.title("🏆 Model Leaderboard")
    
    # Try to load leaderboard if exists
    try:
        if os.path.exists("models/leaderboard.json"):
            with open("models/leaderboard.json", 'r') as f:
                leaderboard = json.load(f)
            data = [{"Model": k, **v} for k, v in leaderboard.items()]
            df = pd.DataFrame(data).sort_values("accuracy", ascending=False)
            
            for i, row in df.iterrows():
                st.markdown(f"### 🤖 {row['Model']}")
                cols = st.columns(5)
                def mbox(lbl, val): return f"""<div class="metric-card"><div class="metric-label">{lbl}</div><div class="metric-value">{val}</div></div>"""
                cols[0].markdown(mbox("Accuracy", f"{row['accuracy']:.1%}"), unsafe_allow_html=True)
                cols[1].markdown(mbox("Precision", f"{row['precision']:.1%}"), unsafe_allow_html=True)
                cols[2].markdown(mbox("Recall", f"{row['recall']:.1%}"), unsafe_allow_html=True)
                cols[3].markdown(mbox("F1 Score", f"{row['f1_score']:.1%}"), unsafe_allow_html=True)
                cols[4].markdown(mbox("AUC", f"{row['auc']:.3f}"), unsafe_allow_html=True)
        else:
            st.info("Leaderboard data available in training logs.")
    except:
        st.info("Performance data not found.")

if st.session_state.page == "home": home_page()
elif st.session_state.page == "predict": predict_page()
else: performance_page()
