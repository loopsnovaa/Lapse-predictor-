import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import plotly.graph_objects as go

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
# 2. PATHS (Pointing to the NEW Kaggle Model)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'src'))

# FIX: Force app to look for the Kaggle model first
MODEL_PATH = os.path.join(BASE_DIR, "models", "kaggle_ensemble_model.joblib")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "kaggle_preprocessor.joblib")

# ---------------------------------------------------------
# 3. CSS (GREEN THEME & FLOATING CIRCLES)
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

        if not os.path.exists(MODEL_PATH): 
            # Fallback to old model if new one is missing, to prevent crash
            old_path = os.path.join(BASE_DIR, "models", "xgboost_optimized_model_new.joblib")
            if os.path.exists(old_path):
                return joblib.load(old_path), None, None # Return old model
            return None, None, f"Missing: {MODEL_PATH}"
    
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
# 5. PREDICTION LOGIC (SHAPE FIX)
# ---------------------------------------------------------
def make_prediction(payload):
    if model is None: return None
        
    try:
        # 1. Create DataFrame
        df = pd.DataFrame([payload])
        
        # 2. DYNAMIC ALIGNMENT (The Fix)
        # If we have feature names from the Kaggle model, we align to them.
        if feature_order and len(feature_order) > 0:
            final_df = pd.DataFrame()
            for col in feature_order:
                # If input exists, use it. If not, 0.
                final_df[col] = df[col] if col in df.columns else 0.0
            
            # Scale
            X_input = scaler.transform(final_df)
        else:
            # Fallback for old model (expects specific 6 columns)
            # This handles the "expected 6" case if old model loaded
            cols = ["RETENTION_POLY_QTY", "PREV_POLY_INFORCE_QTY", "LOSS_RATIO", "LOSS_RATIO_3YR", "GROWTH_RATE_3YR"]
            final_df = pd.DataFrame()
            for col in cols:
                final_df[col] = df[col] if col in df.columns else 0.0
            # Old model likely didn't use this specific scaler
            X_input = final_df.values

        # 3. Predict
        pred = model.predict(X_input)[0]
        try: prob = model.predict_proba(X_input)[0][1]
        except: prob = 1.0 if pred == 1 else 0.0
        
        return {"risk": "High" if pred==1 else "Low", "score": prob}
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
            <span style="color: #2ecc71; font-weight: bold; font-size: 16px;">● ML Engine Loaded</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error(f"🔴 Error: {feature_order}")

    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, _ = st.columns([1, 4])
    with col1:
        if st.button("Start Now"): go_to("predict")

def predict_page():
    st.sidebar.title("Navigation")
    st.sidebar.radio("Go to:", ["Predict", "Performance"], key="nav_pred", on_change=lambda: go_to(st.session_state.nav_pred.lower()))
    st.title("🔮 Lapse Risk Predictor")

    c1, c2 = st.columns([1, 1.3])
    with c1:
        with st.form("risk_form"):
            # RESTORED EXACT INPUTS FROM SCREENSHOTS
            st.markdown("### 1. Customer")
            age = st.number_input("Age", 18, 100, 30)
            prem = st.number_input("Premium", 0, 100000, 3500)
            tenure = st.number_input("Tenure (Yrs)", 0.0, 50.0, 1.5)
            
            st.markdown("### 2. Agency Metrics")
            ret = st.number_input("Retained Qty", 0, 5000, 90)
            prev = st.number_input("Prev. Qty", 0, 5000, 100)
            loss = st.number_input("Loss Ratio", 0.0, 500.0, 65.0)
            loss3 = st.number_input("3-Yr Loss Ratio", 0.0, 500.0, 60.0)
            growth = st.number_input("Growth %", -100.0, 100.0, 2.5)
            curr = st.number_input("Curr. Qty", 0, 5000, 90)
            
            submitted = st.form_submit_button("Analyze Risk")
            
    if submitted:
        # Combine inputs into one payload
        payload = {
            "RETENTION_POLY_QTY": ret, "PREV_POLY_INFORCE_QTY": prev, 
            "LOSS_RATIO": loss, "LOSS_RATIO_3YR": loss3, "GROWTH_RATE_3YR": growth,
            "policy_amount": prem * 10, "premium_amount": prem, "policy_tenure_months": tenure * 12, 
            "age": age, "income": prem * 5, "credit_score": 700
        }
        
        res = make_prediction(payload)
        
        with c2:
            if res:
                color = "#ef4444" if res['risk'] == "High" else "#2ecc71"
                st.markdown(f"""
                <div class="metric-card" style="border-left: 8px solid {color}; text-align: left;">
                    <h3 style="color:{color}; margin:0;">RISK LEVEL: {res['risk'].upper()}</h3>
                    <h1 style="font-size: 4rem; margin: 10px 0;">{res['score']:.1%}</h1>
                    <p style="opacity: 0.8;">{res.get('driver', 'Probability based on model')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Strategies
                st.markdown("### Analysis")
                if ret < prev: st.warning("⚠️ Portfolio Shrinkage detected.")
                if loss > 100: st.error("⚠️ Critical Loss Ratio (>100%).")
                st.markdown("### Strategy")
                st.info("Offer premium reminders")
                st.info("Personalized agent follow-up")

def performance_page():
    st.sidebar.title("Navigation")
    st.sidebar.radio("Go to:", ["Predict", "Performance"], key="nav_perf", on_change=lambda: go_to(st.session_state.nav_perf.lower()))
    st.title("🏆 Model Leaderboard")
    st.info("Check training logs for detailed metrics.")

if st.session_state.page == "home": home_page()
elif st.session_state.page == "predict": predict_page()
else: performance_page()
