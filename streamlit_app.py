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
# 2. PATHS (Pointing to NEW Kaggle Models)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'src'))

# CRITICAL: Point to the correct files
MODEL_PATH = os.path.join(BASE_DIR, "models", "kaggle_ensemble_model.joblib")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "kaggle_preprocessor.joblib")

# ---------------------------------------------------------
# 3. CSS & ANIMATION (Green Theme & Floating Circles)
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
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
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

# INJECT ANIMATION HTML
st.markdown('<ul class="circles"><li></li><li></li><li></li><li></li><li></li><li></li><li></li><li></li><li></li><li></li></ul>', unsafe_allow_html=True)

# ---------------------------------------------------------
# 4. DATA LOADING
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        # Import class definition so joblib doesn't crash
        from models.ensemble import ChurnEnsembleModel
    except ImportError:
        pass

    if not os.path.exists(MODEL_PATH):
        return None, None, f"File missing: {MODEL_PATH}"
    
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor_data = joblib.load(PREPROCESSOR_PATH)
        
        # Handle dict vs object format
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
# 5. LOGIC (FIXES SHAPE MISMATCH)
# ---------------------------------------------------------
def make_prediction(payload):
    if isinstance(feature_order, str) or model is None: 
        return None
        
    try:
        # 1. Create Input DataFrame
        df = pd.DataFrame([payload])
        
        # 2. ALIGN FEATURES (The Fix)
        # Create a dataframe with EXACTLY the columns the model expects
        final_df = pd.DataFrame()
        for col in feature_order:
            if col in df.columns:
                final_df[col] = df[col]
            else:
                final_df[col] = 0.0 # Fill missing with 0
        
        # 3. Scale & Predict
        X_scaled = scaler.transform(final_df)
        pred = model.predict(X_scaled)[0]
        
        try: prob = model.predict_proba(X_scaled)[0][1]
        except: prob = 1.0 if pred == 1 else 0.0
        
        reason = "Stable metrics"
        if prob > 0.5: reason = "High Lapse Probability"
            
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
        # Green Pill
        st.markdown(f"""
        <div style="background: rgba(46, 204, 113, 0.2); border: 1px solid #2ecc71; padding: 12px 25px; border-radius: 50px; display: inline-block;">
            <span style="color: #2ecc71; font-weight: bold; font-size: 16px;">● ML Engine Loaded ({len(feature_order)} features)</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error(f"🔴 Error: {feature_order}")
        st.info("Run: python integrate_kaggle_dataset.py")

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
            st.caption("Policy Inputs")
            p_amt = st.number_input("Policy Amount", 0, 1000000, 50000)
            prem = st.number_input("Premium Amount", 0, 50000, 1000)
            tenure = st.number_input("Tenure (Months)", 0, 360, 24)
            credit = st.number_input("Credit Score", 300, 850, 700)
            age = st.number_input("Age", 18, 90, 35)
            
            submitted = st.form_submit_button("Analyze Risk")
            
    if submitted:
        # Payload matching Kaggle features
        payload = {
            "policy_amount": p_amt, "premium_amount": prem,
            "policy_tenure_months": tenure, "credit_score": credit, "age": age,
            "income": p_amt * 0.1, "premium_to_tenure_ratio": prem / (tenure + 1)
        }
        
        res = make_prediction(payload)
        
        with c2:
            if res:
                color = "#ef4444" if res['risk'] == "High" else "#2ecc71"
                st.markdown(f"""
                <div class="metric-card" style="border-left: 8px solid {color}; text-align: left;">
                    <h3 style="color:{color}; margin:0;">RISK LEVEL: {res['risk'].upper()}</h3>
                    <h1 style="font-size: 4rem; margin: 10px 0;">{res['score']:.1%}</h1>
                    <p style="opacity: 0.8;">{res['driver']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Radar Chart
                categories = ['Premium Risk', 'Tenure Risk', 'Credit Risk']
                vals = [min(1, prem/5000), max(0, 1-(tenure/60)), max(0, 1-(credit/850))]
                fig = go.Figure(go.Scatterpolar(r=vals, theta=categories, fill='toself', line_color=color))
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
                                  font=dict(color="white"), margin=dict(t=20, b=20, l=40, r=40), height=300)
                st.plotly_chart(fig, use_container_width=True)

def performance_page():
    st.sidebar.title("Navigation")
    st.sidebar.radio("Go to:", ["Predict", "Performance"], key="nav_perf", on_change=lambda: go_to(st.session_state.nav_perf.lower()))
    st.title("🏆 Model Leaderboard")
    st.info("Check training logs for detailed metrics.")

if st.session_state.page == "home": home_page()
elif st.session_state.page == "predict": predict_page()
else: performance_page()
