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
# 2. PATHS & SETUP
# ---------------------------------------------------------
# Use absolute paths to find the Kaggle models you just generated
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'src'))

# CRITICAL: Point to the NEW Kaggle models
MODEL_PATH = os.path.join(BASE_DIR, "models", "kaggle_ensemble_model.joblib")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "kaggle_preprocessor.joblib")

# ---------------------------------------------------------
# 3. CSS (EXACT GREEN STYLE FROM YOUR SCREENSHOT)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* 1. APP BACKGROUND */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at center, #0e2a47 0%, #000000 100%);
        color: white;
        overflow-x: hidden;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0b1e33;
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    /* 2. FLOATING CIRCLES (Bokeh Effect) */
    .circles {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        overflow: hidden;
        z-index: 0; /* Behind Content */
        pointer-events: none;
    }
    .circles li {
        position: absolute;
        display: block;
        list-style: none;
        width: 20px;
        height: 20px;
        background: rgba(46, 204, 113, 0.2); /* GREEN TINT */
        animation: animate 25s linear infinite;
        bottom: -150px;
        border-radius: 50%;
    }
    /* Randomize Particles */
    .circles li:nth-child(1) { left: 25%; width: 80px; height: 80px; animation-delay: 0s; }
    .circles li:nth-child(2) { left: 10%; width: 20px; height: 20px; animation-delay: 2s; animation-duration: 12s; }
    .circles li:nth-child(3) { left: 70%; width: 20px; height: 20px; animation-delay: 4s; }
    .circles li:nth-child(4) { left: 40%; width: 60px; height: 60px; animation-delay: 0s; animation-duration: 18s; }
    .circles li:nth-child(5) { left: 65%; width: 20px; height: 20px; animation-delay: 0s; }
    .circles li:nth-child(6) { left: 75%; width: 110px; height: 110px; animation-delay: 3s; }
    .circles li:nth-child(7) { left: 35%; width: 150px; height: 150px; animation-delay: 7s; }
    .circles li:nth-child(8) { left: 50%; width: 25px; height: 25px; animation-delay: 15s; animation-duration: 45s; }
    .circles li:nth-child(9) { left: 20%; width: 15px; height: 15px; animation-delay: 2s; animation-duration: 35s; }
    .circles li:nth-child(10){ left: 85%; width: 150px; height: 150px; animation-delay: 0s; animation-duration: 11s; }

    @keyframes animate {
        0% { transform: translateY(0) rotate(0deg); opacity: 1; border-radius: 50%; }
        100% { transform: translateY(-1000px) rotate(720deg); opacity: 0; border-radius: 50%; }
    }

    /* 3. BUTTON STYLING (From Image: #b2f7b1) */
    .stButton>button {
        background-color: #b2f7b1 !important;
        color: black !important;
        border-radius: 10px;
        border: none;
        padding: 10px 25px; 
        font-size: 18px; 
        font-weight: 600; 
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover { 
        background-color: #A0E15E !important; 
        transform: scale(1.02);
    }

    /* 4. METRIC CARD STYLING (From Image) */
    .metric-card {
        background-color: rgba(255, 255, 255, 0.1); 
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid rgba(255,255,255,0.2); 
        margin-bottom: 15px;
        backdrop-filter: blur(10px);
    }
    .metric-label {
        font-size: 14px;
        color: #A0E15E !important;
        margin-bottom: 5px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: white !important;
        margin: 0;
    }
    
    /* 5. Z-INDEX FIX (Fixes Blank Screen) */
    .block-container {
        z-index: 10 !important;
        position: relative;
        background: transparent;
    }
    
    /* 6. INPUT FIELDS */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background-color: #0b1e33 !important; 
        color: white !important; 
        border: 1px solid rgba(255,255,255,0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 4. INJECT HTML ANIMATION
# ---------------------------------------------------------
st.markdown("""
    <ul class="circles">
        <li></li><li></li><li></li><li></li><li></li>
        <li></li><li></li><li></li><li></li><li></li>
    </ul>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 5. DATA LOADING (SAFE & ROBUST)
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        # Import custom class so joblib knows what 'ChurnEnsembleModel' is
        try: from models.ensemble import ChurnEnsembleModel
        except: pass

        if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
            return None, None, f"Missing files. Expected at: {MODEL_PATH}"
    
        model = joblib.load(MODEL_PATH)
        preprocessor_data = joblib.load(PREPROCESSOR_PATH)
        
        # Handle format difference
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
# 6. LOGIC (FIXES SHAPE MISMATCH)
# ---------------------------------------------------------
def make_prediction(payload):
    if isinstance(feature_order, str): return None
    
    try:
        # 1. Create DataFrame from inputs
        df = pd.DataFrame([payload])
        
        # 2. ALIGN FEATURES (Crucial Step)
        # We create a new DF with exactly the columns the model expects (38 of them)
        # and fill the ones we didn't ask for in the UI with 0.0
        final_df = pd.DataFrame()
        for col in feature_order:
            if col in df.columns:
                final_df[col] = df[col]
            else:
                final_df[col] = 0.0 
        
        # 3. Scale & Predict
        X_scaled = scaler.transform(final_df)
        pred = model.predict(X_scaled)[0]
        
        # Handle probabilities
        try: prob = model.predict_proba(X_scaled)[0][1]
        except: prob = 1.0 if pred == 1 else 0.0
        
        return {"risk": "High" if pred == 1 else "Low", "score": prob}
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        return None

# ---------------------------------------------------------
# 7. UI PAGES
# ---------------------------------------------------------
if "page" not in st.session_state: st.session_state.page = "home"
def go_to(p): st.session_state.page = p

def home_page():
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='font-size: 72px; margin-bottom: 10px; text-shadow: 0 4px 10px rgba(0,0,0,0.5);'>ChurnAlyse</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='opacity: 0.9; font-weight: 300;'>Predict churn, monitor risk, and save customers proactively.</h3>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if model:
        # Green Pill (System Ready)
        st.markdown(f"""
        <div style="background: rgba(178, 247, 177, 0.2); border: 1px solid #b2f7b1; padding: 12px 25px; border-radius: 50px; display: inline-block;">
            <span style="color: #b2f7b1; font-weight: bold; font-size: 16px;">● ML Engine Loaded (Embedded)</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error(f"🔴 {feature_order}") # Show specific loading error

    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Start Now"):
            go_to("predict")

def predict_page():
    st.sidebar.title("Navigation")
    st.sidebar.radio("Go to:", ["Predict", "Performance"], key="nav_pred", on_change=lambda: go_to(st.session_state.nav_pred.lower()))
    st.title("🔮 Lapse Risk Predictor")

    c1, c2 = st.columns([1, 1.3])
    with c1:
        with st.form("risk_form"):
            st.caption("Policy Inputs")
            # Inputs match Kaggle features
            p_amt = st.number_input("Policy Amount", 0, 1000000, 50000)
            prem = st.number_input("Premium", 0, 50000, 1000)
            tenure = st.number_input("Tenure (Months)", 0, 360, 24)
            credit = st.number_input("Credit Score", 300, 850, 700)
            age = st.number_input("Age", 18, 90, 35)
            
            submitted = st.form_submit_button("Analyze Risk")
            
    if submitted:
        # Create payload with simulated extra features
        payload = {
            "policy_amount": p_amt, 
            "premium_amount": prem,
            "policy_tenure_months": tenure, 
            "credit_score": credit,
            "age": age,
            # Calculated features the model might expect
            "income": p_amt * 0.1, 
            "premium_to_tenure_ratio": prem / (tenure + 1)
        }
        
        res = make_prediction(payload)
        
        with c2:
            if res:
                risk = res['risk']
                color = "#ef4444" if risk == "High" else "#2ecc71"
                
                # HTML Result Card
                st.markdown(f"""
                <div class="metric-card" style="border-left: 8px solid {color}; align-items: flex-start; text-align: left; padding-left: 30px;">
                    <h3 style="color:{color}; margin:0; font-size: 24px;">RISK LEVEL: {risk.upper()}</h3>
                    <h1 style="font-size: 4rem; margin: 10px 0;">{res['score']:.1%}</h1>
                    <p style="opacity: 0.8; font-size: 16px;">Based on ensemble prediction</p>
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
