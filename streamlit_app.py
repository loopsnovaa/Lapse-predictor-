import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import json
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION (Critical: Must be first)
# ---------------------------------------------------------
st.set_page_config(
    page_title="ChurnAlyse", 
    layout="wide", 
    page_icon="📉",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. PATHS & SETUP (Updated to match your Kaggle Pipeline)
# ---------------------------------------------------------
# Add src to path so we can load the custom model class if needed
sys.path.append('src')

MODEL_PATH = "models/kaggle_ensemble_model.joblib"
PREPROCESSOR_PATH = "models/kaggle_preprocessor.joblib"

# ---------------------------------------------------------
# 3. CSS & ANIMATION (YOUR EXACT REQUESTED STYLE)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* 1. APP BACKGROUND - Dark Blue Gradient */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at center, #0e2a47 0%, #000000 100%);
        color: white;
        overflow-x: hidden;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0b1e33;
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    /* 2. FLOATING CIRCLES ANIMATION */
    .circles {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        overflow: hidden;
        z-index: 0; /* Behind everything */
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

    /* RANDOMIZE POSITIONS */
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

    /* 3. ENSURE CONTENT IS VISIBLE */
    .block-container {
        z-index: 10 !important;
        position: relative;
        background: transparent;
    }

    /* 4. BUTTON STYLING (Green) */
    .stButton>button {
        background-color: #2ecc71 !important;
        color: white !important;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover { 
        background-color: #27ae60 !important; 
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(46, 204, 113, 0.3);
    }

    /* 5. METRIC CARD STYLING (Glass) */
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05); 
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid rgba(255,255,255,0.1); 
        margin-bottom: 15px;
        backdrop-filter: blur(10px);
    }
    .metric-label {
        font-size: 14px;
        color: #2ecc71 !important;
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
    
    /* 6. INPUT FIELDS */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background-color: #0b1e33 !important; 
        color: white !important; 
        border: 1px solid rgba(255,255,255,0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 4. INJECT HTML FOR ANIMATION
# ---------------------------------------------------------
st.markdown("""
    <ul class="circles">
        <li></li><li></li><li></li><li></li><li></li>
        <li></li><li></li><li></li><li></li><li></li>
    </ul>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 5. DATA LOADING (Updated Logic)
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        # Load the custom model class structure if necessary
        try:
            from src.models.ensemble import ChurnEnsembleModel
        except ImportError:
            pass # It might be pickled with the object

        if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
            return None, None, None
            
        # Load Model
        model = joblib.load(MODEL_PATH)
        
        # Load Preprocessor (Dict containing scaler & encoders)
        preprocessor_data = joblib.load(PREPROCESSOR_PATH)
        scaler = preprocessor_data.get('scaler')
        feature_order = preprocessor_data.get('feature_names', [])
        
        return model, scaler, feature_order
        
    except Exception as e:
        st.error(f"Failed to load artifacts: {e}")
        return None, None, None

model, scaler, feature_order = load_artifacts()

# ---------------------------------------------------------
# 6. LOGIC
# ---------------------------------------------------------
def make_prediction(payload):
    if not model or not scaler: return None
    try:
        # Create DF
        df = pd.DataFrame([payload])
        
        # Ensure strict column ordering matching training
        final_df = pd.DataFrame()
        for col in feature_order:
            if col in df.columns:
                final_df[col] = df[col]
            else:
                final_df[col] = 0.0 # Fill missing features with 0
        
        # Scale
        X_scaled = scaler.transform(final_df)
        
        # Predict
        pred = model.predict(X_scaled)[0]
        try: prob = model.predict_proba(X_scaled)[0][1]
        except: prob = 1.0 if pred == 1 else 0.0
        
        reason = "Stable metrics"
        # Logic for explanation
        if prob > 0.5:
            reason = "High Lapse Probability detected by Ensemble Model"
            
        return {"prediction": "LAPSE" if pred == 1 else "RETAIN", "confidence_score": prob, "primary_driver": reason}
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        return None

# ---------------------------------------------------------
# 7. PAGES
# ---------------------------------------------------------
if "page" not in st.session_state: st.session_state.page = "home"
def go_to(p): st.session_state.page = p

def home_page():
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='font-size: 72px; margin-bottom: 10px; text-shadow: 0 4px 10px rgba(0,0,0,0.5);'>ChurnAlyse</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='opacity: 0.9; font-weight: 300;'>Predict churn, monitor risk, and save customers proactively.</h3>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if model:
        st.markdown(f"""
        <div style="background: rgba(46, 204, 113, 0.2); border: 1px solid #2ecc71; padding: 15px; border-radius: 8px; display: inline-block;">
            <span style="color: #2ecc71; font-weight: bold; font-size: 16px;">● ML Engine Loaded ({len(feature_order)} features)</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("🔴 Models Missing. Please run integrate_kaggle_dataset.py")

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
            st.caption("Policy Metrics")
            
            # Using some of the key features from your Kaggle dataset
            p_amt = st.number_input("Policy Amount", 0, 1000000, 50000)
            prem = st.number_input("Premium Amount", 0, 50000, 1000)
            tenure = st.number_input("Tenure (Months)", 0, 360, 24)
            age = st.number_input("Age", 18, 100, 35)
            credit = st.number_input("Credit Score", 300, 850, 700)
            
            # Advanced inputs (simulated or simplified for UI)
            claims = st.slider("Claims History", 0, 10, 0)
            
            submitted = st.form_submit_button("Analyze Risk")
            
    if submitted:
        # Construct payload matching your Kaggle Schema
        payload = {
            "policy_amount": p_amt,
            "premium_amount": prem,
            "policy_tenure_months": tenure,
            "age": age,
            "credit_score": credit,
            "claims_history": claims,
            # Add calculated fields expected by engineer_features
            "income": p_amt * 0.1, # Estimated
            "premium_to_tenure_ratio": prem / (tenure + 1)
        }
        
        res = make_prediction(payload)
        
        with c2:
            if res:
                risk = "High" if res['prediction'] == "LAPSE" else "Low"
                # Dynamic Color: Red for High Risk, Green for Low
                color = "#ef4444" if risk == "High" else "#2ecc71"
                
                # HTML Card Result
                st.markdown(f"""
                <div class="metric-card" style="border-left: 8px solid {color}; align-items: flex-start; text-align: left; padding-left: 30px;">
                    <h3 style="color:{color}; margin:0; font-size: 24px;">RISK LEVEL: {risk.upper()}</h3>
                    <h1 style="font-size: 4rem; margin: 10px 0;">{res['confidence_score']:.1%}</h1>
                    <p style="opacity: 0.8; font-size: 16px;">{res['primary_driver']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Radar Chart
                categories = ['Premium Risk', 'Tenure Risk', 'Credit Risk']
                # Normalize values for visualization roughly
                vals = [
                    min(1, prem/5000), 
                    max(0, 1 - (tenure/60)), 
                    max(0, 1 - (credit/850))
                ]
                
                fig = go.Figure(go.Scatterpolar(r=vals, theta=categories, fill='toself', line_color=color))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", 
                    plot_bgcolor="rgba(0,0,0,0)", 
                    font=dict(color="white"), 
                    margin=dict(t=20, b=20, l=40, r=40), 
                    height=300,
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1]))
                )
                st.plotly_chart(fig, use_container_width=True)

def performance_page():
    st.sidebar.title("Navigation")
    st.sidebar.radio("Go to:", ["Predict", "Performance"], key="nav_perf", on_change=lambda: go_to(st.session_state.nav_perf.lower()))
    st.title("🏆 Model Leaderboard")
    
    # Leaderboard isn't saved as JSON in the new pipeline, 
    # but we can try to load it if it exists or show a placeholder
    try:
        if os.path.exists("models/leaderboard.json"):
            with open("models/leaderboard.json", 'r') as f:
                leaderboard = json.load(f)
                
            data = [{"Model": k, **v} for k, v in leaderboard.items()]
            df = pd.DataFrame(data).sort_values("accuracy", ascending=False)
            
            for i, row in df.iterrows():
                st.markdown(f"### 🤖 {row['Model']}")
                cols = st.columns(5)
                
                def mbox(lbl, val):
                    return f"""<div class="metric-card"><div class="metric-label">{lbl}</div><div class="metric-value">{val}</div></div>"""
                
                cols[0].markdown(mbox("Accuracy", f"{row['accuracy']:.1%}"), unsafe_allow_html=True)
                cols[1].markdown(mbox("Precision", f"{row['precision']:.1%}"), unsafe_allow_html=True)
                cols[2].markdown(mbox("Recall", f"{row['recall']:.1%}"), unsafe_allow_html=True)
                cols[3].markdown(mbox("F1 Score", f"{row['f1_score']:.1%}"), unsafe_allow_html=True)
                cols[4].markdown(mbox("AUC", f"{row['auc']:.3f}"), unsafe_allow_html=True)
        else:
            st.info("Leaderboard JSON not generated in this run. (Check console logs for training metrics)")
            
    except Exception as e:
        st.error(f"Error loading leaderboard: {e}")

if st.session_state.page == "home": home_page()
elif st.session_state.page == "predict": predict_page()
else: performance_page()
