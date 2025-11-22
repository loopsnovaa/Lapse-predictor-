import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
import os
import json

# ---------------------------------------------------------
# SETUP & MODEL LOADING (Cloud-Native Logic)
# ---------------------------------------------------------
st.set_page_config(page_title="ChurnAlyse", layout="wide", page_icon="📉")

# Paths
MODEL_PATH = "models/xgboost_optimized_model_new.joblib"
SCALER_PATH = "models/scaler_new.joblib"
FEATURE_ORDER_PATH = "models/training_feature_order_new.joblib"
LEADERBOARD_PATH = "models/leaderboard.json"

@st.cache_resource
def load_model_artifacts():
    """Load models directly into memory (No API needed)"""
    try:
        if not os.path.exists(MODEL_PATH):
            return None, None, None
            
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        features = joblib.load(FEATURE_ORDER_PATH)
        return model, scaler, features
    except Exception as e:
        return None, None, None

@st.cache_data
def load_leaderboard():
    """Load leaderboard directly from JSON file"""
    try:
        if not os.path.exists(LEADERBOARD_PATH):
            return None
        with open(LEADERBOARD_PATH, 'r') as f:
            return json.load(f)
    except:
        return None

# Load artifacts on startup
model, scaler, feature_order = load_model_artifacts()

# ---------------------------------------------------------
# INTERNAL PREDICTION LOGIC
# ---------------------------------------------------------
def make_prediction(payload):
    """Runs XGBoost prediction locally in the dashboard"""
    if not model or not scaler:
        return None
    
    try:
        # Convert payload to DataFrame
        df = pd.DataFrame([payload])
        
        # Ensure all training columns exist
        for col in feature_order:
            if col not in df.columns:
                df[col] = 0
                
        # Sort and Scale
        df_sorted = df[feature_order]
        X_scaled = scaler.transform(df_sorted)
        
        # Predict
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]
        
        # Explanation Rule (Retention vs Previous)
        retention = payload.get('RETENTION_POLY_QTY', 0)
        prev = payload.get('PREV_POLY_INFORCE_QTY', 0)
        reason = "Stable metrics"
        if prediction == 1:
            reason = f"Retention Qty ({retention}) < Previous Qty ({prev})"

        return {
            "prediction": "LAPSE" if prediction == 1 else "RETAIN",
            "confidence_score": probability,
            "primary_driver": reason
        }
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        return None

# ---------------------------------------------------------
# CSS STYLING (FIXED)
# ---------------------------------------------------------
CUSTOM_CSS = """
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }
[data-testid="stAppViewContainer"] { background-color: #0d3a66 !important; color: white !important; }
[data-testid="stSidebar"] { background-color: #0f4c81 !important; }
h1, h2, h3, h4, p, label, .stMarkdown { color: white !important; }

/* Input Fields */
.stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>div {
    color: black !important; background-color: #e6f2ff !important; border-radius: 5px;
}

/* Buttons */
.stButton>button {
    background-color: #b2f7b1 !important; color: black !important; border-radius: 10px;
    border: none; padding: 10px 25px; font-size: 18px; font-weight: 600; width: 100%;
}
.stButton>button:hover { background-color: #A0E15E !important; }

/* Metric Cards (The Boxes) */
.metric-card {
    background-color: rgba(255, 255, 255, 0.1); 
    padding: 20px; 
    border-radius: 12px; 
    border: 1px solid rgba(255,255,255,0.2); 
    margin-bottom: 10px;
    min-height: 120px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
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
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def explain_channels(data):
    ch1, ch2, ch3 = data.get("channel1", 0), data.get("channel2", 0), data.get("channel3", 0)
    explanation = []
    if ch1 == 0 and ch2 == 0 and ch3 == 0: explanation.append("Low-engagement channel (walk-in/telemarketing).")
    if ch1 >= 1: explanation.append("Acquired through advisor/agent (strong follow-up).")
    if ch2 >= 1: explanation.append("Acquired through digital channel (medium risk).")
    if ch3 >= 1: explanation.append("Bancassurance channel (moderate stability).")
    return explanation if explanation else ["Mixed channel combination."]

def explain_risk_factors(data, risk_level):
    reasons = []
    if data.get("RETENTION_POLY_QTY", 0) < data.get("PREV_POLY_INFORCE_QTY", 0):
        reasons.append("⚠️ Portfolio Shrinkage detected (Retention < Previous).")
    if data.get("LOSS_RATIO", 0) > 1.0:
        reasons.append("⚠️ Critical Loss Ratio (>100%).")
    if data.get("premium_amount", 0) > 3000:
        reasons.append("💰 High Premium (>3000).")
    if data.get("policy_tenure_years", 0) < 2:
        reasons.append("⏳ Short tenure (< 2 years).")
    
    strategies = ["Offer premium reminders", "Personalized agent follow-up", "Explain long-term benefits"]
    return reasons, strategies

# ---------------------------------------------------------
# NAVIGATION
# ---------------------------------------------------------
if "page" not in st.session_state: st.session_state.page = "home"
def go_to(p): st.session_state.page = p

def home_page():
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.title("ChurnAlyse")
    st.subheader("Cloud-Native Insurance Analytics")
    
    if model:
        st.success("🟢 AI Engine Loaded (Embedded)")
    else:
        st.error("🔴 Model Files Missing. Please check 'models/' folder.")

    if st.button("Start Now"): go_to("predict")

def predict_page():
    st.sidebar.title("Navigation")
    st.sidebar.radio("Go to:", ["Predict", "Performance"], key="nav_pred", on_change=lambda: go_to(st.session_state.nav_pred.lower()))
    st.title("Predict Policy Lapse Risk")

    col1, col2 = st.columns([1, 1.2])
    with col1:
        with st.form("main_form"):
            st.markdown("### 1. Customer")
            age = st.number_input("Age", 18, 99, 30)
            prem = st.number_input("Premium", 1, 100000, 3500)
            ten = st.number_input("Tenure (Yrs)", 0.0, 50.0, 1.5)
            ch1 = st.number_input("Agent Channel", 0, 1, 0)
            ch2 = st.number_input("Digital Channel", 0, 1, 1)
            ch3 = st.number_input("Bancassurance", 0, 1, 0)
            
            st.markdown("### 2. Agency Metrics")
            ret_qty = st.number_input("Retained Qty", 0, 10000, 90)
            prev_qty = st.number_input("Prev. Qty", 0, 10000, 100)
            curr_qty = st.number_input("Curr. Qty", 0, 10000, 90)
            loss_r = st.number_input("Loss Ratio", 0.0, 500.0, 65.0)
            loss_3 = st.number_input("3-Yr Loss Ratio", 0.0, 500.0, 60.0)
            growth = st.number_input("Growth %", -100.0, 100.0, 2.5)
            
            submit = st.form_submit_button("Predict")

    if submit:
        # Macro Data
        api_payload = {
            "RETENTION_POLY_QTY": ret_qty, "PREV_POLY_INFORCE_QTY": prev_qty,
            "POLY_INFORCE_QTY": curr_qty, "LOSS_RATIO": loss_r,
            "LOSS_RATIO_3YR": loss_3, "GROWTH_RATE_3YR": growth
        }
        # Micro Data
        full_data = {**api_payload, "premium_amount": prem, "policy_tenure_years": ten, 
                     "channel1": ch1, "channel2": ch2, "channel3": ch3}
        
        # Internal Prediction Call (No API)
        res = make_prediction(api_payload)
        
        with col2:
            if res:
                prob = res['confidence_score']
                risk = "High" if res['prediction'] == "LAPSE" else "Low"
                color = "#d00000" if risk == "High" else "#A0E15E"
                
                st.markdown(f"""
                <div class="metric-card" style="border-left: 5px solid {color}; display: block; text-align: left;">
                    <h3 style="color:white; margin:0;">Risk Level: <span style="color:{color}">{risk}</span></h3>
                    <h1 style="color:white; margin:10px 0;">{prob*100:.1f}% <span style="font-size: 20px">Probability</span></h1>
                    <p style="color:#ccc;">{res['primary_driver']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                reasons, strats = explain_risk_factors(full_data, risk)
                st.markdown("### Analysis")
                for r in reasons: st.write(r)
                if risk == "High":
                    st.markdown("### Strategy")
                    for s in strats: st.info(s)

def performance_page():
    st.sidebar.title("Navigation")
    st.sidebar.radio("Go to:", ["Predict", "Performance"], key="nav_perf", on_change=lambda: go_to(st.session_state.nav_perf.lower()))
    st.title("🏆 Model Performance Leaderboard")
    
    leaderboard = load_leaderboard()
    if not leaderboard:
        st.warning("⚠️ Leaderboard data not found. Run `train_leaderboard.py` locally and upload 'models/leaderboard.json'.")
        return

    # --- PREPARE DATA ---
    model_data = []
    for name, metrics in leaderboard.items():
        model_data.append({
            "Model": name,
            "Accuracy": metrics.get('accuracy', 0),
            "Precision": metrics.get('precision', 0),
            "Recall": metrics.get('recall', 0),
            "F1 Score": metrics.get('f1_score', 0),
            "AUC": metrics.get('auc', 0)
        })
    
    # Sort by Accuracy
    df = pd.DataFrame(model_data).sort_values(by="Accuracy", ascending=False)

    # --- RENDER MODEL CARDS (WITH BOXES) ---
    for index, row in df.iterrows():
        st.markdown(f"### 🤖 {row['Model']}")
        
        c1, c2, c3, c4, c5 = st.columns(5)
        
        # Helper for cleaner code
        def metric_box(label, value):
            return f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """
            
        c1.markdown(metric_box("Accuracy", f"{row['Accuracy']:.1%}"), unsafe_allow_html=True)
        c2.markdown(metric_box("Precision", f"{row['Precision']:.3f}"), unsafe_allow_html=True)
        c3.markdown(metric_box("Recall", f"{row['Recall']:.3f}"), unsafe_allow_html=True)
        c4.markdown(metric_box("F1 Score", f"{row['F1 Score']:.3f}"), unsafe_allow_html=True)
        c5.markdown(metric_box("AUC", f"{row['AUC']:.3f}"), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

if st.session_state.page == "home": home_page()
elif st.session_state.page == "predict": predict_page()
else: performance_page()
