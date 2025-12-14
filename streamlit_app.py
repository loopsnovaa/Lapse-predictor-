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
# CSS STYLING & ANIMATION
# ---------------------------------------------------------
def add_bg_animation():
    st.markdown("""
    <style>
    .animation-wrapper {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        /* Dark Green to Black Radial Gradient */
        background: radial-gradient(ellipse at center, #051a0d 0%, #000000 100%);
        z-index: -2;
    }

    .head-silhouette {
        position: fixed;
        top: 50%;
        left: 5%; 
        transform: translateY(-50%);
        width: 400px; 
        height: 500px;
        background-image: url('https://i.imgur.com/8Q5X54Q.png'); 
        background-size: contain;
        background-repeat: no-repeat;
        background-position: center left;
        opacity: 0.5;
        z-index: -1;
        /* Green Glow on the silhouette */
        filter: drop-shadow(0 0 10px rgba(46, 204, 113, 0.2)); 
    }

    .circles {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        overflow: hidden;
        margin: 0;
        padding: 0;
    }

    .circles li {
        position: absolute;
        display: block;
        list-style: none;
        width: 10px;
        height: 10px;
        /* NEON GREEN PARTICLES */
        background: rgba(57, 255, 20, 0.5);
        box-shadow: 0 0 10px rgba(57, 255, 20, 0.8);
        animation: animate 20s linear infinite;
        bottom: -150px;
        border-radius: 50%;
    }

    .circles li:nth-child(1) { left: 15%; width: 15px; height: 15px; animation-delay: 0s; animation-duration: 15s; }
    .circles li:nth-child(2) { left: 25%; width: 10px; height: 10px; animation-delay: 2s; animation-duration: 18s; }
    .circles li:nth-child(3) { left: 10%; width: 20px; height: 20px; animation-delay: 4s; }
    .circles li:nth-child(4) { left: 35%; width: 12px; height: 12px; animation-delay: 0s; animation-duration: 14s; }
    .circles li:nth-child(5) { left: 50%; width: 8px; height: 8px; animation-delay: 0s; }
    
    @keyframes animate {
        0% { transform: translateY(0) scale(0.8); opacity: 0; }
        20% { opacity: 0.8; }
        100% { transform: translateY(-800px) scale(1.5); opacity: 0; }
    }
    
    .stApp { background: transparent !important; }
    [data-testid="stAppViewContainer"] { background: transparent !important; }
    </style>

    <div class="animation-wrapper">
        <div class="head-silhouette"></div>
        <ul class="circles">
            <li></li><li></li><li></li><li></li><li></li>
            <li></li><li></li><li></li><li></li><li></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

CUSTOM_CSS = """
<style>
/* Base Styles */
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }
[data-testid="stAppViewContainer"] { background-color: transparent !important; color: white !important; }

/* --- CHANGED: Sidebar color to match the top header bar --- */
[data-testid="stSidebar"] { background-color: #072540 !important; }

h1, h2, h3, h4, p, label, .stMarkdown { color: white !important; }

/* Input Fields */
.stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>div {
    color: black !important; background-color: #e6f2ff !important; border-radius: 5px;
}

/* --- GREEN BUTTON STYLING --- */
.stButton>button {
    background-color: #2ECC71 !important;
    color: white !important;
    border-radius: 10px;
    border: none; 
    padding: 10px 25px; 
    font-size: 18px; 
    font-weight: 700; 
    width: 100%;
    transition: all 0.3s ease;
}

/* Button Hover Effect (Darker Green) */
.stButton>button:hover { 
    background-color: #219150 !important; 
    box-shadow: 0 5px 15px rgba(46, 204, 113, 0.4);
    transform: translateY(-2px);
}

/* Metric Cards */
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
    # Identify Risk Drivers
    if data.get("RETENTION_POLY_QTY", 0) < data.get("PREV_POLY_INFORCE_QTY", 0):
        reasons.append("⚠️ Portfolio Shrinkage (Retention < Previous).")
    if data.get("LOSS_RATIO", 0) > 100.0:
        reasons.append("⚠️ Critical Loss Ratio (>100%).")
    if data.get("premium_amount", 0) > 3000:
        reasons.append("💰 High Premium (>3000).")
    if data.get("policy_tenure_years", 0) < 2:
        reasons.append("⏳ Early Stage Policy (< 2 years).")

    strategies = []
    
    # 1. Tenure Strategy
    tenure = data.get("policy_tenure_years", 0)
    if tenure < 1.0:
        strategies.append("🆕 Onboarding: Schedule 'Welcome Call' to reinforce policy value & benefits.")
    elif tenure < 3.0:
        strategies.append("🔄 Engagement: Send 'Policy Anniversary' review checking coverage adequacy.")
    else:
        strategies.append("💎 Loyalty: Offer 'Tenure-Based Discount' or upgrade options for loyalty.")

    # 2. Financial Strategy (Premium)
    prem = data.get("premium_amount", 0)
    if prem > 5000:
         strategies.append("💼 VIP Retention: Assign Senior Relationship Manager for personal financial review.")
    elif prem > 3000:
         strategies.append("💳 Flexibility: Offer 'Premium Holiday' or switch to monthly payment mode.")
    
    # 3. Channel Strategy
    if data.get("channel1", 0) == 1: # Agent
        strategies.append("🤝 Agent Prompt: Trigger urgent task for Agent: 'Client at Risk - Call ASAP'.")
    elif data.get("channel2", 0) == 1: # Digital
        strategies.append("📧 Digital Campaign: Send automated 'Why Stay?' email series with success stories.")
    elif data.get("channel3", 0) == 1: # Bancassurance
        strategies.append("🏦 Bank Partner: Notify Bank Agent to discuss insurance during next account review.")

    # 4. Critical Risk Action (if High Risk)
    if risk_level == "High":
        strategies.insert(0, "🚨 Immediate Action: Offer one-time 'Lapse Prevention Discount' valid for 7 days.")

    # Ensure we have something
    if not strategies:
        strategies = ["Offer premium reminders via SMS/Email.", "Conduct a satisfaction survey.", "Highlight loss of accumulated benefits."]

    return reasons, strategies

# ---------------------------------------------------------
# NAVIGATION
# ---------------------------------------------------------
if "page" not in st.session_state: st.session_state.page = "home"
def go_to(p): st.session_state.page = p

def home_page():
    # --- HOMEPAGE SPECIFIC BACKGROUND CSS ---
    st.markdown("""
    <style>
        /* Override global background for homepage */
        [data-testid="stAppViewContainer"] {
            background-color: #111 !important; 
            background-image: none !important; 
            color: #f2f2f2;
        }
        header[data-testid="stHeader"] {
            background-color: #111 !important;
        }
        
        /* --- GLOWING TEXT CSS (FIX ADDED HERE) --- */
        .glowing-text {
            color: #FFFFFF;
            animation: glow 1.5s ease-in-out infinite alternate;
        }
        
        @keyframes glow {
            from {
                text-shadow: 0 0 5px #2ECC71, 0 0 10px #2ECC71, 0 0 20px #2ECC71;
            }
            to {
                text-shadow: 0 0 10px #2ECC71, 0 0 20px #2ECC71, 0 0 30px #2ECC71, 0 0 40px #2ECC71;
            }
        }
        
        /* The container for the lines */
        .lines {
            position: fixed; 
            top: 0;
            left: 0;
            right: 0;
            height: 100%;
            margin: auto;
            width: 90vw;
            display: flex;
            justify-content: space-between;
            z-index: 0; 
            pointer-events: none; 
        }

        .line {
            position: relative;
            width: 1px;
            height: 100%;
            background: transparent; 
            overflow: hidden;
        }

        .line::after {
            content: '';
            display: block;
            position: absolute;
            height: 15vh;
            width: 100%;
            top: -50%;
            left: 0;
            background: linear-gradient(to bottom, rgba(255, 255, 255, 0) 0%, #ffffff 75%, #ffffff 100%);
            /* SPEED UP: Changed from 7s to 1.5s */
            animation: drop 3s 0s infinite;
            animation-fill-mode: forwards;
            animation-timing-function: cubic-bezier(0.4, 0.26, 0, 0.97);
        }

        /* TIGHTER DELAYS for faster speed */
        .line:nth-child(1)::after { background: linear-gradient(to bottom, rgba(255, 255, 255, 0) 0%, #FF4500 75%, #FF4500 100%); animation-delay: 0.1s; }
        .line:nth-child(2)::after { background: linear-gradient(to bottom, rgba(255, 255, 255, 0) 0%, #32CD32 75%, #32CD32 100%); animation-delay: 0.25s; }
        .line:nth-child(3)::after { background: linear-gradient(to bottom, rgba(255, 255, 255, 0) 0%, #1E90FF 75%, #1E90FF 100%); animation-delay: 0.4s; }
        .line:nth-child(4)::after { background: linear-gradient(to bottom, rgba(255, 255, 255, 0) 0%, #FFD700 75%, #FFD700 100%); animation-delay: 0.55s; }
        .line:nth-child(5)::after { background: linear-gradient(to bottom, rgba(255, 255, 255, 0) 0%, #8A2BE2 75%, #8A2BE2 100%); animation-delay: 0.7s; }
        .line:nth-child(6)::after { background: linear-gradient(to bottom, rgba(255, 255, 255, 0) 0%, #20B2AA 75%, #20B2AA 100%); animation-delay: 0.85s; }
        .line:nth-child(7)::after { background: linear-gradient(to bottom, rgba(255, 255, 255, 0) 0%, #DC143C 75%, #DC143C 100%); animation-delay: 1.0s; }
        .line:nth-child(8)::after { background: linear-gradient(to bottom, rgba(255, 255, 255, 0) 0%, #00FA9A 75%, #00FA9A 100%); animation-delay: 1.15s; }
        .line:nth-child(9)::after { background: linear-gradient(to bottom, rgba(255, 255, 255, 0) 0%, #FF1493 75%, #FF1493 100%); animation-delay: 1.3s; }
        .line:nth-child(10)::after { background: linear-gradient(to bottom, rgba(255, 255, 255, 0) 0%, #00BFFF 75%, #00BFFF 100%); animation-delay: 1.45s; }

        @keyframes drop {
            0% { top: -50%; }
            100% { top: 110%; }
        }
    </style>
    
    <div class="lines">
        <div class="line"></div>
        <div class="line"></div>
        <div class="line"></div>
        <div class="line"></div>
        <div class="line"></div>
        <div class="line"></div>
        <div class="line"></div>
        <div class="line"></div>
        <div class="line"></div>
        <div class="line"></div>
    </div>
    """, unsafe_allow_html=True)

    # --- RESTORED HOMEPAGE UI ---
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    
    # Title with the glowing animation
    st.markdown("<h1 class='glowing-text' style='font-size: 72px; margin-bottom: 10px; text-align: center;'>ChurnAlyse</h1>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='opacity: 0.9; font-weight: 300; text-align: center;'>Predict churn, monitor risk, and save customers proactively.</h3>", unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if model:
            st.markdown(f"""
            <div style="background: rgba(46, 204, 113, 0.2); border: 1px solid #2ecc71; padding: 12px 25px; border-radius: 50px; display: inline-block; width: 100%; text-align: center;">
                <span style="color: #2ecc71; font-weight: bold; font-size: 16px;">● ML Engine Loaded</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("🔴 Error: Model not found. Please run training script.")

    st.markdown("<br>", unsafe_allow_html=True)
    col4, col5, col6 = st.columns([1, 1, 1])
    with col5:
        if st.button("Start Risk Analysis", use_container_width=True, key='home_btn'): 
            go_to("predict")

def predict_page():
    # --- FORCE BLACK BACKGROUND FOR PREDICT PAGE ---
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #000000 !important;
    }
    </style>
    """, unsafe_allow_html=True)

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
                # Updated Logic for Green/Red display
                color = "#d00000" if risk == "High" else "#2ECC71"
                
                st.markdown(f"""
                <div style="
                    background-color: rgba(208, 0, 0, 0.2);
                    border: 2px solid {color};
                    padding: 20px;
                    border-radius: 12px;
                    margin-bottom: 20px;
                ">
                <h3 style="color:white; margin:0;">
                    Risk Level: <span style="color:{color}">{risk}</span>
                </h3>
                <h1 style="color:white; margin:10px 0;">
                    {prob*100:.1f}% <span style="font-size: 20px">Probability</span>
                </h1>
                <p style="color:#ccc;">{res['primary_driver']}</p>
                </div>
                """, unsafe_allow_html=True)

                
                reasons, strats = explain_risk_factors(full_data, risk)
                st.markdown("### Analysis")
                for r in reasons: st.write(r)
                if risk == "High":
                    st.markdown("### Strategy")
                    for s in strats: 
                         st.markdown(f"""
                         <div style="
                             background-color: rgba(46, 204, 113, 0.25);
                             border: 2px solid #2ECC71;
                             padding: 15px 20px;
                             border-radius: 12px;
                             margin-bottom: 10px;
                             color: white;
                             font-size: 16px;
                             font-weight: 500;
                         ">
                             {s}
                        </div>
                        """, unsafe_allow_html=True)

def performance_page():
    # --- FORCE BLACK BACKGROUND FOR PERFORMANCE PAGE ---
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #000000 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.title("Navigation")
    st.sidebar.radio("Go to:", ["Predict", "Performance"], key="nav_perf", on_change=lambda: go_to(st.session_state.nav_perf.lower()))
    st.title("Model Performance Leaderboard")
    
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
        st.markdown(f"### {row['Model']}")
        
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

    # --- ADDED: COMPARATIVE GRAPH ---
    st.markdown("---")
    st.subheader("📊 Comparative Analysis")
    
    # Prepare data for Plotly
    models = df['Model'].tolist()
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    
    fig = go.Figure()
    
    # Add a bar trace for each metric
    for metric in metrics:
        fig.add_trace(go.Bar(
            name=metric,
            x=models,
            y=df[metric],
            text=df[metric].apply(lambda x: f"{x:.2f}"),
            textposition='auto'
        ))

    # Update Layout
    fig.update_layout(
        barmode='group',
        height=500,
        margin=dict(t=50, b=50, l=50, r=50),
        xaxis_title="Machine Learning Models",
        yaxis_title="Score (0-1)",
        legend_title="Metrics",
        template="plotly_dark",  # Fits the dark theme
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="DM Sans", size=14, color="white")
    )
    
    st.plotly_chart(fig, use_container_width=True)

if st.session_state.page == "home": home_page()
elif st.session_state.page == "predict": predict_page()
else: performance_page()
