import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
import os
import sys
import json

# ---------------------------------------------------------
# 1. SETUP & CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="ChurnAlyse AI", 
    layout="wide", 
    page_icon="📉",
    initial_sidebar_state="expanded"
)

# Constants (Pointing to the successfully trained model)
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_new.joblib")
FEATURE_PATH = os.path.join(MODEL_DIR, "feature_names.joblib")
LEADERBOARD_PATH = os.path.join(MODEL_DIR, "leaderboard.json")

# ---------------------------------------------------------
# 2. CUSTOM CSS & ANIMATION
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
    .circles li:nth-child(6) { left: 75%; width: 110px; height: 110px; animation-delay: 3s; animation-duration: 15s; }
    .circles li:nth-child(7) { left: 35%; width: 150px; height: 150px; animation-delay: 7s; animation-duration: 10s; }

    @keyframes animate {
        0% { transform: translateY(0) rotate(0deg); opacity: 1; border-radius: 50%; }
        100% { transform: translateY(-1000px) rotate(720deg); opacity: 0; border-radius: 50%; }
    }
    /* Title Glow Animation */
@keyframes neon-glow {
    0%, 100% { 
        text-shadow: 0 0 1px #fff, 0 0 5px #2ecc71, 0 0 10px #2ecc71; 
        color: #fff;
    }
    50% { 
        text-shadow: 0 0 2px #fff, 0 0 15px #3498db, 0 0 25px #3498db; 
        color: #ddd;
    }
}
.glowing-text {
    animation: neon-glow 4s ease-in-out infinite alternate;
}

    /* UI ELEMENTS */
    .block-container { z-index: 10; position: relative; }
    .stButton>button { background-color: #2ecc71 !important; color: white !important; border-radius: 8px; border: none; padding: 10px 24px; font-weight: 600; transition: all 0.3s ease; }
    .stButton>button:hover { transform: scale(1.02); }
    .metric-card { background-color: rgba(255, 255, 255, 0.05); padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); backdrop-filter: blur(10px); margin-bottom: 15px; }
    .stTextInput>div>div>input, .stNumberInput>div>div>input { background-color: #0b1e33 !important; color: white !important; border: 1px solid rgba(255,255,255,0.2) !important; }
    
    /* HIDE SIDEBAR ON HOMEPAGE */
    .st-emotion-cache-163d83s { display: none; }
</style>
""", unsafe_allow_html=True)

st.markdown('<ul class="circles"><li></li><li></li><li></li><li></li><li></li><li></li><li></li><li></li><li></li><li></li></ul>', unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. DATA LOADING
# ---------------------------------------------------------
# ---------------------------------------------------------
# 3. DATA LOADING (FINAL CORRECT STRUCTURE)
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """Loads model, scaler, and feature list safely."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_order = joblib.load(FEATURE_PATH)
        return model, scaler, feature_order
    except:
        return None, None, []

def get_leaderboard():
    """Loads leaderboard JSON data without caching."""
    if not os.path.exists(LEADERBOARD_PATH): return None
    try:
        with open(LEADERBOARD_PATH, 'r') as f: 
            return json.load(f)
    except: 
        return None

# THIS LINE IS CRUCIAL AND MUST BE PLACED AFTER THE FUNCTION DEFINITIONS
model, scaler, feature_order = load_artifacts()
# ---------------------------------------------------------
# 4. PREDICTION LOGIC
# ---------------------------------------------------------
def make_prediction(payload):
    if not model or not scaler: return None
        
    try:
        df = pd.DataFrame([payload])
        final_df = pd.DataFrame()
        
        # Align all inputs to the trained feature order
        for col in feature_order:
            final_df[col] = df[col] if col in df.columns else 0.0
            
        X_input = scaler.transform(final_df)
        pred = model.predict(X_input)[0]
        prob = model.predict_proba(X_input)[0][1]
        
        return {"risk": "High" if pred==1 else "Low", "score": prob}
    except:
        return None

# ---------------------------------------------------------
# 5. PAGES
# ---------------------------------------------------------
# Navigation state management
if "page" not in st.session_state: st.session_state.page = "home"
def go_to(p):
    st.session_state.page = p
    st.rerun() 


def home_page():
    def home_page():
    # --- RESTORED HOMEPAGE UI ---
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    
    # Title with the new glowing animation class
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
        # THE DEFINITIVE BUTTON FIX
        if st.button("Start Risk Analysis", use_container_width=True, key='home_btn'): 
            go_to("Predict")
def predict_page():
    st.title("🔮 Lapse Risk Predictor")

    c1, c2 = st.columns([1, 1.3])
    with c1:
        with st.form("risk_form"):
            st.markdown("### 1. Customer")
            c3, c4 = st.columns(2)
            age = c3.number_input("Age", 18, 100, 30, key='age')
            tenure = c4.number_input("Tenure (Yrs)", 0.0, 50.0, 3.5, key='tenure')
            prem = st.number_input("Premium Amount", 0, 100000, 3500, key='prem')
            
            st.markdown("### 2. Channels")
            ch1, ch2, ch3 = st.columns(3)
            agent_ch = ch1.checkbox("Agent", True)
            digital_ch = ch2.checkbox("Digital", False)
            banca_ch = ch3.checkbox("Bancassurance", False)

            st.markdown("### 3. Metrics (High-Accuracy Drivers)")
            p1, p2 = st.columns(2)
            ret = p1.number_input("Retained Qty", 0, 5000, 90, help="RETENTION_POLY_QTY")
            prev = p2.number_input("Previous Qty", 0, 5000, 100, help="PREV_POLY_INFORCE_QTY")
            
            p3, p4 = st.columns(2)
            loss = p3.number_input("Loss Ratio", 0.0, 500.0, 65.0)
            loss3 = p4.number_input("3-Yr Loss Ratio", 0.0, 500.0, 60.0)
            growth = st.number_input("Growth %", -100.0, 100.0, 2.5)
            
            submitted = st.form_submit_button("Analyze Risk")
            
    if submitted:
        # Construct payload with ALL 11 FEATURES
        payload = {
            "AGE": age, "PREMIUM": prem, "TENURE": tenure,
            "AGENT_CHANNEL": int(agent_ch), "DIGITAL_CHANNEL": int(digital_ch), "BANCASSURANCE": int(banca_ch),
            "RETENTION_POLY_QTY": ret, "PREV_POLY_INFORCE_QTY": prev,
            "LOSS_RATIO": loss, "LOSS_RATIO_3YR": loss3, "GROWTH_RATE_3YR": growth
        }
        
        res = make_prediction(payload)
        
        with c2:
            if res:
                color = "#ef4444" if res['risk'] == "High" else "#2ecc71"
                st.markdown(f"""
                <div class="metric-card" style="border-left: 8px solid {color}; text-align: left;">
                    <h3 style="color:{color}; margin:0;">RISK LEVEL: {res['risk'].upper()}</h3>
                    <h1 style="font-size: 4rem; margin: 10px 0;">{res['score']:.1%}<span style="font-size: 1rem; color: #aaa"> Probability</span></h1>
                </div>
                """, unsafe_allow_html=True)
                
                # Strategies
                st.markdown("### Analysis")
                if ret < prev: st.warning("⚠️ Portfolio Shrinkage detected (Retention Gap).")
                if loss > 100: st.error("⚠️ Critical Loss Ratio (>100%). Review claims.")
                st.markdown("### Strategy")
                st.info("Offer personalized agent follow-up and premium reminders.")
                
                # Radar Chart
                categories = ['Retention Gap', 'Loss Ratio', 'Growth Lag']
                ret_gap = max(0, (prev - ret) / prev) if prev > 0 else 0
                loss_norm = min(1, loss / 100.0)
                growth_inv = min(1, max(0, (10 - growth)/20))

                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=[ret_gap, loss_norm, growth_inv], theta=categories, fill='toself', name='Current Policy', line_color=color))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), margin=dict(t=20, b=20, l=40, r=40))
                st.plotly_chart(fig, use_container_width=True)


def performance_page():
    st.title("🏆 Model Performance Leaderboard")
    
    # Load the JSON data
    data = get_leaderboard()
    
    if not data:
        st.error("❌ Leaderboard data not found. Please ensure 'train_full.py' ran successfully.")
        return

    # Convert dictionary to DataFrame, sort by accuracy (highest first)
    df = pd.DataFrame.from_dict(data, orient='index')
    
    if 'accuracy' in df.columns:
        # Sort in descending order (highest accuracy first)
        df_sorted = df.sort_values(by='accuracy', ascending=False)
    else:
        df_sorted = df
        st.warning("Could not sort metrics as 'accuracy' column is missing.")

    st.markdown("---")

    # Iterate through the SORTED DataFrame rows
    for model_name, metrics in df_sorted.iterrows():
        
        st.markdown(f"### {model_name}")
        
        cols = st.columns(5)
        
        metric_keys = ["accuracy", "precision", "recall", "f1_score", "auc"]
        
        for i, key in enumerate(metric_keys):
            value = metrics.get(key, 0.0)
            
            # Use the correct display format (percentage for accuracy, float for others)
            if key == "accuracy":
                display_value = f"{value:.2%}"
            else:
                display_value = f"{value:.3f}"
            
            with cols[i]:
                st.markdown(f"""
                <div class="metric-card" style="text-align: center; padding: 10px;">
                    <p style="margin:0; font-size: 14px; color: #aaa">{key.upper().replace('_', ' ')}</p>
                    <h4 style="margin: 5px 0; font-size: 1.5rem; color: #2ecc71;">{display_value}</h4>
                </div>
                """, unsafe_allow_html=True)
                
        st.markdown("---")

# ---------------------------------------------------------
# 6. MAIN NAVIGATION
# ---------------------------------------------------------
def main():
    
    # 1. Handle sidebar visibility based on page state
    if st.session_state.page in ("home", "Home"):
        # Hide sidebar completely on the homepage
        st.markdown('<style> [data-testid="stSidebar"] {display: none;} </style>', unsafe_allow_html=True)
    
    # 2. Render Sidebar Navigation for other pages
    else:
        with st.sidebar:
            st.title("Navigation")
            page = st.radio("Go to", ["Home", "Predict", "Performance"], label_visibility="collapsed", key='nav_main', index=["Home", "Predict", "Performance"].index(st.session_state.page))
            st.markdown("---")
            if model:
                st.caption(f"🟢 Model Loaded")
            else:
                st.caption("🔴 No Model Found")
            
            # Sidebar navigation action
            if page != st.session_state.page:
                st.session_state.page = page
                st.rerun()

    # 3. Render the current page
    if st.session_state.page in ("home", "Home"):
        home_page()
    elif st.session_state.page == "Predict":
        predict_page()
    else: # Performance
        performance_page()

if __name__ == "__main__":
    main()
