import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
import os
import sys

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="ChurnAlyse AI", 
    layout="wide", 
    page_icon="📉",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. PATHS (Pointing to the HIGH-ACCURACY MODEL)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Note: Feature and Scaler paths are fixed to match the output of train_full.py
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler_new.joblib")
FEATURE_PATH = os.path.join(BASE_DIR, "models", "feature_names.joblib") # Changed to feature_names.joblib

# --- (CSS code remains unchanged for aesthetics) ---

st.markdown("""
<style>
     /* CSS STYLING HERE */
    [data-testid="stAppViewContainer"] { background: radial-gradient(circle at center, #0e2a47 0%, #000000 100%); color: white; overflow-x: hidden; }
    [data-testid="stSidebar"] { background-color: #0b1e33; border-right: 1px solid rgba(255,255,255,0.1); }
    /* FLOATING CIRCLES */
    .circles { position: fixed; top: 0; left: 0; width: 100%; height: 100%; overflow: hidden; z-index: 0; pointer-events: none; }
    .circles li { position: absolute; display: block; list-style: none; width: 20px; height: 20px; background: rgba(46, 204, 113, 0.2); animation: animate 25s linear infinite; bottom: -150px; border-radius: 50%; }
    .circles li:nth-child(1) { left: 25%; width: 80px; height: 80px; animation-delay: 0s; }
    .circles li:nth-child(2) { left: 10%; width: 20px; height: 20px; animation-delay: 2s; animation-duration: 12s; }
    .circles li:nth-child(3) { left: 70%; width: 20px; height: 20px; animation-delay: 4s; }
    .circles li:nth-child(4) { left: 40%; width: 60px; height: 60px; animation-delay: 0s; animation-duration: 18s; }
    .circles li:nth-child(5) { left: 65%; width: 20px; height: 20px; animation-delay: 0s; }
    @keyframes animate { 0% { transform: translateY(0) rotate(0deg); opacity: 1; border-radius: 50%; } 100% { transform: translateY(-1000px) rotate(720deg); opacity: 0; border-radius: 50%; } }
    .circles li:nth-child(6) { left: 75%; width: 110px; height: 110px; animation-delay: 3s; animation-duration: 15s; }
    .circles li:nth-child(7) { left: 35%; width: 150px; height: 150px; animation-delay: 7s; animation-duration: 10s; }
    .stButton>button { background-color: #2ecc71 !important; color: white !important; border-radius: 8px; border: none; padding: 10px 24px; font-weight: 600; transition: all 0.3s ease; }
    .stButton>button:hover { transform: scale(1.02); }
    .metric-card { background-color: rgba(255, 255, 255, 0.05); padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); backdrop-filter: blur(10px); margin-bottom: 15px; }
    .stTextInput>div>div>input, .stNumberInput>div>div>input { background-color: #0b1e33 !important; color: white !important; border: 1px solid rgba(255,255,255,0.2) !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<ul class="circles"><li></li><li></li><li></li><li></li><li></li><li></li><li></li><li></li><li></li><li></li></ul>', unsafe_allow_html=True)


# ---------------------------------------------------------
# 4. LOAD ARTIFACTS
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        # Load best_model.joblib
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_order = joblib.load(FEATURE_PATH)
        return model, scaler, feature_order
    except Exception as e:
        # Fallback error reporting
        st.error(f"❌ CRITICAL ERROR: Could not load artifacts. Did you run the 'train_full.py' script? Error: {e}")
        return None, None, []

model, scaler, feature_order = load_artifacts()

# ---------------------------------------------------------
# 5. PREDICTION LOGIC (FIXED PAYLOAD ALIGNMENT)
# ---------------------------------------------------------
def make_prediction(payload):
    if model is None: return None
        
    try:
        df = pd.DataFrame([payload])
        
        # 1. Dynamic Alignment to 11 features
        final_df = pd.DataFrame()
        for col in feature_order:
            # Map inputs to the required feature list, filling 0 if UI input is missing
            final_df[col] = df[col] if col in df.columns else 0.0
            
        # 2. Scale
        X_input = scaler.transform(final_df)
        
        # 3. Predict
        pred = model.predict(X_input)[0]
        prob = model.predict_proba(X_input)[0][1]
        
        return {"risk": "High" if pred==1 else "Low", "score": prob}
    except Exception as e:
        st.error(f"Prediction Error (Check Console for trace): {e}")
        return None

# ---------------------------------------------------------
# 6. PAGES
# ---------------------------------------------------------
if "page" not in st.session_state: st.session_state.page = "predict"
def go_to(p): st.session_state.page = p

def predict_page():
    st.title("🔮 Lapse Risk Predictor")

    c1, c2 = st.columns([1, 1.3])
    with c1:
        with st.form("risk_form"):
            # --- RESTORED ALL 11 INPUTS ---
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
            
            # Additional UI input that is not a model feature (removed from payload)
            # curr = st.number_input("Curr. Qty (Display)", 0, 5000, 90) 
            
            submitted = st.form_submit_button("Analyze Risk")
            
    if submitted:
        # --- FIXED PAYLOAD TO MATCH 11 FEATURES ---
        payload = {
            # UI Inputs (Demographics/Channels)
            "AGE": age,
            "PREMIUM": prem,
            "TENURE": tenure,
            "AGENT_CHANNEL": int(agent_ch),
            "DIGITAL_CHANNEL": int(digital_ch),
            "BANCASSURANCE": int(banca_ch),

            # Core Performance Inputs (Accuracy Drivers)
            "RETENTION_POLY_QTY": ret, 
            "PREV_POLY_INFORCE_QTY": prev,
            "LOSS_RATIO": loss, 
            "LOSS_RATIO_3YR": loss3, 
            "GROWTH_RATE_3YR": growth
        }
        # ---------------------------------------------
        
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
                
                # Radar Chart (simplified for space)
                categories = ['Retention Gap', 'Loss Ratio', 'Growth Lag']
                ret_gap = max(0, (prev - ret) / prev) if prev > 0 else 0
                loss_norm = min(1, loss / 100.0)
                growth_inv = min(1, max(0, (10 - growth)/20))

                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=[ret_gap, loss_norm, growth_inv], theta=categories, fill='toself', name='Current Policy', line_color=color))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), margin=dict(t=20, b=20, l=40, r=40))
                st.plotly_chart(fig, use_container_width=True)


def performance_page():
    # Placeholder for the Performance page logic (not fully implemented in your snippet)
    st.title("🏆 Model Leaderboard")
    st.warning("Leaderboard not fully implemented in this script version, but your model is saved!")
    st.info("Run `train_full.py` to ensure the `leaderboard.json` exists.")


def main():
    with st.sidebar:
        st.title("ChurnAlyse")
        st.markdown("Cloud-Native Insurance Analytics")
        st.markdown("---")
        
        page = st.radio("Go to", ["Predict", "Performance"], label_visibility="collapsed", key='nav_main')
        
        st.markdown("---")
        if model:
            st.caption(f"🟢 Model Loaded: XGBoost")
            st.caption(f" Features: {len(feature_order)}")
        else:
            st.caption("🔴 No Model Found")

    if page == "Predict":
        predict_page()
    else:
        performance_page()

if __name__ == "__main__":
    main()
