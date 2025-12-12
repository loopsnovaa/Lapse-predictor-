import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
import os
import sys
import json
import random

# ---------------------------------------------------------
# 1. SETUP & CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="ChurnAlyse AI", 
    layout="wide", 
    page_icon="📉",
    initial_sidebar_state="expanded"
)

# Constants
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_new.joblib")
FEATURE_PATH = os.path.join(MODEL_DIR, "feature_names.joblib")
LEADERBOARD_PATH = os.path.join(MODEL_DIR, "leaderboard.json")

# ---------------------------------------------------------
# 2. GLOBAL CSS FIXES (WIDENING & LAYERING)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* --- 1. WIDEN THE APP (Fixing the "Empty Space") --- */
    .block-container {
        max-width: 95% !important;  /* Widen to 95% of the screen */
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 2rem !important;
    }

    /* --- 2. FORCE BLACK BACKGROUND --- */
    [data-testid="stAppViewContainer"] {
        background-color: black !important;
        background-image: none !important;
    }
    
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* --- 3. SIDEBAR STYLE --- */
    [data-testid="stSidebar"] {
        background-color: #0b1e33;
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    /* --- 4. TEXT GLOW --- */
    @keyframes neon-glow {
        0%, 100% { text-shadow: 0 0 1px #fff, 0 0 5px #2ecc71, 0 0 10px #2ecc71; color: #fff; }
        50% { text-shadow: 0 0 2px #fff, 0 0 15px #3498db, 0 0 25px #3498db; color: #ddd; }
    }
    .glowing-text {
        animation: neon-glow 4s ease-in-out infinite alternate;
    }
    
    /* --- 5. UI ELEMENTS --- */
    .stButton>button { 
        background-color: #2ecc71 !important; 
        color: white !important; 
        border-radius: 8px; 
        border: none; 
        padding: 10px 24px; 
        font-weight: 600; 
        width: 100%; /* Make buttons fill their column */
    }
    .metric-card { 
        background-color: rgba(255, 255, 255, 0.05); 
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid rgba(255,255,255,0.1); 
        margin-bottom: 15px; 
    }
    .stTextInput>div>div>input, .stNumberInput>div>div>input { 
        background-color: #0b1e33 !important; 
        color: white !important; 
    }
    
    /* Hide sidebar specifically on Home if needed, otherwise keep it */
    /* .st-emotion-cache-163d83s { display: none; } */
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# 3. ANIMATION: FALLING LINES (HOME)
# ---------------------------------------------------------
def inject_falling_lines_bg():
    st.markdown("""
    <style>
        .lines {
            position: fixed; top: 0; left: 0; right: 0; height: 100vh; width: 100vw;
            display: flex; justify-content: space-between; 
            z-index: 0; /* Background layer */
            pointer-events: none; 
        }
        .line { position: relative; width: 1px; height: 100%; background: rgba(255,255,255,0.05); overflow: hidden; }
        .line::after {
            content: ''; display: block; position: absolute; height: 15vh; width: 100%; top: -50%; left: 0;
            background: linear-gradient(to bottom, rgba(255, 255, 255, 0) 0%, #ffffff 75%, #ffffff 100%);
            animation: drop 1.5s 0s infinite; animation-fill-mode: forwards; animation-timing-function: cubic-bezier(0.4, 0.26, 0, 0.97);
        }
        .line:nth-child(1)::after { background: #FF4500; animation-delay: 0.1s; }
        .line:nth-child(2)::after { background: #32CD32; animation-delay: 0.3s; }
        .line:nth-child(3)::after { background: #1E90FF; animation-delay: 0.5s; }
        .line:nth-child(4)::after { background: #FFD700; animation-delay: 0.7s; }
        .line:nth-child(5)::after { background: #8A2BE2; animation-delay: 0.9s; }
        .line:nth-child(6)::after { background: #20B2AA; animation-delay: 1.1s; }
        .line:nth-child(7)::after { background: #DC143C; animation-delay: 1.3s; }
        .line:nth-child(8)::after { background: #00FA9A; animation-delay: 1.5s; }
        .line:nth-child(9)::after { background: #FF1493; animation-delay: 1.7s; }
        .line:nth-child(10)::after { background: #00BFFF; animation-delay: 1.9s; }
        @keyframes drop { 0% { top: -50%; } 100% { top: 110%; } }
    </style>
    <div class="lines"><div class="line"></div><div class="line"></div><div class="line"></div><div class="line"></div><div class="line"></div><div class="line"></div><div class="line"></div><div class="line"></div><div class="line"></div><div class="line"></div></div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 4. ANIMATION: PARTICLE ORB (PREDICT/PERFORMANCE)
# ---------------------------------------------------------
@st.cache_resource
def get_particle_orb_css():
    """Generates the CSS for the Rotating Particle Orb"""
    total = 150 # Reduced slightly for better performance
    orb_size = 120
    time = 14
    
    # Z-INDEX 0 puts it behind the form (z-index 1 implied)
    css = f"""
    <style>
    .wrap {{
        position: fixed;
        top: 50%;
        left: 50%;
        width: 0; 
        height: 0; 
        transform-style: preserve-3d;
        perspective: 1000px;
        animation: rotate {time}s infinite linear;
        z-index: 0; 
        pointer-events: none;
    }}
    
    @keyframes rotate {{
        100% {{ transform: rotateY(360deg) rotateX(360deg); }}
    }}
    
    .c {{
        position: absolute;
        width: 4px; 
        height: 4px;
        border-radius: 50%;
        opacity: 1; 
    }}
    </style>
    """
    
    rng = random.Random(42) 
    keyframe_styles = ""
    
    for i in range(1, total + 1):
        z = rng.randint(0, 360)
        y = rng.randint(0, 360)
        hue = (40 / total * i)
        
        keyframe_styles += f"""
        .c:nth-child({i}) {{
            animation: orbit{i} {time}s infinite;
            animation-delay: {i * 0.01}s; 
            background-color: hsla({hue}, 100%, 50%, 1);
        }}
        @keyframes orbit{i} {{ 
            0% {{ transform: rotateZ(-{z}deg) rotateY({y}deg) translateX({orb_size}px) rotateZ({z}deg); }} 
            100% {{ transform: rotateZ(-{z}deg) rotateY({y}deg) translateX({orb_size}px) rotateZ({z}deg); }}
        }}
        """
    
    return css + "<style>" + keyframe_styles + "</style>"

def inject_particle_orb():
    st.markdown(get_particle_orb_css(), unsafe_allow_html=True)
    total = 150
    particles = "".join(['<div class="c"></div>' for _ in range(total)])
    st.markdown(f'<div class="wrap">{particles}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# 5. DATA LOADING & MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_order = joblib.load(FEATURE_PATH)
        return model, scaler, feature_order
    except:
        return None, None, []

def get_leaderboard():
    if not os.path.exists(LEADERBOARD_PATH): return None
    try:
        with open(LEADERBOARD_PATH, 'r') as f: 
            return json.load(f)
    except: 
        return None

model, scaler, feature_order = load_artifacts()

def make_prediction(payload):
    if not model or not scaler: return None
    try:
        df = pd.DataFrame([payload])
        final_df = pd.DataFrame()
        for col in feature_order:
            final_df[col] = df[col] if col in df.columns else 0.0
        X_input = scaler.transform(final_df)
        pred = model.predict(X_input)[0]
        prob = model.predict_proba(X_input)[0][1]
        return {"risk": "High" if pred==1 else "Low", "score": prob}
    except:
        return None

# ---------------------------------------------------------
# 6. PAGES
# ---------------------------------------------------------
if "page" not in st.session_state: st.session_state.page = "home"
def go_to(p):
    st.session_state.page = p
    st.rerun() 

def home_page():
    inject_falling_lines_bg()
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h1 class='glowing-text' style='font-size: 80px; text-align: center;'>ChurnAlyse</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='opacity: 0.8; font-weight: 300; text-align: center;'>Predict churn, monitor risk, and save customers proactively.</h3>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Widened Layout for the Button
    c1, c2, c3 = st.columns([1, 2, 1]) 
    with c2:
        if model:
            st.markdown("""<div style="background: rgba(46, 204, 113, 0.2); border: 1px solid #2ecc71; padding: 12px; border-radius: 50px; text-align: center; margin-bottom: 20px;"><span style="color: #2ecc71; font-weight: bold;">● ML Engine Loaded</span></div>""", unsafe_allow_html=True)
        else:
            st.error("🔴 Error: Model not found.")
        
        if st.button("Start Risk Analysis", use_container_width=True, key='home_btn'): 
            go_to("Predict")

def predict_page():
    # --- ORB ANIMATION ---
    inject_particle_orb()
    
    st.title("🔮 Lapse Risk Predictor")
    
    # Widened form layout
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

            st.markdown("### 3. Metrics")
            p1, p2 = st.columns(2)
            ret = p1.number_input("Retained Qty", 0, 5000, 90, help="RETENTION_POLY_QTY")
            prev = p2.number_input("Previous Qty", 0, 5000, 100, help="PREV_POLY_INFORCE_QTY")
            p3, p4 = st.columns(2)
            loss = p3.number_input("Loss Ratio", 0.0, 500.0, 65.0)
            loss3 = p4.number_input("3-Yr Loss Ratio", 0.0, 500.0, 60.0)
            growth = st.number_input("Growth %", -100.0, 100.0, 2.5)
            
            submitted = st.form_submit_button("Analyze Risk", use_container_width=True)
            
    if submitted:
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
                
                categories = ['Retention Gap', 'Loss Ratio', 'Growth Lag']
                ret_gap = max(0, (prev - ret) / prev) if prev > 0 else 0
                loss_norm = min(1, loss / 100.0)
                growth_inv = min(1, max(0, (10 - growth)/20))

                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=[ret_gap, loss_norm, growth_inv], theta=categories, fill='toself', name='Current Policy', line_color=color))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), margin=dict(t=20, b=20, l=40, r=40))
                st.plotly_chart(fig, use_container_width=True)

def performance_page():
    inject_particle_orb()

    st.title("🏆 Model Performance Leaderboard")
    data = get_leaderboard()
    if not data:
        st.error("❌ Leaderboard data not found.")
        return

    df = pd.DataFrame.from_dict(data, orient='index')
    if 'accuracy' in df.columns:
        df_sorted = df.sort_values(by='accuracy', ascending=False)
    else:
        df_sorted = df
        st.warning("Could not sort metrics as 'accuracy' column is missing.")

    st.markdown("---")
    for model_name, metrics in df_sorted.iterrows():
        st.markdown(f"### {model_name}")
        cols = st.columns(5)
        metric_keys = ["accuracy", "precision", "recall", "f1_score", "auc"]
        for i, key in enumerate(metric_keys):
            value = metrics.get(key, 0.0)
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
# 7. MAIN NAVIGATION
# ---------------------------------------------------------
def main():
    if st.session_state.page in ("home", "Home"):
        st.markdown('<style> [data-testid="stSidebar"] {display: none;} </style>', unsafe_allow_html=True)
    else:
        with st.sidebar:
            st.title("Navigation")
            page = st.radio("Go to", ["Home", "Predict", "Performance"], label_visibility="collapsed", key='nav_main', index=["Home", "Predict", "Performance"].index(st.session_state.page))
            st.markdown("---")
            if model:
                st.caption(f"🟢 Model Loaded")
            else:
                st.caption("🔴 No Model Found")
            
            if page != st.session_state.page:
                st.session_state.page = page
                st.rerun()

    if st.session_state.page in ("home", "Home"):
        home_page()
    elif st.session_state.page == "Predict":
        predict_page()
    else: 
        performance_page()

if __name__ == "__main__":
    main()
