import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
import os
import json
st.set_page_config(
    page_title="ChurnAlyse AI", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")  
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_new.joblib")
FEATURE_PATH = os.path.join(MODEL_DIR, "feature_names.joblib")
LEADERBOARD_PATH = os.path.join(MODEL_DIR, "leaderboard.json")

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #0e1117; color: #fafafa; }
    [data-testid="stSidebar"] { background-color: #262730; }
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background-color: #2d3748; color: white; border-radius: 5px; border: 1px solid #4a5568;
    }
    .stButton>button {
        background-color: #A0E15E; color: #000; border: none; 
        font-weight: bold; border-radius: 8px; width: 100%; padding: 0.5rem;
    }
    .stButton>button:hover { background-color: #b2f7b1; }
    .metric-card {
        background-color: #1e293b; padding: 20px; border-radius: 10px;
        border: 1px solid #334155; margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        features = joblib.load(FEATURE_PATH)
        return model, scaler, features
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

@st.cache_data
def get_leaderboard():
    if not os.path.exists(LEADERBOARD_PATH): return None
    try:
        with open(LEADERBOARD_PATH, 'r') as f: return json.load(f)
    except: return None

model, scaler, feature_names = load_artifacts()

def predict_churn(payload):
    if not model or not scaler: return None, 0.0

    df = pd.DataFrame([payload])
    df = df[feature_names]
    
    X_scaled = scaler.transform(df)
    pred_class = model.predict(X_scaled)[0]
    pred_prob = model.predict_proba(X_scaled)[0][1]

    return pred_class, pred_prob

def page_predict():
    st.title("🛡️ Policy Lapse Risk Predictor")
    
    if not model:
        st.error(f"⚠️ Model not found at {MODEL_PATH}. Run 'train_full.py' first.")
        return

    col_input, col_res = st.columns([1, 1.2], gap="large")

    with col_input:
        st.subheader("1. Customer Profile")
        c1, c2 = st.columns(2)
        age = c1.number_input("Age", 18, 100, 30)
        tenure = c2.number_input("Tenure (Years)", 0.0, 50.0, 3.5)
        premium = st.number_input("Premium Amount", 0, 1000000, 3500)
        
        st.write("Channel")
        ch_cols = st.columns(3)
        agent_ch = ch_cols[0].checkbox("Agent", False)
        digital_ch = ch_cols[1].checkbox("Digital", True)
        banca_ch = ch_cols[2].checkbox("Bancassurance", False)
        
        st.markdown("---")
        st.subheader("2. Performance Metrics")
        
        p1, p2 = st.columns(2)
        prev_qty = p1.number_input("Prev. In-Force Qty", 1, 10000, 100)
        ret_qty = p2.number_input("Retained Qty", 0, 10000, 90)
        
        p3, p4 = st.columns(2)
        loss_r = p3.number_input("Loss Ratio (%)", 0.0, 200.0, 65.0)
        loss_3yr = p4.number_input("3-Yr Loss Ratio", 0.0, 200.0, 60.0)
        growth = st.number_input("Growth Rate (%)", -100.0, 100.0, 2.5)
        
        btn = st.button("🚀 Analyze Risk")

    if btn:
        payload = {
            "AGE": age,
            "PREMIUM": premium,
            "TENURE": tenure,
            "AGENT_CHANNEL": int(agent_ch),
            "DIGITAL_CHANNEL": int(digital_ch),
            "BANCASSURANCE": int(banca_ch),
            "RETENTION_POLY_QTY": ret_qty,
            "PREV_POLY_INFORCE_QTY": prev_qty,
            "LOSS_RATIO": loss_r,
            "LOSS_RATIO_3YR": loss_3yr,
            "GROWTH_RATE_3YR": growth
        }

        pred, prob = predict_churn(payload)

        with col_res:
            is_risk = prob > 0.5
            color = "#ef4444" if is_risk else "#22c55e"
            status = "HIGH RISK" if is_risk else "SAFE"
            
            st.markdown(f"""
            <div class="metric-card" style="border-left: 8px solid {color}">
                <h2 style="margin:0; color:{color}">{status}</h2>
                <h1 style="font-size: 3.5rem; margin:0">{prob:.1%}<span style="font-size: 1rem; color: #aaa"> Probability</span></h1>
            </div>
            """, unsafe_allow_html=True)
            
            categories = ['Retention Gap', 'Loss Ratio', 'Growth Lag']

            ret_gap = max(0, (prev_qty - ret_qty) / prev_qty)
            loss_norm = min(1, loss_r / 100.0)
            growth_inv = min(1, max(0, (10 - growth)/20))

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=[ret_gap, loss_norm, growth_inv],
                theta=categories,
                fill='toself',
                line_color=color,
                name='Current Policy'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                margin=dict(t=20, b=20, l=40, r=40)
            )
            st.plotly_chart(fig, use_container_width=True)

            if is_risk:
                st.error("⚠️ **Action Required:** Policy is likely to lapse.")
                st.info(f"📉 Retention dropped by {prev_qty - ret_qty} policies.")
            else:
                st.success("✅ **Good Standing:** Policy is stable.")

def page_performance():
    st.title("🏆 Model Leaderboard")
    data = get_leaderboard()
    if not data:
        st.warning("No leaderboard data found.")
        return

    df = pd.DataFrame.from_dict(data, orient='index').sort_values(by="accuracy", ascending=False)
    
    # Top Card
    best = df.iloc[0]
    st.markdown(f"""
    <div class="metric-card">
        <h3>🥇 Best Model: {df.index[0]}</h3>
        <h1>Accuracy: {best['accuracy']:.2%}</h1>
        <p>F1 Score: {best['f1_score']:.4f}</p>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(df.style.highlight_max(axis=0, color='#A0E15E'), use_container_width=True)

def main():
    with st.sidebar:
        st.title("ChurnAlyse")
        page = st.radio("Navigation", ["Predict Risk", "Model Leaderboard"])
        st.markdown("---")
        st.caption("v2.0 | High Accuracy Build")

    if page == "Predict Risk":
        page_predict()
    else:
        page_performance()

if __name__ == "__main__":
    main()
