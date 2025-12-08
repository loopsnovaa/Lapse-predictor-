import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
import os
import json

# ---------------------------------------------------------
# 1. SETUP & CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="ChurnAlyse AI", 
    layout="wide", 
    #page_icon="📉",
    initial_sidebar_state="expanded"
)

# Constants
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_optimized_model_new.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_new.joblib")
FEATURE_PATH = os.path.join(MODEL_DIR, "training_feature_order_new.joblib")
LEADERBOARD_PATH = os.path.join(MODEL_DIR, "leaderboard.json")

# CSS Styling
st.markdown("""
<style>
    /* Main Background & Fonts */
    [data-testid="stAppViewContainer"] { background-color: #0e1117; color: #fafafa; }
    [data-testid="stSidebar"] { background-color: #262730; }
    
    /* Inputs */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background-color: #2d3748; color: white; border-radius: 5px; border: 1px solid #4a5568;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #A0E15E; color: #000; border: none; 
        font-weight: bold; border-radius: 8px; width: 100%; padding: 0.5rem;
    }
    .stButton>button:hover { background-color: #b2f7b1; box-shadow: 0 4px 12px rgba(160, 225, 94, 0.3); }
    
    /* Cards */
    .metric-card {
        background-color: #1e293b; padding: 20px; border-radius: 10px;
        border: 1px solid #334155; margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1, h2, h3 { font-family: 'DM Sans', sans-serif; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. DATA LOADING
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """Load model, scaler, and feature list safely."""
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
    """Load leaderboard JSON."""
    if not os.path.exists(LEADERBOARD_PATH):
        return None
    try:
        with open(LEADERBOARD_PATH, 'r') as f:
            return json.load(f)
    except:
        return None

# Initialize
model, scaler, feature_order = load_artifacts()

# ---------------------------------------------------------
# 3. PREDICTION ENGINE
# ---------------------------------------------------------
def predict_churn(payload):
    """Encapsulates prediction logic."""
    if not model or not scaler:
        return None

    # Create DF and align columns
    df = pd.DataFrame([payload])
    
    # Ensure all columns expected by the model exist, default to 0
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0
            
    # Sort columns to match training order exactly
    X = df[feature_order]
    
    # Scale
    X_scaled = scaler.transform(X)
    
    # Predict
    pred_class = model.predict(X_scaled)[0]
    
    # Handle probability (some models don't support predict_proba)
    try:
        pred_prob = model.predict_proba(X_scaled)[0][1]
    except:
        pred_prob = 1.0 if pred_class == 1 else 0.0

    return pred_class, pred_prob

# ---------------------------------------------------------
# 4. PAGES
# ---------------------------------------------------------

def page_predict():
    st.title("Lapse Risk Predictor")
    
    if not model:
        st.warning("⚠️ Model artifacts not found. Please train the model first.")
        return

    col_input, col_res = st.columns([1, 1.5], gap="large")

    with col_input:
        st.subheader("Policy Metrics")
        with st.form("pred_form"):
            # Agency Performance Inputs
            c1, c2 = st.columns(2)
            ret_qty = c1.number_input("Retained Policies", 0, 5000, 90, help="Number of policies renewed")
            prev_qty = c2.number_input("Previous In-Force", 0, 5000, 100, help="Total policies at start of period")
            
            c3, c4 = st.columns(2)
            loss_r = c3.number_input("Current Loss Ratio (%)", 0.0, 500.0, 65.0)
            loss_3yr = c4.number_input("3-Yr Avg Loss Ratio", 0.0, 500.0, 60.0)
            
            growth = st.number_input("Growth Rate (%)", -100.0, 500.0, 2.5)
            
            # Additional Context (Not used in model but useful for strategy)
            st.markdown("---")
            st.caption("Contextual Data (Non-Model)")
            prem = st.number_input("Premium Amount", 0, 1000000, 3500)
            
            submitted = st.form_submit_button("Analyze Risk")

    if submitted:
        # Prepare Payload
        payload = {
            "RETENTION_POLY_QTY": ret_qty,
            "PREV_POLY_INFORCE_QTY": prev_qty,
            "LOSS_RATIO": loss_r,
            "LOSS_RATIO_3YR": loss_3yr,
            "GROWTH_RATE_3YR": growth
        }

        # Predict
        pred, prob = predict_churn(payload)

        with col_res:
            # Result Card
            is_risk = pred == 1
            color = "#ef4444" if is_risk else "#22c55e" # Red or Green
            status = "HIGH RISK" if is_risk else "SAFE"
            
            st.markdown(f"""
            <div class="metric-card" style="border-left: 8px solid {color}">
                <h2 style="margin:0; color:{color}">{status}</h2>
                <h1 style="font-size: 3rem; margin:0">{prob:.1%}<span style="font-size: 1rem; color: #aaa"> Probability</span></h1>
                <p style="margin-top:10px">Prediction based on <b>{len(feature_order)}</b> agency performance factors.</p>
            </div>
            """, unsafe_allow_html=True)

            # Analysis Tabs
            tab1, tab2 = st.tabs(["📊 Visual Analysis", "💡 AI Recommendations"])
            
            with tab1:
                # Radar Chart for Visual Context
                categories = ['Retention Gap', 'Loss Ratio', 'Growth Lag']
                
                # Normalize values for simple visualization (0-1 scale approx)
                ret_gap = max(0, (prev_qty - ret_qty) / prev_qty) if prev_qty > 0 else 0
                loss_norm = min(1, loss_r / 100.0)
                growth_inv = min(1, max(0, (10 - growth)/20)) # Higher is worse for this chart
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=[ret_gap, loss_norm, growth_inv],
                    theta=categories,
                    fill='toself',
                    name='Current Policy',
                    line_color=color
                ))
                fig.add_trace(go.Scatterpolar(
                    r=[0.1, 0.4, 0.3],
                    theta=categories,
                    fill='toself',
                    name='Safe Threshold',
                    line_color='gray',
                    opacity=0.3
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    margin=dict(t=20, b=20, l=40, r=40),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white")
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                if is_risk:
                    st.error("⚠️ Retention Alert: Portfolio is shrinking.")
                    if loss_r > 80:
                        st.warning(f"⚠️ High Claims: Loss Ratio is {loss_r}%. Review underwriting quality.")
                    st.info("Strategy: Initiate 'Save Squad' call sequence. Offer loyalty discount if Premium > 5000.")
                else:
                    st.success("✅ Healthy Portfolio Metrics.")
                    st.markdown("* Monitor 3-year growth trends.\n* Cross-sell opportunities available.")

def page_performance():
    st.title("Comparative Analysis")
    
    data = get_leaderboard()
    if not data:
        st.info("No leaderboard data found. Run training script first.")
        return

    # Convert to DF
    df = pd.DataFrame.from_dict(data, orient='index')
    df.index.name = 'Model'
    df.reset_index(inplace=True)
    df = df.sort_values(by="accuracy", ascending=False)

    # Top Metric Cards
    best_model = df.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Best Model", best_model['Model'])
    c2.metric("Top Accuracy", f"{best_model['accuracy']:.2%}")
    c3.metric("Top AUC", f"{best_model['auc']:.3f}")

    st.divider()

    # Dynamic Grid for Model Cards
    cols = st.columns(3)
    for i, (index, row) in enumerate(df.iterrows()):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="metric-card">
                <h4>{row['Model']}</h4>
                <div style="display:flex; justify-content:space-between; align-items:end;">
                    <span style="font-size: 2rem; font-weight:bold; color: #A0E15E;">{row['accuracy']:.1%}</span>
                    <span style="color: #aaa;">Acc</span>
                </div>
                 <div style="display:flex; justify-content:space-between; margin-top:5px;">
                    <span>F1-Score: <b>{row['f1_score']:.2f}</b></span>
                    <span>AUC: <b>{row['auc']:.2f}</b></span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Comparison Chart
    st.subheader("Metric Comparison")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['Model'], y=df['accuracy'], name='Accuracy', marker_color='#A0E15E'
    ))
    fig.add_trace(go.Bar(
        x=df['Model'], y=df['auc'], name='AUC', marker_color='#219ebc'
    ))
    fig.update_layout(
        barmode='group',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# 5. MAIN NAVIGATION
# ---------------------------------------------------------
def main():
    with st.sidebar:
        st.title("ChurnAlyse")
        st.markdown("Cloud-Native Insurance Analytics")
        st.markdown("---")
        
        # New Streamlit Navigation Pattern
        page = st.radio("Go to", ["Predict Risk", "Model Leaderboard"], label_visibility="collapsed")
        
        st.markdown("---")
        if model:
            st.caption(f"🟢 Model Loaded: XGBoost")
            st.caption(f" Features: {len(feature_order)}")
        else:
            st.caption("🔴 No Model Found")

    if page == "Predict Risk":
        page_predict()
    else:
        page_performance()

if __name__ == "__main__":
    main()