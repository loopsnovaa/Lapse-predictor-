import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
import random

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="ChurnAlyse", layout="wide", page_icon="📉")

# ---------------------------------------------------------
# CSS STYLING (YOUR CUSTOM DARK THEME)
# ---------------------------------------------------------
CUSTOM_CSS = """
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap" rel="stylesheet">

<style>
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}

/* MAIN BACKGROUND – DARK BLUE */
[data-testid="stAppViewContainer"] {
    background-color: #0d3a66 !important;
    color: white !important;
}

/* SIDEBAR – LIGHTER BLUE */
[data-testid="stSidebar"] {
    background-color: #0f4c81 !important;
}

/* TEXT COLORS */
h1, h2, h3, h4, p, label, .stMarkdown {
    color: white !important;
}

/* INPUT FIELDS */
.stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>div {
    color: black !important;
    background-color: #e6f2ff !important;
    border-radius: 5px;
}

/* BUTTONS */
.stButton>button {
    background-color: #b2f7b1 !important;
    color: black !important;
    border-radius: 10px;
    border: none;
    padding: 10px 25px;
    font-size: 18px;
    font-weight: 600;
    width: 100%;
}

.stButton>button:hover {
    background-color: #A0E15E !important;
}

/* CARDS */
.metric-card {
    background-color: rgba(255, 255, 255, 0.1); 
    padding: 20px; 
    border-radius: 12px; 
    border: 1px solid rgba(255,255,255,0.2);
    margin-bottom: 20px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# API CONFIGURATION
# ---------------------------------------------------------
API_URL = "http://127.0.0.1:5000"

def get_prediction(payload):
    """Sends data to the XGBoost API"""
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        response.raise_for_status()
        return response.json()
    except: return None

def get_leaderboard():
    """Fetches the multi-model comparison data"""
    try:
        response = requests.get(f"{API_URL}/leaderboard", timeout=2)
        if response.status_code == 200:
            return response.json()
    except: return None

# ---------------------------------------------------------
# EXPLANATION LOGIC (FROM YOUR SNIPPET)
# ---------------------------------------------------------
def explain_channels(data):
    ch1 = data.get("channel1", 0)
    ch2 = data.get("channel2", 0)
    ch3 = data.get("channel3", 0)

    explanation = []
    if ch1 == 0 and ch2 == 0 and ch3 == 0:
        explanation.append("Customer came through a low-engagement channel (0,0,0) — usually walk-in, telemarketing or low-advice channels, leading to higher lapse.")
    if ch1 >= 1:
        explanation.append("Customer acquired through advisor/agent — usually lower lapse risk due to strong follow-up.")
    if ch2 >= 1:
        explanation.append("Customer acquired through digital/online channel — medium lapse due to limited counselling.")
    if ch3 >= 1:
        explanation.append("Customer bought through bancassurance channel — typically more stable with moderate lapse.")
    if len(explanation) == 0:
        explanation.append("Customer acquired through a mixed or less common channel combination.")
    return explanation

def explain_risk_factors(data, risk_level):
    reasons = []
    
    # Financial/Macro Factors (AI Driven)
    if data.get("RETENTION_POLY_QTY", 0) < data.get("PREV_POLY_INFORCE_QTY", 0):
        reasons.append("⚠️ Portfolio Shrinkage: Retention count is lower than previous in-force count.")
    if data.get("LOSS_RATIO", 0) > 1.0:
        reasons.append("⚠️ Critical Loss Ratio: Agency is losing money on claims.")

    # Demographic/Micro Factors (Your Logic)
    if data.get("premium_amount", 0) > 3000:
        reasons.append("💰 Premium amount is high (>3000), increasing price sensitivity.")
    if data.get("policy_tenure_years", 0) < 2:
        reasons.append("⏳ Short tenure (< 2 years) indicates unstable loyalty.")
    if data.get("substandard_risk", 0) == 1:
        reasons.append("🏥 Substandard risk indicator flagged.")
    if data.get("number_of_advance_premium", 0) == 0:
        reasons.append("💳 No advance premiums paid.")

    strategies = [
        "Offer premium payment reminders or auto-debit option",
        "Provide a personalized follow-up call through an agent",
        "Explain long-term benefits clearly to increase commitment"
    ]
    return reasons, strategies

# ---------------------------------------------------------
# STATE MANAGEMENT
# ---------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to(p):
    st.session_state.page = p

# ---------------------------------------------------------
# PAGES
# ---------------------------------------------------------

def home_page():
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.title("ChurnAlyse")
    st.subheader("A modern way to analyze and prevent policy lapses.")
    
    # Check API Connectivity
    if get_leaderboard():
        st.success("🟢 AI Engine Online")
    else:
        st.error("🔴 AI Engine Offline (Run api.py)")

    st.write("Predict churn, track customer behavior, and reduce lapse risk using XGBoost.")
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Start Now"):
        go_to("predict")

def predict_page():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    st.sidebar.radio("Go to:", ["Predict", "Performance"], key="nav_pred", on_change=lambda: go_to(st.session_state.nav_pred.lower()))

    st.title("Predict Policy Lapse Risk")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        with st.form("main_form"):
            st.markdown("### 1. Customer Profile")
            # --- MICRO INPUTS (Your Specific Fields) ---
            age = st.number_input("Age", 18, 80, 30)
            gender = st.selectbox("Gender", ["Female", "Male"])
            prem = st.number_input("Premium Amount", 1, 100000, 3500)
            ten = st.number_input("Policy Tenure (years)", 0.0, 20.0, 1.5)
            ch1 = st.number_input("Channel 1 (Agent)", 0, 10, 0)
            ch2 = st.number_input("Channel 2 (Digital)", 0, 10, 1)
            ch3 = st.number_input("Channel 3 (Bancassurance)", 0, 10, 0)
            sr = st.selectbox("Substandard Risk", [0, 1])
            adv = st.number_input("Advance Premium Count", 0, 10, 0)

            st.markdown("### 2. Agency Context (Required for AI)")
            st.caption("These metrics feed the High-Accuracy XGBoost Model")
            # --- MACRO INPUTS (Required for API) ---
            retention_qty = st.number_input("Retained Policies", 0, 10000, 90)
            prev_inforce = st.number_input("Prev. In-Force", 0, 10000, 100)
            curr_inforce = st.number_input("Curr. In-Force", 0, 10000, 90)
            loss_ratio = st.number_input("Loss Ratio", 0.0, 500.0, 65.0)
            loss_3yr = st.number_input("3-Year Loss Ratio", 0.0, 500.0, 60.0)
            growth = st.number_input("Growth Rate", -100.0, 100.0, 2.5)

            st.markdown("<br>", unsafe_allow_html=True)
            submit = st.form_submit_button("Predict Lapse Risk")

    if submit:
        # Prepare Payload for API (Macro Data)
        api_payload = {
            "RETENTION_POLY_QTY": retention_qty,
            "PREV_POLY_INFORCE_QTY": prev_inforce,
            "POLY_INFORCE_QTY": curr_inforce,
            "LOSS_RATIO": loss_ratio,
            "LOSS_RATIO_3YR": loss_3yr,
            "GROWTH_RATE_3YR": growth
        }
        
        # Prepare Data for Explanations (Micro Data)
        local_data = {
            "premium_amount": prem, "policy_tenure_years": ten,
            "substandard_risk": sr, "number_of_advance_premium": adv,
            "channel1": ch1, "channel2": ch2, "channel3": ch3,
            **api_payload # Merge for unified logic
        }

        # Call API
        api_res = get_prediction(api_payload)
        
        with col2:
            if api_res and "results" in api_res:
                res = api_res["results"][0]
                prob = res['confidence_score']
                risk_label = "High" if res['prediction'] == "LAPSE" else "Low"
                color = "#d00000" if risk_label == "High" else "#A0E15E"

                st.markdown(f"""
                <div class="metric-card" style="border-left: 5px solid {color};">
                    <h3>Risk Level: <span style="color:{color}">{risk_label}</span></h3>
                    <h1>{prob*100:.1f}% <span style="font-size: 20px">Lapse Probability</span></h1>
                    <p>{res['primary_driver']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Explanations
                reasons, strategies = explain_risk_factors(local_data, risk_label)
                channel_exp = explain_channels(local_data)

                st.markdown("### 🔍 Risk Analysis")
                for r in reasons:
                    st.write(r)
                
                st.markdown("### 📡 Channel Insight")
                for c in channel_exp:
                    st.write(f"- {c}")

                if risk_label == "High":
                    st.markdown("### 🛡️ Recommended Strategy")
                    for s in strategies:
                        st.info(s)
            else:
                st.warning("Could not get prediction from AI Engine.")

def performance_page():
    st.sidebar.title("Navigation")
    st.sidebar.radio("Go to:", ["Predict", "Performance"], key="nav_perf", on_change=lambda: go_to(st.session_state.nav_perf.lower()))

    st.title("Model Performance Dashboard")
    
    leaderboard = get_leaderboard()
    
    if not leaderboard:
        st.warning("⚠️ Metrics unavailable. Ensure `api.py` is running.")
        return

    # --- LEADERBOARD ---
    st.subheader("Model Leaderboard (Real-Time)")
    
    # Create comparison dataframe
    model_data = []
    for name, metrics in leaderboard.items():
        model_data.append({
            "Model": name,
            "Accuracy": metrics['accuracy'],
            "F1-Score": metrics['f1_score'],
            "AUC": metrics['auc']
        })
    df_models = pd.DataFrame(model_data).sort_values(by="Accuracy", ascending=False)
    
    # Display Metrics in Columns
    cols = st.columns(len(df_models))
    for idx, (index, row) in enumerate(df_models.iterrows()):
        with cols[idx]:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #A0E15E;">{row['Model']}</h4>
                <h1>{row['Accuracy']*100:.1f}%</h1>
                <p>Accuracy</p>
                <hr style="border-color: rgba(255,255,255,0.2);">
                <small>AUC: {row['AUC']:.3f}</small>
            </div>
            """, unsafe_allow_html=True)

    # --- BAR GRAPH ---
    st.subheader("Accuracy Comparison")
    bar_colors = ["#A0E15E", "#b2f7b1", "#8ecae6", "#219ebc"]
    
    fig = go.Figure(go.Bar(
        x=df_models["Model"],
        y=df_models["Accuracy"],
        text=[f"{v*100:.1f}%" for v in df_models["Accuracy"]],
        textposition="auto",
        marker=dict(color=bar_colors)
    ))
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        yaxis=dict(range=[0.8, 1.0]) # Zoom in to show differences
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "predict":
    predict_page()
else:
    performance_page()
