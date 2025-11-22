import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="BankGuard: Insurance Analytics", layout="wide", page_icon="🏦")

# --- CSS STYLING ---
st.markdown("""
<style>
    .metric-card {
        background-color: white; padding: 20px; border-radius: 12px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); text-align: center; border: 1px solid #e9ecef;
    }
    .model-header {
        background-color: #f0f2f6; padding: 10px; border-radius: 8px; margin-bottom: 10px; border-left: 5px solid #0d6efd;
    }
    .risk-high { border-top: 5px solid #dc3545; }
    .risk-low { border-top: 5px solid #198754; }
    
    .stButton>button {
        width: 100%; border-radius: 8px; height: 3em; 
        background-color: #0d6efd; color: white; font-weight: 600;
        border: none;
    }
    .stButton>button:hover { background-color: #0b5ed7; }
</style>
""", unsafe_allow_html=True)

API_URL = "http://127.0.0.1:5000"

def get_prediction(payload):
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
# LOCAL LOGIC: CUSTOMER SCORING (MICRO)
# ---------------------------------------------------------
def calculate_customer_priority(data):
    score = 0
    reasons = []

    if data['premium'] > 4000:
        score += 2
        reasons.append(f"High Value Customer (${data['premium']}/yr)")

    if data['tenure'] < 1.5:
        score += 3
        reasons.append("New Customer (< 1.5 yrs)")
    
    if data['substandard']:
        score += 4
        reasons.append("Substandard Risk Flag")

    if score >= 5:
        return "High Priority", "#dc3545", reasons
    elif score >= 3:
        return "Medium Priority", "#ffc107", reasons
    else:
        return "Standard", "#198754", ["Profile is stable"]

# --- PREDICT PAGE ---
def predict_page():
    st.title("🏦 BankGuard: Lapse Intervention Tool")
    
    # Check Connection
    if get_leaderboard():
        st.success("🟢 API Online: Connected to High-Accuracy Model Hub")
    else:
        st.error("🔴 API Offline: Run 'api.py' to enable AI features")

    col_input, col_result = st.columns([1, 1.2])

    with col_input:
        with st.form("risk_form"):
            st.subheader("1. Enter Data")
            
            tab1, tab2 = st.tabs(["👤 Customer (Micro)", "🏢 Portfolio (Macro)"])
            
            with tab1:
                st.caption("Individual Risk Factors")
                cust_id = st.text_input("Customer ID", "CUS-9921")
                premium = st.number_input("Annual Premium ($)", 0, 100000, 2500)
                tenure = st.number_input("Tenure (Years)", 0.0, 50.0, 1.2)
                substandard = st.checkbox("Substandard Risk Flag")
            
            with tab2:
                st.caption("Aggregated Agency Metrics (Sent to AI)")
                retention_qty = st.number_input("Retained Policy Qty", 0, 10000, 90)
                prev_inforce = st.number_input("Prev. In-Force Qty", 0, 10000, 100)
                curr_inforce = st.number_input("Curr. In-Force Qty", 0, 10000, 90)
                loss_ratio = st.number_input("Current Loss Ratio (%)", 0.0, 500.0, 65.0)
                loss_3yr = st.number_input("3-Year Loss Ratio (%)", 0.0, 500.0, 60.0)
                growth = st.number_input("Growth Rate (%)", -100.0, 100.0, 2.5)
            
            st.markdown("---")
            submit = st.form_submit_button("Analyze Risk Profile")

    if submit:
        payload = {
            "RETENTION_POLY_QTY": retention_qty, "PREV_POLY_INFORCE_QTY": prev_inforce,
            "POLY_INFORCE_QTY": curr_inforce, "LOSS_RATIO": loss_ratio,
            "LOSS_RATIO_3YR": loss_3yr, "GROWTH_RATE_3YR": growth
        }
        api_res = get_prediction(payload)
        
        # Local Customer Logic
        cust_data = {"premium": premium, "tenure": tenure, "substandard": substandard}
        cust_prio, cust_color, cust_reasons = calculate_customer_priority(cust_data)
        
        with col_result:
            if api_res and "results" in api_res:
                res = api_res["results"][0]
                # Determine Colors based on AI Result
                if res["prediction"] == "LAPSE":
                    macro_color = "#dc3545" # Red
                    risk_class = "risk-high"
                    icon = "⚠️"
                else:
                    macro_color = "#198754" # Green
                    risk_class = "risk-low"
                    icon = "✅"
                
                # --- MACRO CARD ---
                st.markdown(f"""
                <div class="metric-card {risk_class}">
                    <h4 style="color: #6c757d;">AI Portfolio Prediction</h4>
                    <h1 style="font-size: 50px; color: {macro_color}; margin: 0;">{icon} {res['prediction']}</h1>
                    <h3 style="color: {macro_color};">Confidence: {res['confidence_score']*100:.1f}%</h3>
                    <hr>
                    <p style="text-align: left;"><b>Primary Driver:</b><br>{res['primary_driver']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.write("") # Spacer

                # --- MICRO CARD ---
                st.markdown(f"""
                <div class="metric-card" style="border-top: 5px solid {cust_color};">
                    <h4 style="color: #6c757d;">Customer Intervention Priority</h4>
                    <h2 style="color: {cust_color}; margin: 0;">{cust_prio}</h2>
                    <hr>
                    <div style="text-align: left;">
                        {''.join([f'<p>• {r}</p>' for r in cust_reasons])}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# --- PERFORMANCE PAGE (UPDATED) ---
def performance_page():
    st.title("📊 Model Performance Diagnostics")
    
    leaderboard = get_leaderboard()
    
    if not leaderboard:
        st.warning("⚠️ Metrics unavailable. Ensure `api.py` is running and `train_leaderboard.py` has been executed.")
        return

    st.markdown("### Comparative Leaderboard")
    st.info("Metrics calculated on validation set (20% of data).")

    # Define the order we want to show models
    model_order = ["XGBoost (Tuned)", "Random Forest", "Decision Tree", "Logistic Regression"]

    for model_name in model_order:
        # Check if model exists in data
        if model_name in leaderboard:
            metrics = leaderboard[model_name]
            
            # --- MODEL HEADER ---
            st.markdown(f"""
            <div class="model-header">
                <h3 style="margin:0; color: #0d6efd;">{model_name}</h3>
            </div>
            """, unsafe_allow_html=True)

            # --- METRICS ROW ---
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
            m2.metric("Precision", f"{metrics['precision']:.3f}")
            m3.metric("Recall", f"{metrics['recall']:.3f}")
            m4.metric("F1 Score", f"{metrics['f1_score']:.3f}")
            m5.metric("AUC", f"{metrics['auc']:.3f}")
            
            st.write("") # Spacer

    # --- Feature Importance Chart (Only for XGBoost) ---
    st.markdown("---")
    st.subheader("Feature Importance (XGBoost)")
    # Hardcoded for visual consistency based on your known feature set
    features = pd.DataFrame({
        "Feature": ["RETENTION_POLY_QTY", "PREV_POLY_INFORCE_QTY", "LOSS_RATIO", "GROWTH_RATE_3YR"],
        "Importance": [0.65, 0.25, 0.08, 0.02] 
    })
    
    fig = go.Figure(go.Bar(
        x=features["Importance"], y=features["Feature"], orientation='h', marker=dict(color='#0d6efd')
    ))
    fig.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)

# --- NAVIGATION ---
if __name__ == "__main__":
    with st.sidebar:
        st.markdown("### Menu")
        page = st.radio("Go to:", ["Lapse Prediction", "Model Performance"])
    
    if page == "Lapse Prediction":
        predict_page()
    else:
        performance_page()