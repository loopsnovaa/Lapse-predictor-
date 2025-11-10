import streamlit as st
import requests
import json
import pandas as pd

# Define the API Endpoint URL (must match the address where your api.py is running)
API_URL = "http://127.0.0.1:5000/predict_lapse"

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Lapse Risk Predictor", layout="centered")

st.title("🛡️ Lapse Risk Prediction Dashboard")
st.markdown("Enter policy details to receive an instant risk assessment and triage recommendation.")
st.divider()

# --- Functions ---

def get_risk_assessment(data):
    """Sends policy features to the deployed API and returns the structured risk assessment."""
    try:
        # Send POST request with JSON data
        response = requests.post(
            API_URL, 
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        response.raise_for_status() # Raise error for bad status codes
        return response.json()
        
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to Prediction API. Please ensure the API is running at {API_URL}.")
        st.code(f"Details: {e}")
        return None

def display_triage_recommendation(risk_level, probability):
    """Displays the colored, actionable result based on the Triage Protocol."""
    
    st.subheader("Action Protocol")
    
    if risk_level == "High":
        st.error(f"🚨 URGENT ACTION: HIGH RISK ({probability:.2f}%)")
        st.markdown("**Action:** Route to Senior Agent. **SLA:** Call policyholder within **4 hours**.")
    elif risk_level == "Medium":
        st.warning(f"🟡 PROACTIVE ACTION: MEDIUM RISK ({probability:.2f}%)")
        st.markdown("**Action:** Flag for Standard Agent. **SLA:** Call policyholder within **48 hours**.")
    else:
        st.success(f"🟢 MONITORING: LOW RISK ({probability:.2f}%)")
        st.markdown("**Action:** No immediate manual intervention required. Continue monitoring.")
        
    st.markdown("---")

# --- User Input Form ---

# The inputs are grouped to guide the user and match your model's 15 input features

with st.form("policy_form"):
    st.markdown("#### Customer and Policy Basics")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        policy_tenure_years = st.number_input("Policy Tenure (Years)", min_value=0, max_value=10, value=1)
        policy_amount = st.number_input("Policy Amount (Benefit)", min_value=100.0, value=10000.0)
        channel1 = st.selectbox("Channel 1 (Primary)", options=[1, 2, 3, 4, 5, 6], index=5)
        channel3 = st.selectbox("Channel 3 (Tertiary)", options=[0, 1, 2, 3, 4], index=0)
        substandard_risk = st.selectbox("Substandard Risk", options=[0.0, 1.0], index=0)

    with col2:
        # Note: The selectbox returns a tuple (Label, Value), so we extract the [1] value
        gender = st.selectbox("Gender", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0], index=0)[1] 
        policy_tenure_decimal = st.number_input("Policy Tenure (Decimal)", min_value=0.0, max_value=10.0, value=0.9, format="%.2f")
        premium_amount = st.number_input("Premium Amount", min_value=0.0, value=500.0)
        policy_type_1 = st.selectbox("Policy Type 1", options=[1, 2, 3, 4], index=0)
        policy_type_2 = st.selectbox("Policy Type 2", options=[1, 2, 3, 4], index=1)
        channel2 = st.selectbox("Channel 2 (Secondary)", options=[1, 2, 3, 4], index=2)
        number_of_advance_premium = st.number_input("Advance Premiums Paid", min_value=0, value=0)
        initial_benefit = st.number_input("Initial Benefit", min_value=0.0, value=0.0)

    st.markdown("---")
    submitted = st.form_submit_button("Predict Lapse Risk")

# --- Submission Logic ---
if submitted:
    # Package Input Data to match the API's expectation (15 features)
    input_data = {
        "age": age,
        "gender": gender,
        "policy_type_1": policy_type_1,
        "policy_type_2": policy_type_2,
        "policy_amount": policy_amount,
        "premium_amount": premium_amount,
        "policy_tenure_years": policy_tenure_years,
        "policy_tenure_decimal": policy_tenure_decimal,
        "channel1": channel1,
        "channel2": channel2,
        "channel3": channel3,
        "substandard_risk": substandard_risk,
        "number_of_advance_premium": number_of_advance_premium,
        "initial_benefit": initial_benefit,
    }

    # Call the API
    results = get_risk_assessment(input_data)
    
    if results and results['status'] == 'success':
        # Display the Results and Triage Protocol
        display_triage_recommendation(
            results['risk_level'], 
            results['lapse_probability_percent']
        )
        
        st.sidebar.success("Prediction Complete!")
        st.sidebar.json(results)

# --- Python Entry Point ---
if __name__ == "__main__":
    # Note: The API server (api.py) must be running in a separate window!
    pass