import requests
import json
import time

# 1. Define the API Endpoint URL
API_URL = "http://127.0.0.1:5000/predict_lapse"

def get_lapse_risk(policy_data):
    """Sends policy features to the deployed API and returns the risk score."""
    print(f"\n[CLIENT] Sending data for Age {policy_data['age']} to API...")
    
    # 2. Package the Data (JSON)
    # The 'requests' library handles the POST request
    try:
        response = requests.post(
            API_URL, 
            json=policy_data,
            headers={"Content-Type": "application/json"}
        )
        
        # Raise an error if the API request failed (e.g., 404 or 500)
        response.raise_for_status()
        
        # 3. Process the Response
        risk_data = response.json()
        
        return risk_data
        
    except requests.exceptions.RequestException as e:
        print(f"--- ERROR: Could not connect to API or API failed. ---")
        print(f"Details: {e}")
        return None

# --- Main simulation loop ---
if __name__ == "__main__":
    
    # Example 1: New customer data fetched when agent loads a policy page
    high_risk_policy = {
        "age": 35,
        "gender": 1,
        "policy_type_1": 2,
        "policy_type_2": 3,
        "policy_amount": 5000.0,
        "premium_amount": 100.0,
        "policy_tenure_years": 1,
        "policy_tenure_decimal": 0.9, # Very new policy, often high risk
        "channel1": 6,
        "channel2": 3,
        "channel3": 11,
        "substandard_risk": 0.0,
        "number_of_advance_premium": 0,
        "initial_benefit": 0.0
    }

    result = get_lapse_risk(high_risk_policy)
    
    if result and result.get('status') == 'success':
        risk = result['risk_level']
        prob = result['lapse_probability_percent']
        
        print("\n==============================================")
        print(f"Policy Risk Analysis (Optimized XGBoost)")
        print("==============================================")
        print(f"✅ Displaying Risk Level: {risk}")
        print(f"Risk Probability: {prob:.2f}%")
        
        # Automatically trigger the triage protocol based on the result
        if risk == "High":
            print("🚨 ACTION REQUIRED: Route this customer to a Senior Agent (SLA 4 hours).")
        elif risk == "Medium":
            print("🟡 ACTION REQUIRED: Flag for Standard Agent follow-up (SLA 48 hours).")
        else:
            print("🟢 ACTION: No immediate intervention required.")