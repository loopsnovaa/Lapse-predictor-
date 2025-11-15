import streamlit as st
import requests
import plotly.graph_objects as go

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="ChurnAlyse", layout="wide")

# ---------------------------------------------------------
# GLOBAL CSS STYLING
# ---------------------------------------------------------
CUSTOM_CSS = """
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap" rel="stylesheet">

<style>
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background-color: #c6def1 !important;
}

[data-testid="stSidebar"] {
    background-color: #eef5ff !important;
}

/* BUTTON — pastel light green */
.stButton>button {
    background-color: #b2f7b1 !important;
    color: black !important;
    border-radius: 10px;
    border: none;
    padding: 10px 25px;
    font-size: 18px;
    font-weight: 600;
}

.stButton>button:hover {
    background-color: #9be99a !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# EXPLANATION FUNCTIONS (NEW + CLEAN)
# ---------------------------------------------------------

# 🟩 LOW RISK → PROTECTIVE FACTORS ONLY
def explain_low_risk(data):
    reasons = []

    if data["premium_amount"] <= 3000:
        reasons.append("Premium amount is affordable")

    if data["number_of_advance_premium"] > 0:
        reasons.append("Customer has paid advance premiums consistently")

    if data["substandard_risk"] == 0:
        reasons.append("Customer has no substandard risk indicators")

    if data["policy_tenure_years"] >= 3:
        reasons.append("Policy tenure is stable and long-term")

    if data["channel1"] < 5 and data["channel2"] < 5:
        reasons.append("Sales channels indicate stable customer behavior")

    if 25 <= data["age"] <= 55:
        reasons.append("Customer is in a low-risk age range")

    if len(reasons) == 0:
        reasons.append("Customer shows strong stability factors")

    return reasons


# 🟧 MEDIUM RISK → moderate or mixed indicators
def explain_medium_risk(data):
    reasons = []

    if data["premium_amount"] > 3000:
        reasons.append("Premium amount is moderately high")

    if data["policy_tenure_years"] < 2.5:
        reasons.append("Policy tenure is slightly short")

    if data["number_of_advance_premium"] == 0:
        reasons.append("Customer has no advance premium payments")

    if data["channel1"] >= 4 or data["channel2"] >= 4:
        reasons.append("Sales channel history increases moderate risk")

    if data["age"] < 25 or data["age"] > 55:
        reasons.append("Customer is outside the low-risk age range")

    if len(reasons) == 0:
        reasons.append("Model detected moderate risk based on internal patterns")

    return reasons


# 🟥 HIGH RISK → strong risk drivers
def explain_high_risk(data):
    reasons = []

    if data["policy_amount"] > 500000:
        reasons.append("High policy amount increases financial burden")

    if data["premium_amount"] > 3000:
        reasons.append("Premium amount is high relative to affordability")

    if data["policy_tenure_years"] < 2:
        reasons.append("Very short remaining policy tenure")

    if data["substandard_risk"] == 1:
        reasons.append("Customer is marked as substandard risk")

    if data["number_of_advance_premium"] == 0:
        reasons.append("No advance premium payments recorded")

    if data["channel1"] >= 5 or data["channel2"] >= 5:
        reasons.append("Sales channel indicators correlate with high lapse risk")

    if data["age"] < 25 or data["age"] > 55:
        reasons.append("Customer belongs to a high-risk age segment")

    if len(reasons) == 0:
        reasons.append("Model detected high risk based on internal patterns")

    return reasons


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to(page):
    st.session_state.page = page


# ---------------------------------------------------------
# HOME PAGE (UNCHANGED)
# ---------------------------------------------------------
def home_page():
    st.markdown("<br><br><br>", unsafe_allow_html=True)

    st.title("ChurnAlyse")
    st.subheader("A modern way to analyze and prevent policy lapses.")
    st.write("Designed for insurance teams, powered by machine learning.")
    st.write("Predict churn, monitor risk, and save customers proactively.")
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Start Now"):
        go_to("predict")


# ---------------------------------------------------------
# PREDICTION PAGE
# ---------------------------------------------------------
def predict_page():

    st.sidebar.title("Navigation")
    st.sidebar.radio("Go to:", ["Predict", "Performance"],
                     key="nav_predict",
                     on_change=lambda: go_to(st.session_state.nav_predict.lower()))

    st.title("Predict Policy Lapse Risk")

    # INPUTS
    age = st.number_input("Age", 18, 80, 30)

    gender = st.selectbox("Gender", ["Female", "Male"])
    gender = 0 if gender == "Female" else 1

    policy_type_1 = st.number_input("Policy Type 1", 1, 10, 3)
    policy_type_2 = st.number_input("Policy Type 2", 1, 40, 4)
    policy_amount = st.number_input("Policy Amount", 1000, 1000000, 50000)
    premium_amount = st.number_input("Premium Amount", 1, 100000, 200)
    tenure = st.number_input("Policy Tenure (years)", 0.0, 20.0, 2.0)
    tenure_dec = st.number_input("Policy Tenure Decimal", 0.0, 10.0, 1.5)
    channel1 = st.number_input("Channel 1", 0, 10, 2)
    channel2 = st.number_input("Channel 2", 0, 10, 2)
    channel3 = st.number_input("Channel 3", 0, 10, 1)
    subrisk = st.selectbox("Substandard Risk", [0, 1])
    adv = st.number_input("Advance Premium Count", 0, 10, 1)
    initial_benefit = st.number_input("Initial Benefit", 0, 2000000, 10000)

    # PREDICT BUTTON
    if st.button("Predict"):

        data = {
            "age": age,
            "gender": gender,
            "policy_type_1": policy_type_1,
            "policy_type_2": policy_type_2,
            "policy_amount": policy_amount,
            "premium_amount": premium_amount,
            "policy_tenure_years": tenure,
            "policy_tenure_decimal": tenure_dec,
            "channel1": channel1,
            "channel2": channel2,
            "channel3": channel3,
            "substandard_risk": subrisk,
            "number_of_advance_premium": adv,
            "initial_benefit": initial_benefit
        }

        try:
            response = requests.post("http://127.0.0.1:5000/predict_lapse", json=data)

            if response.status_code == 200:
                result = response.json()

                st.subheader(f"Risk Level: **{result['risk_level']}**")
                st.write(f"Lapse Probability: **{result['lapse_probability_percent']}%**")

                st.subheader("Why this customer got this risk result")

                # ----------------------------------------
                # HIGH RISK
                # ----------------------------------------
                if result["risk_level"] == "High":
                    st.markdown("""
                    <div style="
                        background-color:#f8d7da;
                        padding:15px;
                        border-radius:12px;
                        font-size:22px;
                        font-weight:700;
                        color:#b43434;
                        margin-top:10px;
                        margin-bottom:15px;">
                        High-Risk Customer
                    </div>
                    """, unsafe_allow_html=True)

                    for r in explain_high_risk(data):
                        st.write(f"- {r}")

                    st.markdown("---")
                    st.warning("### Recommended Retention Strategies")
                    st.write("""
                    - Immediate agent outreach  
                    - Premium grace period extension  
                    - Offer payment flexibility  
                    - Personalized communication  
                    """)

                # ----------------------------------------
                # MEDIUM RISK
                # ----------------------------------------
                elif result["risk_level"] == "Medium":
                    st.markdown("### Customer Risk Factors")

                    for r in explain_medium_risk(data):
                        st.write(f"- {r}")

                # ----------------------------------------
                # LOW RISK
                # ----------------------------------------
                else:
                    st.markdown("### Customer Protective Factors")

                    for r in explain_low_risk(data):
                        st.write(f"- {r}")

            else:
                st.error("API Error: Could not fetch prediction")

        except Exception as e:
            st.error(f"Could not connect to API: {e}")


# ---------------------------------------------------------
# PERFORMANCE PAGE
# ---------------------------------------------------------
def performance_page():

    st.sidebar.title("Navigation")
    st.sidebar.radio("Go to:", ["Predict", "Performance"],
                     key="nav_perf",
                     on_change=lambda: go_to(st.session_state.nav_perf.lower()))

    st.title("Model Performance Dashboard")

    try:
        stats = requests.get("http://127.0.0.1:5000/model_stats").json()

        st.subheader("Overall Prediction Summary")
        st.write(f"**Total Predictions:** {stats['total_predictions']}")
        st.write(f"**Average Predicted Risk:** {stats['average_predicted_risk']:.3f}")

        labels = ["Low", "Medium", "High"]
        values = [
            stats["low_risk_count"],
            stats["medium_risk_count"],
            stats["high_risk_count"]
        ]

        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
        fig.update_layout(title_text="Risk Level Distribution")
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error loading stats: {e}")


# ---------------------------------------------------------
# ROUTER
# ---------------------------------------------------------
if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "predict":
    predict_page()
elif st.session_state.page == "performance":
    performance_page()
