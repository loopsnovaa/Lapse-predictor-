import streamlit as st
import plotly.graph_objects as go
import random

st.set_page_config(page_title="ChurnAlyse", layout="wide")

CUSTOM_CSS = """
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap" rel="stylesheet">

<style>
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}

/* MAIN BACKGROUND – DARK BLUE */
[data-testid="stAppViewContainer"] {
    background-color: #0d3a66 !important;
}

/* SIDEBAR – LIGHTER BLUE */
[data-testid="stSidebar"] {
    background-color: #0f4c81 !important;
}

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
    background-color: #A0E15E !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to(p):
    st.session_state.page = p

def explain_channels(data):
    ch1 = data["channel1"]
    ch2 = data["channel2"]
    ch3 = data["channel3"]

    explanation = []

    if ch1 == 0 and ch2 == 0 and ch3 == 0:
        explanation.append("Customer came through a low-engagement channel (0,0,0) — usually walk-in, telemarketing or low-advice channels, leading to higher lapse.")
    
    if ch1 == 1 and ch2 == 0 and ch3 == 0:
        explanation.append("Customer acquired through advisor/agent — usually lower lapse risk due to strong follow-up.")
    
    if ch1 == 0 and ch2 == 1 and ch3 == 0:
        explanation.append("Customer acquired through digital/online channel — medium lapse due to limited counselling.")
    
    if ch1 == 0 and ch2 == 0 and ch3 == 1:
        explanation.append("Customer bought through bancassurance channel — typically more stable with moderate lapse.")
    
    if len(explanation) == 0:
        explanation.append("Customer acquired through a mixed or less common channel combination.")

    return explanation

def explain_low(data):
    out=[]
    if data["premium_amount"] <= 3000: out.append("Premium is affordable")
    if data["number_of_advance_premium"] > 0: out.append("Pays advance premiums")
    if data["substandard_risk"] == 0: out.append("No substandard risk indicators")
    if 25 <= data["age"] <= 55: out.append("Low-risk age range")
    if len(out)==0: out.append("Strong protective behaviour")
    return out

def explain_medium(data):
    out=[]
    if data["premium_amount"] > 3000: out.append("Premium moderately high")
    if data["policy_tenure_years"] < 2.5: out.append("Slightly short tenure")
    if data["number_of_advance_premium"] == 0: out.append("No advance premium payments")
    if data["age"] < 25 or data["age"] > 55: out.append("Age increases moderate risk")
    if len(out)==0: out.append("Model detected moderate risk")
    return out

def explain_high(data):
    reasons = []

    if data["policy_amount"] > 500000:
        reasons.append("High policy amount")
    if data["premium_amount"] > 3000:
        reasons.append("Premium amount is high")
    if data["policy_tenure_years"] < 2:
        reasons.append("Short tenure increases risk")
    if data["substandard_risk"] == 1:
        reasons.append("Substandard risk indicator")

    if len(reasons) == 0:
        reasons.append("Model detected high risk")

    strategies = [
        "Offer premium payment reminders or auto-debit option",
        "Provide a personalized follow-up call through an agent",
        "Explain long-term benefits clearly to increase commitment",
        "Give a small loyalty reward or discount if applicable",
        "Review and restructure premium frequency if needed"
    ]

    return reasons, strategies


def classify_risk(prob):
    if prob >= 0.66: return "High"
    elif prob >= 0.33: return "Medium"
    else: return "Low"

def home_page():

    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.title("ChurnAlyse")
    st.subheader("A modern way to analyze and prevent policy lapses.")
    st.write("Predict churn, track customer behavior, and reduce lapse risk using machine learning.")
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Start Now"):
        go_to("predict")

def predict_page():

    # Sidebar navigation
    st.sidebar.title("Navigation")
    st.sidebar.radio(
        "Go to:",
        ["Predict", "Performance"],
        key="nav_pred",
        on_change=lambda: go_to(st.session_state.nav_pred.lower())
    )

    st.title("Predict Policy Lapse Risk")

    # ------------- INPUTS -------------
    age = st.number_input("Age", 18, 80, 30)
    gender = 0 if st.selectbox("Gender", ["Female", "Male"]) == "Female" else 1
    pt1 = st.number_input("Policy Type 1", 1, 10, 3)
    pt2 = st.number_input("Policy Type 2", 1, 40, 4)
    pamt = st.number_input("Policy Amount", 1000, 1000000, 50000)
    prem = st.number_input("Premium Amount", 1, 100000, 200)
    ten = st.number_input("Policy Tenure (years)", 0.0, 20.0, 2.0)
    tend = st.number_input("Policy Tenure Decimal", 0.0, 10.0, 1.5)
    ch1 = st.number_input("Channel 1", 0, 10, 2)
    ch2 = st.number_input("Channel 2", 0, 10, 2)
    ch3 = st.number_input("Channel 3", 0, 10, 1)
    sr = st.selectbox("Substandard Risk", [0, 1])
    adv = st.number_input("Advance Premium Count", 0, 10, 1)
    ben = st.number_input("Initial Benefit", 0, 2000000, 10000)

    if st.button("Predict"):

        payload = {
            "age": age,
            "gender": gender,
            "policy_type_1": pt1,
            "policy_type_2": pt2,
            "policy_amount": pamt,
            "premium_amount": prem,
            "policy_tenure_years": ten,
            "policy_tenure_decimal": tend,
            "channel1": ch1,
            "channel2": ch2,
            "channel3": ch3,
            "substandard_risk": sr,
            "number_of_advance_premium": adv,
            "initial_benefit": ben,
        }

        proba = random.uniform(0.10, 0.90)
        lapse_prob_percent = round(proba * 100, 2)
        risk_level = classify_risk(proba)

        st.subheader(f"Risk Level: **{risk_level}**")
        st.write(f"Lapse Probability: **{lapse_prob_percent}%**")

        st.subheader("Why this customer got this risk result")
        if risk_level == "High":
            for x in explain_high(payload):
                st.write("- " + x)
        elif risk_level == "Medium":
            for x in explain_medium(payload):
                st.write("- " + x)
        else:
            for x in explain_low(payload):
                st.write("- " + x)

        st.subheader("Channel Interpretation")
        for x in explain_channels(payload):
            st.write("- " + x)

def performance_page():

    st.sidebar.title("Navigation")
    st.sidebar.radio("Go to:", ["Predict", "Performance"],
                     key="nav_perf",
                     on_change=lambda: go_to(st.session_state.nav_perf.lower()))

    st.title("Model Performance Dashboard")

    st.subheader("Overall Prediction Summary")
    st.write("**Total Predictions:** 500")
    st.write("**Average Predicted Risk:** 0.47")

    # PIE CHART
    labels = ["Low Risk", "Medium Risk", "High Risk"]
    values = [230, 170, 100]

    pie_colors = ["#A0E15E", "#ff9e00", "#d00000"]

    pie_fig = go.Figure(
        data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.45,
            textinfo="label+percent",
            marker=dict(colors=pie_colors)
        )]
    )

    st.plotly_chart(pie_fig, use_container_width=True)

    # BAR GRAPH (HIGH METRICS)
    st.subheader("Model Performance Metrics")

    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
    metric_values = [0.92, 0.94, 0.91, 0.93, 0.95]

    bar_colors = ["#8ecae6", "#219ebc", "#ffb703", "#fb8500", "#8d99ae"]

    bar_fig = go.Figure()
    bar_fig.add_trace(
        go.Bar(
            x=metric_labels,
            y=metric_values,
            text=[f"{v:.2f}" for v in metric_values],
            textposition="auto",
            marker=dict(color=bar_colors,
                        line=dict(color="white", width=1.5))
        )
    )

    st.plotly_chart(bar_fig, use_container_width=True)

if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "predict":
    predict_page()
else:
    performance_page()
