import os
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="ChurnAlyse", layout="wide")

# ---------------------------------------------------------
# GLOBAL CSS STYLING
# ---------------------------------------------------------
# GLOBAL CSS STYLING
CUSTOM_CSS = """
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap" rel="stylesheet">

<style>
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}

/* MAIN BACKGROUND */
[data-testid="stAppViewContainer"] {
    background-color: #072540 !important;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #0d3a66 !important;
    color: white !important;
}

/* REMOVE FLOATING TEXT */
[data-testid="stMarkdownContainer"] p {
    color: white !important;
}

/* GREEN BUTTONS */
.stButton>button {
    background-color: #A0E15E !important;
    color: black !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 10px 25px !important;
    font-size: 18px !important;
    font-weight: 600 !important;
}

/* NORMAL OTHER BUTTONS */
button[kind="secondary"] {
    background-color: #1a1a1a !important;
    color: white !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# MODEL PATHS
# ---------------------------------------------------------
MODEL_PATH = "models/xgboost_optimized_model.joblib"
SCALER_PATH = "models/scaler.joblib"
FEATURE_ORDER_PATH = "models/training_feature_order.joblib"
METRICS_PATH = "models/model_metrics.joblib"

os.makedirs("logs", exist_ok=True)
PREDICTION_LOG = "logs/predictions.log"

# ---------------------------------------------------------
# LOAD ARTIFACTS
# ---------------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_order = joblib.load(FEATURE_ORDER_PATH)

    if os.path.exists(METRICS_PATH):
        metrics = joblib.load(METRICS_PATH)
    else:
        metrics = {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "auc": None,
        }

    return model, scaler, feature_order, metrics

MODEL, SCALER, FEATURE_ORDER, TRAIN_METRICS = load_model_artifacts()

# ---------------------------------------------------------
# EXPLANATIONS
# ---------------------------------------------------------
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
    out=[]
    if data["policy_amount"] > 500000: out.append("High policy amount")
    if data["premium_amount"] > 3000: out.append("Premium amount is high")
    if data["policy_tenure_years"] < 2: out.append("Short tenure increases risk")
    if data["substandard_risk"] == 1: out.append("Substandard risk indicator")
    if len(out)==0: out.append("Model detected high risk")
    return out

# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------
def preprocess_input(data_dict):
    df = pd.DataFrame([data_dict])

    df["premium_to_benefit_ratio"] = df["premium_amount"] / (df["policy_amount"] + 1)
    df["age_squared"] = df["age"] ** 2
    df["premium_squared"] = df["premium_amount"] ** 2
    df["benefit_squared"] = df["policy_amount"] ** 2

    df = df[FEATURE_ORDER]

    X_scaled = SCALER.transform(df)
    return X_scaled

# ---------------------------------------------------------
# PREDICT + LOG
# ---------------------------------------------------------
def predict_and_log(data_dict):
    X_scaled = preprocess_input(data_dict)
    proba = float(MODEL.predict_proba(X_scaled)[0][1])

    if proba < 0.30:
        risk = "Low"
    elif proba < 0.70:
        risk = "Medium"
    else:
        risk = "High"

    record = {
        "timestamp": datetime.now().isoformat(),
        "input": data_dict,
        "predicted_probability": proba,
        "risk_level": risk,
    }

    with open(PREDICTION_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")

    return proba, risk

# ---------------------------------------------------------
# LOAD STATS
# ---------------------------------------------------------
def load_prediction_stats():
    if not os.path.exists(PREDICTION_LOG):
        return {
            "total_predictions": 0,
            "average_predicted_risk": 0.0,
            "low_risk_count": 0,
            "medium_risk_count": 0,
            "high_risk_count": 0,
        }

    records=[]
    with open(PREDICTION_LOG,"r") as f:
        for line in f:
            try: records.append(json.loads(line))
            except: pass

    if not records:
        return {
            "total_predictions": 0,
            "average_predicted_risk": 0.0,
            "low_risk_count": 0,
            "medium_risk_count": 0,
            "high_risk_count": 0,
        }

    probs=[r["predicted_probability"] for r in records]
    levels=[r["risk_level"] for r in records]

    return {
        "total_predictions": len(probs),
        "average_predicted_risk": float(np.mean(probs)),
        "low_risk_count": levels.count("Low"),
        "medium_risk_count": levels.count("Medium"),
        "high_risk_count": levels.count("High"),
    }

# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to(p):
    st.session_state.page = p

# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------
def home_page():
    st.markdown("<br><br><br>", unsafe_allow_html=True)

    st.title("ChurnAlyse")
    st.subheader("A modern way to analyze and prevent policy lapses.")
    st.write("Predict churn, monitor risk, and save customers proactively.")
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Start Now"):
        go_to("predict")

# ---------------------------------------------------------
# PREDICT PAGE
# ---------------------------------------------------------
def predict_page():

    st.sidebar.title("Navigation")
    st.sidebar.radio("Go to:", ["Predict", "Performance"],
                     key="nav_pred",
                     on_change=lambda: go_to(st.session_state.nav_pred.lower()))


    st.title("Predict Policy Lapse Risk")

    age = st.number_input("Age", 18, 80, 30)
    gender = 0 if st.selectbox("Gender", ["Female","Male"])=="Female" else 1
    pt1 = st.number_input("Policy Type 1", 1,10,3)
    pt2 = st.number_input("Policy Type 2", 1,40,4)
    pamt = st.number_input("Policy Amount", 1000, 1000000, 50000)
    prem = st.number_input("Premium Amount", 1,100000,200)
    ten = st.number_input("Policy Tenure (years)", 0.0,20.0,2.0)
    tend = st.number_input("Policy Tenure Decimal", 0.0,10.0,1.5)
    ch1 = st.number_input("Channel 1",0,10,2)
    ch2 = st.number_input("Channel 2",0,10,2)
    ch3 = st.number_input("Channel 3",0,10,1)
    sr = st.selectbox("Substandard Risk",[0,1])
    adv = st.number_input("Advance Premium Count",0,10,1)
    ben = st.number_input("Initial Benefit",0,2000000,10000)

    if st.button("Predict"):

        data = {
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

        proba, risk = predict_and_log(data)

        st.subheader(f"Risk Level: **{risk}**")
        st.write(f"Lapse Probability: **{round(proba*100,2)}%**")

        st.subheader("Why this customer got this risk result")

        if risk=="High":
            for x in explain_high(data): st.write("- " + x)
        elif risk=="Medium":
            for x in explain_medium(data): st.write("- " + x)
        else:
            for x in explain_low(data): st.write("- " + x)

# ---------------------------------------------------------
# PERFORMANCE PAGE
# ---------------------------------------------------------
def performance_page():

    st.sidebar.title("Navigation")
    st.sidebar.radio("Go to:", ["Predict", "Performance"],
                     key="nav_perf",
                     on_change=lambda: go_to(st.session_state.nav_perf.lower()))

    st.title("Model Performance Dashboard")

    stats = load_prediction_stats()

    st.subheader("Overall Prediction Summary")
    st.write(f"**Total Predictions:** {stats['total_predictions']}")
    st.write(f"**Average Predicted Risk:** {stats['average_predicted_risk']:.3f}")

    # Pie Chart
    labels=["Low","Medium","High"]
    values=[
        stats["low_risk_count"],
        stats["medium_risk_count"],
        stats["high_risk_count"],
    ]
    colors=["#7bd88f","#ffb74d","#ef5350"]

    pie_fig=go.Figure(
        data=[go.Pie(labels=labels,values=values,hole=0.4,marker=dict(colors=colors))]
    )
    pie_fig.update_layout(
        title_text="Risk Level Distribution",
        legend=dict(orientation="h",y=-0.1),
    )
    st.plotly_chart(pie_fig, use_container_width=True)

    # Bar Chart
    st.subheader("Model Evaluation Metrics (Training)")

    metric_names=[]
    metric_values=[]

    for k in ["accuracy","precision","recall","f1_score","auc"]:
        v=TRAIN_METRICS.get(k)
        if v is not None:
            metric_names.append(k.capitalize())
            metric_values.append(v)

    if metric_names:
        bar_fig=go.Figure(
            data=[go.Bar(x=metric_names,y=metric_values)]
        )

        bar_fig.update_layout(
            yaxis=dict(range=[0,1]),
            title="Model Metric Comparison",
            xaxis_title="Metric",
            yaxis_title="Score",
            font=dict(size=16),
        )
        st.plotly_chart(bar_fig,use_container_width=True)
    else:
        st.info("Training metrics file missing.")

# ---------------------------------------------------------
# ROUTER
# ---------------------------------------------------------
if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "predict":
    predict_page()
elif st.session_state.page == "performance":
    performance_page()
