import streamlit as st
import requests
import plotly.graph_objects as go

API_URL = "http://127.0.0.1:5000/model_stats"

def app():
    st.title("Model Performance Dashboard")

    try:
        stats = requests.get(API_URL).json()

        if "error" in stats:
            st.error("No prediction logs available yet.")
            return

        st.subheader("Overall Prediction Summary")
        st.write(f"Total Predictions: **{stats['total_predictions']}**")
        st.write(f"Average Predicted Risk: **{stats['average_predicted_risk']:.3f}**")
        st.write(f"Estimated Saved Customers: **{stats['saved_customers_estimate']}**")

        labels = ["Low", "Medium", "High"]
        values = [
            stats["low_risk_count"],
            stats["medium_risk_count"],
            stats["high_risk_count"],
        ]

        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
        fig.update_layout(title_text="Risk Level Distribution")

        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Failed to fetch stats: {e}")
