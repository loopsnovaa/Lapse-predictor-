"""
Interactive monitoring dashboard for churn prediction system.
Built with Dash and Plotly for real-time monitoring and visualization.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Churn Prediction Monitoring Dashboard"

# Configuration
API_BASE_URL = "http://localhost:8000"
REFRESH_INTERVAL = 30000  # 30 seconds


class MonitoringDashboard:
    """Monitoring dashboard for churn prediction system."""
    
    def __init__(self):
        self.model_info = {}
        self.performance_history = []
        self.prediction_history = []
        
    def load_model_info(self):
        """Load model information from API."""
        try:
            response = requests.get(f"{API_BASE_URL}/model/info")
            if response.status_code == 200:
                self.model_info = response.json()
                logger.info("Model info loaded successfully")
            else:
                logger.warning("Could not load model info")
        except Exception as e:
            logger.error(f"Error loading model info: {e}")
    
    def load_performance_metrics(self):
        """Load performance metrics from API."""
        try:
            response = requests.get(f"{API_BASE_URL}/metrics")
            if response.status_code == 200:
                metrics = response.json()
                metrics['timestamp'] = datetime.now().isoformat()
                self.performance_history.append(metrics)
                
                # Keep only last 100 records
                if len(self.performance_history) > 100:
                    self.performance_history = self.performance_history[-100:]
                
                logger.info("Performance metrics loaded successfully")
            else:
                logger.warning("Could not load performance metrics")
        except Exception as e:
            logger.error(f"Error loading performance metrics: {e}")
    
    def get_sample_predictions(self, n_samples: int = 10):
        """Get sample predictions for demonstration."""
        try:
            response = requests.get(f"{API_BASE_URL}/predict/sample")
            if response.status_code == 200:
                prediction = response.json()
                prediction['timestamp'] = datetime.now().isoformat()
                self.prediction_history.append(prediction)
                
                # Keep only last 1000 records
                if len(self.prediction_history) > 1000:
                    self.prediction_history = self.prediction_history[-1000:]
                
                logger.info("Sample prediction loaded successfully")
            else:
                logger.warning("Could not load sample prediction")
        except Exception as e:
            logger.error(f"Error loading sample prediction: {e}")


# Initialize dashboard
dashboard = MonitoringDashboard()
dashboard.load_model_info()


def create_header():
    """Create dashboard header."""
    return dbc.NavbarSimple(
        brand="Churn Prediction Monitoring Dashboard",
        brand_href="#",
        color="primary",
        dark=True,
        className="mb-4"
    )


def create_model_info_card():
    """Create model information card."""
    return dbc.Card([
        dbc.CardHeader("Model Information"),
        dbc.CardBody([
            html.Div(id="model-info-content")
        ])
    ])


def create_performance_metrics_card():
    """Create performance metrics card."""
    return dbc.Card([
        dbc.CardHeader("Performance Metrics"),
        dbc.CardBody([
            html.Div(id="performance-metrics-content")
        ])
    ])


def create_prediction_distribution_chart():
    """Create prediction distribution chart."""
    return dbc.Card([
        dbc.CardHeader("Prediction Distribution"),
        dbc.CardBody([
            dcc.Graph(id="prediction-distribution-chart")
        ])
    ])


def create_risk_level_chart():
    """Create risk level distribution chart."""
    return dbc.Card([
        dbc.CardHeader("Risk Level Distribution"),
        dbc.CardBody([
            dcc.Graph(id="risk-level-chart")
        ])
    ])


def create_performance_trend_chart():
    """Create performance trend chart."""
    return dbc.Card([
        dbc.CardHeader("Performance Trends"),
        dbc.CardBody([
            dcc.Graph(id="performance-trend-chart")
        ])
    ])


def create_feature_importance_chart():
    """Create feature importance chart."""
    return dbc.Card([
        dbc.CardHeader("Feature Importance"),
        dbc.CardBody([
            dcc.Graph(id="feature-importance-chart")
        ])
    ])


def create_recent_predictions_table():
    """Create recent predictions table."""
    return dbc.Card([
        dbc.CardHeader("Recent Predictions"),
        dbc.CardBody([
            html.Div(id="recent-predictions-table")
        ])
    ])


# App layout
app.layout = dbc.Container([
    # Auto-refresh interval
    dcc.Interval(
        id='interval-component',
        interval=REFRESH_INTERVAL,
        n_intervals=0
    ),
    
    # Header
    create_header(),
    
    # Main content
    dbc.Row([
        # Left column
        dbc.Col([
            create_model_info_card(),
            html.Br(),
            create_performance_metrics_card(),
            html.Br(),
            create_recent_predictions_table()
        ], width=4),
        
        # Right column
        dbc.Col([
            dbc.Row([
                dbc.Col([create_prediction_distribution_chart()], width=6),
                dbc.Col([create_risk_level_chart()], width=6)
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([create_performance_trend_chart()], width=6),
                dbc.Col([create_feature_importance_chart()], width=6)
            ])
        ], width=8)
    ])
], fluid=True)


@app.callback(
    [Output('model-info-content', 'children'),
     Output('performance-metrics-content', 'children'),
     Output('prediction-distribution-chart', 'figure'),
     Output('risk-level-chart', 'figure'),
     Output('performance-trend-chart', 'figure'),
     Output('feature-importance-chart', 'figure'),
     Output('recent-predictions-table', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n_intervals):
    """Update dashboard components."""
    
    # Load fresh data
    dashboard.load_model_info()
    dashboard.load_performance_metrics()
    dashboard.get_sample_predictions()
    
    # Model info content
    model_info_content = []
    if dashboard.model_info:
        model_info_content = [
            html.P(f"Model Type: {dashboard.model_info.get('model_type', 'N/A')}"),
            html.P(f"Version: {dashboard.model_info.get('version', 'N/A')}"),
            html.P(f"Training Date: {dashboard.model_info.get('training_date', 'N/A')}"),
            html.P(f"Feature Count: {dashboard.model_info.get('feature_count', 'N/A')}"),
            html.P(f"Status: {'Loaded' if dashboard.model_info.get('is_loaded', False) else 'Not Loaded'}")
        ]
    
    # Performance metrics content
    performance_content = []
    if dashboard.performance_history:
        latest_metrics = dashboard.performance_history[-1].get('model_metrics', {})
        performance_content = [
            html.P(f"AUC: {latest_metrics.get('auc', 0):.4f}"),
            html.P(f"Accuracy: {latest_metrics.get('accuracy', 0):.4f}"),
            html.P(f"Precision: {latest_metrics.get('precision', 0):.4f}"),
            html.P(f"Recall: {latest_metrics.get('recall', 0):.4f}"),
            html.P(f"F1-Score: {latest_metrics.get('f1_score', 0):.4f}")
        ]
    
    # Prediction distribution chart
    prediction_dist_fig = go.Figure()
    if dashboard.prediction_history:
        probabilities = [p['churn_probability'] for p in dashboard.prediction_history]
        prediction_dist_fig.add_trace(go.Histogram(
            x=probabilities,
            nbinsx=20,
            name='Churn Probability Distribution'
        ))
        prediction_dist_fig.update_layout(
            title="Churn Probability Distribution",
            xaxis_title="Probability",
            yaxis_title="Count"
        )
    
    # Risk level chart
    risk_level_fig = go.Figure()
    if dashboard.prediction_history:
        risk_levels = [p['risk_level'] for p in dashboard.prediction_history]
        risk_counts = pd.Series(risk_levels).value_counts()
        
        risk_level_fig.add_trace(go.Bar(
            x=risk_counts.index,
            y=risk_counts.values,
            name='Risk Level Distribution'
        ))
        risk_level_fig.update_layout(
            title="Risk Level Distribution",
            xaxis_title="Risk Level",
            yaxis_title="Count"
        )
    
    # Performance trend chart
    performance_trend_fig = go.Figure()
    if len(dashboard.performance_history) > 1:
        timestamps = [datetime.fromisoformat(p['timestamp']) for p in dashboard.performance_history]
        auc_scores = [p.get('model_metrics', {}).get('auc', 0) for p in dashboard.performance_history]
        
        performance_trend_fig.add_trace(go.Scatter(
            x=timestamps,
            y=auc_scores,
            mode='lines+markers',
            name='AUC Score'
        ))
        performance_trend_fig.update_layout(
            title="AUC Score Trend",
            xaxis_title="Time",
            yaxis_title="AUC Score"
        )
    
    # Feature importance chart
    feature_importance_fig = go.Figure()
    try:
        # Try to load feature importance from model files
        models_dir = "models"
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.startswith("ensemble_model_")]
            if model_files:
                latest_model = sorted(model_files)[-1]
                model_path = os.path.join(models_dir, latest_model)
                
                model_data = joblib.load(model_path)
                if 'feature_importance' in model_data and model_data['feature_importance'] is not None:
                    feature_importance = model_data['feature_importance'].head(10)
                    
                    feature_importance_fig.add_trace(go.Bar(
                        x=feature_importance['importance'],
                        y=feature_importance['feature'],
                        orientation='h',
                        name='Feature Importance'
                    ))
                    feature_importance_fig.update_layout(
                        title="Top 10 Feature Importance",
                        xaxis_title="Importance Score",
                        yaxis_title="Feature"
                    )
    except Exception as e:
        logger.warning(f"Could not load feature importance: {e}")
    
    # Recent predictions table
    recent_predictions_content = []
    if dashboard.prediction_history:
        recent_predictions = dashboard.prediction_history[-10:]  # Last 10 predictions
        
        table_data = []
        for pred in recent_predictions:
            table_data.append([
                pred.get('customer_id', 'N/A'),
                f"{pred['churn_probability']:.4f}",
                pred['risk_level'],
                f"{pred['confidence']:.4f}",
                pred['timestamp'][:19]  # Truncate timestamp
            ])
        
        recent_predictions_content = dbc.Table.from_dataframe(
            pd.DataFrame(table_data, columns=[
                'Customer ID', 'Probability', 'Risk Level', 'Confidence', 'Timestamp'
            ]),
            striped=True,
            bordered=True,
            hover=True,
            responsive=True,
            size='sm'
        )
    
    return (
        model_info_content,
        performance_content,
        prediction_dist_fig,
        risk_level_fig,
        performance_trend_fig,
        feature_importance_fig,
        recent_predictions_content
    )


def main():
    """Main function to run the dashboard."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Churn Prediction Monitoring Dashboard')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                       help='Host to bind the dashboard')
    parser.add_argument('--port', type=int, default=8050,
                       help='Port to bind the dashboard')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    app.run_server(
        host=args.host,
        port=args.port,
        debug=args.debug
    )


if __name__ == "__main__":
    main()



