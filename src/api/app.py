"""
FastAPI-based prediction service for churn prediction.
Provides real-time scoring and batch prediction capabilities.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import joblib
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from src.data.preprocessing import DataPreprocessor
from src.models.ensemble import ChurnEnsembleModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="ML-driven churn and lapse risk prediction system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessor
model = None
preprocessor = None
model_info = {}


class PolicyData(BaseModel):
    """Pydantic model for policy data input."""
    policy_id: Optional[str] = None
    age: Optional[int] = Field(None, ge=18, le=100)
    gender: Optional[str] = Field(None, regex="^(Male|Female)$")
    marital_status: Optional[str] = Field(None, regex="^(Single|Married|Divorced|Widowed)$")
    education: Optional[str] = Field(None, regex="^(High School|Bachelor|Master|PhD)$")
    income: Optional[float] = Field(None, ge=0)
    policy_type: Optional[str] = Field(None, regex="^(Life|Health|Auto|Home)$")
    policy_amount: Optional[float] = Field(None, ge=0)
    premium_amount: Optional[float] = Field(None, ge=0)
    policy_tenure_months: Optional[int] = Field(None, ge=0)
    payment_frequency: Optional[str] = Field(None, regex="^(Monthly|Quarterly|Semi-Annual|Annual)$")
    payment_method: Optional[str] = Field(None, regex="^(Bank Transfer|Credit Card|Check|Cash)$")
    claims_history: Optional[int] = Field(None, ge=0)
    credit_score: Optional[int] = Field(None, ge=300, le=850)
    employment_status: Optional[str] = Field(None, regex="^(Employed|Self-Employed|Unemployed|Retired)$")
    smoking_status: Optional[str] = Field(None, regex="^(Non-Smoker|Smoker|Former Smoker)$")
    health_conditions: Optional[str] = Field(None, regex="^(None|Diabetes|Hypertension|Heart Disease)$")


class PredictionResponse(BaseModel):
    """Pydantic model for prediction response."""
    policy_id: Optional[str]
    lapse_probability: float = Field(..., ge=0, le=1)
    lapse_prediction: int = Field(..., ge=0, le=1)
    risk_level: str
    confidence: float = Field(..., ge=0, le=1)
    timestamp: str
    model_version: str


class BatchPredictionRequest(BaseModel):
    """Pydantic model for batch prediction request."""
    policies: List[PolicyData]


class BatchPredictionResponse(BaseModel):
    """Pydantic model for batch prediction response."""
    predictions: List[PredictionResponse]
    total_customers: int
    processing_time: float
    timestamp: str


class ModelInfo(BaseModel):
    """Pydantic model for model information."""
    model_type: str
    version: str
    training_date: str
    performance_metrics: Dict[str, float]
    feature_count: int
    is_loaded: bool


def load_model(model_path: str, preprocessor_path: str):
    """
    Load the trained model and preprocessor.
    
    Args:
        model_path: Path to the trained model
        preprocessor_path: Path to the preprocessor
    """
    global model, preprocessor, model_info
    
    try:
        # Load model
        model = ChurnEnsembleModel()
        model.load_model(model_path)
        
        # Load preprocessor
        preprocessor = DataPreprocessor()
        preprocessor.load_preprocessor(preprocessor_path)
        
        # Update model info
        model_info.update({
            'model_type': 'Ensemble (Logistic Regression + XGBoost)',
            'version': '1.0.0',
            'training_date': datetime.now().isoformat(),
            'is_loaded': True,
            'feature_count': len(preprocessor.feature_names) if preprocessor.feature_names else 0
        })
        
        logger.info("Model and preprocessor loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")


def preprocess_customer_data(customer_data: CustomerData) -> np.ndarray:
    """
    Preprocess customer data for prediction.
    
    Args:
        customer_data: Customer data input
        
    Returns:
        Preprocessed feature array
    """
    # Convert to DataFrame
    df = pd.DataFrame([customer_data.dict()])
    
    # Remove customer_id if present
    if 'customer_id' in df.columns:
        df = df.drop(columns=['customer_id'])
    
    # Handle missing values with defaults
    defaults = {
        'age': 40,
        'tenure': 24,
        'monthly_charges': 60.0,
        'total_charges': 1500.0,
        'gender': 'Male',
        'partner': 'No',
        'dependents': 'No',
        'phone_service': 'Yes',
        'internet_service': 'DSL',
        'contract': 'Month-to-month',
        'payment_method': 'Electronic check'
    }
    
    for col, default_val in defaults.items():
        if col in df.columns and df[col].isnull().any():
            df[col].fillna(default_val, inplace=True)
    
    # Apply feature engineering
    df_engineered = preprocessor.engineer_features(df)
    
    # Encode categorical features
    df_encoded = preprocessor.encode_categorical_features(df_engineered)
    
    # Scale features
    X_scaled = preprocessor.scale_features(df_encoded.values, fit=False)
    
    return X_scaled


def calculate_risk_level(probability: float) -> str:
    """
    Calculate risk level based on churn probability.
    
    Args:
        probability: Churn probability
        
    Returns:
        Risk level string
    """
    if probability < 0.3:
        return "Low"
    elif probability < 0.6:
        return "Medium"
    else:
        return "High"


def calculate_confidence(probability: float) -> float:
    """
    Calculate confidence score based on probability.
    
    Args:
        probability: Churn probability
        
    Returns:
        Confidence score
    """
    # Confidence is higher when probability is closer to 0 or 1
    return max(probability, 1 - probability) * 2 - 1


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    # Try to load the latest model
    models_dir = "models"
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.startswith("ensemble_model_")]
        if model_files:
            latest_model = sorted(model_files)[-1]
            model_path = os.path.join(models_dir, latest_model)
            
            preprocessor_files = [f for f in os.listdir(models_dir) if f.startswith("preprocessor_")]
            if preprocessor_files:
                latest_preprocessor = sorted(preprocessor_files)[-1]
                preprocessor_path = os.path.join(models_dir, latest_preprocessor)
                
                load_model(model_path, preprocessor_path)
            else:
                logger.warning("No preprocessor found")
        else:
            logger.warning("No trained model found")
    else:
        logger.warning("Models directory not found")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Churn Prediction API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=404, detail="Model not loaded")
    
    return ModelInfo(**model_info)


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer_data: CustomerData):
    """
    Predict churn risk for a single customer.
    
    Args:
        customer_data: Customer data input
        
    Returns:
        Prediction response with churn probability and risk level
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess data
        X = preprocess_customer_data(customer_data)
        
        # Make prediction
        churn_probability = model.predict_proba(X)[0, 1]
        churn_prediction = int(churn_probability > 0.5)
        
        # Calculate additional metrics
        risk_level = calculate_risk_level(churn_probability)
        confidence = calculate_confidence(churn_probability)
        
        return PredictionResponse(
            customer_id=customer_data.customer_id,
            churn_probability=round(churn_probability, 4),
            churn_prediction=churn_prediction,
            risk_level=risk_level,
            confidence=round(confidence, 4),
            timestamp=datetime.now().isoformat(),
            model_version=model_info.get('version', '1.0.0')
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_churn_batch(request: BatchPredictionRequest):
    """
    Predict churn risk for multiple customers.
    
    Args:
        request: Batch prediction request with customer data
        
    Returns:
        Batch prediction response
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = datetime.now()
    
    try:
        predictions = []
        
        for customer_data in request.customers:
            # Preprocess data
            X = preprocess_customer_data(customer_data)
            
            # Make prediction
            churn_probability = model.predict_proba(X)[0, 1]
            churn_prediction = int(churn_probability > 0.5)
            
            # Calculate additional metrics
            risk_level = calculate_risk_level(churn_probability)
            confidence = calculate_confidence(churn_probability)
            
            predictions.append(PredictionResponse(
                customer_id=customer_data.customer_id,
                churn_probability=round(churn_probability, 4),
                churn_prediction=churn_prediction,
                risk_level=risk_level,
                confidence=round(confidence, 4),
                timestamp=datetime.now().isoformat(),
                model_version=model_info.get('version', '1.0.0')
            ))
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_customers=len(predictions),
            processing_time=round(processing_time, 4),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/predict/sample")
async def get_sample_prediction():
    """Get a sample prediction for testing."""
    sample_customer = CustomerData(
        customer_id="sample_001",
        age=45,
        tenure=36,
        monthly_charges=75.0,
        total_charges=2700.0,
        gender="Male",
        partner="Yes",
        dependents="No",
        phone_service="Yes",
        internet_service="Fiber optic",
        contract="Two year",
        payment_method="Credit card"
    )
    
    return await predict_churn(sample_customer)


@app.post("/model/reload")
async def reload_model(model_path: str, preprocessor_path: str):
    """
    Reload model and preprocessor.
    
    Args:
        model_path: Path to the model file
        preprocessor_path: Path to the preprocessor file
    """
    try:
        load_model(model_path, preprocessor_path)
        return {"message": "Model reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")


@app.get("/metrics")
async def get_metrics():
    """Get model performance metrics."""
    if model is None:
        raise HTTPException(status_code=404, detail="Model not loaded")
    
    # Load evaluation results if available
    models_dir = "models"
    eval_files = [f for f in os.listdir(models_dir) if f.startswith("evaluation_results_")]
    
    if eval_files:
        latest_eval = sorted(eval_files)[-1]
        eval_path = os.path.join(models_dir, latest_eval)
        
        try:
            with open(eval_path, 'r') as f:
                evaluation_results = json.load(f)
            
            return {
                "model_metrics": {
                    "auc": evaluation_results.get('ensemble_auc', 0),
                    "accuracy": evaluation_results.get('ensemble_accuracy', 0),
                    "precision": evaluation_results.get('ensemble_precision', 0),
                    "recall": evaluation_results.get('ensemble_recall', 0),
                    "f1_score": evaluation_results.get('ensemble_f1', 0)
                },
                "model_info": model_info,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.warning(f"Could not load evaluation results: {e}")
    
    return {
        "model_info": model_info,
        "timestamp": datetime.now().isoformat()
    }


def main():
    """Main function to run the API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Churn Prediction API')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind the server')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port to bind the server')
    parser.add_argument('--reload', action='store_true',
                       help='Enable auto-reload for development')
    
    args = parser.parse_args()
    
    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
