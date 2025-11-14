# Lapse Predictor

Lapse Predictor is a machine learning–based web application designed to estimate the probability of life insurance policy lapse. The system processes real insurance datasets, identifies important factors influencing lapse behaviour, and provides real-time predictions through an integrated Flask API and Streamlit interface.

---

## Project Overview

This project implements multiple machine learning models, including Logistic Regression, Random Forest, and XGBoost, trained on a real Kaggle insurance dataset. The solution includes data preprocessing, class imbalance handling using SMOTE-ENN, prediction serving, and a lightweight web dashboard.

---

## Features

- Machine learning models for lapse risk prediction  
- Data preprocessing and feature engineering tailored for insurance datasets  
- Class imbalance handling using SMOTE-ENN  
- Flask API for real-time predictions  
- Streamlit dashboard for user interaction  
- Explanation of key predictive features  
- Support for retraining with new data  

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train or retrain the model
```bash
python simple_kaggle_integration.py
```

### 3. Start the API
```bash
python api.py
```

### 4. Launch the dashboard
```bash
python dashboard_app.py
```

---

## Project Structure

```
├── api.py
├── dashboard_app.py
├── simple_kaggle_integration.py
├── models/
├── data/
├── src/
└── requirements.txt
```

---

## Model Performance Summary

Performance varies depending on dataset quality and distributions, but typical results from XGBoost on the processed Kaggle dataset include:

- AUC: 0.68–0.75  
- Accuracy: around 0.64  
- Recall: approximately 0.70  
- Top features: policy tenure, policy type, channel attributes, premium-to-benefit ratio  

---

## API Endpoints

### POST `/predict_lapse`
Submits policy attributes and returns the predicted lapse probability and risk category.

### GET `/health`
Health check endpoint for API availability.

---

## License

MIT License
