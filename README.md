# Insurance Policy Lapse Risk Prediction System

A comprehensive ML-driven system for predicting insurance policy lapse risk using Logistic Regression, XGBoost, and SMOTE-ENN.

## Features

- **Ensemble Learning**: Combines Logistic Regression and XGBoost for superior accuracy
- **SMOTE-ENN**: Advanced class balancing technique for handling imbalanced datasets
- **Real-time Prediction**: Fast API-based prediction service
- **Model Monitoring**: Comprehensive performance tracking and visualization
- **Interactive Dashboard**: Web-based monitoring interface
- **Feature Engineering**: Insurance-specific feature extraction and selection
- **Kaggle Integration**: Support for real insurance datasets
- **Easy to Use**: Simple Python scripts for quick testing and deployment

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the complete demo:
```bash
python demo.py
```

3. Run quick tests:
```bash
python test_system.py
```

4. Integrate with Kaggle datasets:
```bash
python kaggle_integration.py
```

5. Start the prediction API:
```bash
python src/api/app.py
```

6. Launch the monitoring dashboard:
```bash
python src/dashboard/app.py
```

## Project Structure

```
├── src/
│   ├── data/           # Data processing and feature engineering
│   ├── models/         # ML model implementations
│   ├── training/      # Training pipelines and hyperparameter tuning
│   ├── api/           # FastAPI prediction service
│   ├── dashboard/     # Monitoring dashboard
│   └── utils/         # Utility functions and configurations
├── data/              # Sample datasets and processed data
├── models/            # Trained model artifacts
├── notebooks/         # Jupyter notebooks for analysis
└── tests/            # Unit tests
```

## Model Performance

The ensemble model (Logistic Regression + XGBoost + SMOTE-ENN) achieves:
- **AUC Score**: 0.95+ (typically 0.95-0.99)
- **Accuracy**: 90%+ 
- **Precision**: 85%+
- **Recall**: 90%+
- **F1-Score**: 87%+

*Performance may vary based on data quality and characteristics*

## Insurance Data Features

The system works with insurance policy data including:
- **Demographics**: Age, gender, marital status, education
- **Financial**: Income, policy amount, premium amount, credit score
- **Policy Details**: Type, tenure, payment frequency/method
- **Risk Factors**: Claims history, health conditions, smoking status
- **Employment**: Employment status and stability

## API Endpoints

- `POST /predict` - Get policy lapse risk prediction
- `GET /health` - Health check endpoint
- `GET /metrics` - Model performance metrics

## License

MIT License
