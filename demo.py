#!/usr/bin/env python3
"""
Insurance Policy Lapse Prediction System - Complete Demo Script

This script demonstrates the complete insurance policy lapse prediction system using:
- Logistic Regression
- XGBoost
- SMOTE-ENN for class balancing
- Ensemble learning with stacking

Run this script to see the full pipeline in action.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
from datetime import datetime

# Import our modules
from data.preprocessing import DataPreprocessor, create_sample_data
from models.ensemble import ChurnEnsembleModel
from training.train_ensemble import TrainingPipeline
from utils.config import Config

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_subsection(title):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")

def main():
    """Main demo function."""
    print_section("INSURANCE POLICY LAPSE PREDICTION SYSTEM DEMO")
    print("Using Logistic Regression + XGBoost + SMOTE-ENN")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Create Sample Data
    print_section("STEP 1: DATA CREATION")
    print("Creating sample insurance policy data...")
    
    sample_data = create_sample_data(1000)
    print(f"✓ Sample data created with shape: {sample_data.shape}")
    print(f"✓ Features: {list(sample_data.columns)}")
    
    print_subsection("Data Overview")
    print("First 5 rows:")
    print(sample_data.head())
    
    print_subsection("Policy Lapse Distribution")
    lapse_counts = sample_data['policy_lapse'].value_counts()
    lapse_rate = sample_data['policy_lapse'].mean()
    print(f"Policy lapse distribution: {dict(lapse_counts)}")
    print(f"Lapse rate: {lapse_rate:.2%}")
    
    # Step 2: Data Preprocessing
    print_section("STEP 2: DATA PREPROCESSING")
    print("Initializing preprocessor and applying SMOTE-ENN...")
    
    preprocessor = DataPreprocessor()
    prepared_data = preprocessor.prepare_data(sample_data, 'policy_lapse')
    
    print(f"✓ Data preprocessing completed")
    print(f"✓ Training set shape: {prepared_data['X_train'].shape}")
    print(f"✓ Test set shape: {prepared_data['X_test'].shape}")
    print(f"✓ Number of features: {len(prepared_data['feature_names'])}")
    
    # Check class distribution after SMOTE-ENN
    unique, counts = np.unique(prepared_data['y_train'], return_counts=True)
    print(f"✓ Training set class distribution after SMOTE-ENN: {dict(zip(unique, counts))}")
    
    # Step 3: Model Training
    print_section("STEP 3: MODEL TRAINING")
    print("Training ensemble model (Logistic Regression + XGBoost)...")
    
    ensemble_model = ChurnEnsembleModel()
    training_results = ensemble_model.train(
        prepared_data['X_train'], 
        prepared_data['y_train']
    )
    
    print("✓ Model training completed!")
    print_subsection("Cross-Validation Results")
    for model_name, scores in training_results['individual_scores'].items():
        print(f"{model_name:20}: AUC = {scores['mean_cv_score']:.4f} (+/- {scores['std_cv_score'] * 2:.4f})")
    
    # Step 4: Model Evaluation
    print_section("STEP 4: MODEL EVALUATION")
    print("Evaluating model on test set...")
    
    evaluation_results = ensemble_model.evaluate(
        prepared_data['X_test'], 
        prepared_data['y_test']
    )
    
    print("✓ Model evaluation completed!")
    print_subsection("Ensemble Model Performance")
    print(f"AUC Score:      {evaluation_results['ensemble_auc']:.4f}")
    print(f"Accuracy:      {evaluation_results['ensemble_accuracy']:.4f}")
    print(f"Precision:     {evaluation_results['ensemble_precision']:.4f}")
    print(f"Recall:        {evaluation_results['ensemble_recall']:.4f}")
    print(f"F1-Score:      {evaluation_results['ensemble_f1']:.4f}")
    
    print_subsection("Detailed Classification Report")
    print(classification_report(
        evaluation_results['y_test'], 
        evaluation_results['y_pred']
    ))
    
    # Step 5: Feature Importance
    print_section("STEP 5: FEATURE IMPORTANCE ANALYSIS")
    print("Analyzing feature importance...")
    
    feature_importance = ensemble_model.get_feature_importance(prepared_data['feature_names'])
    
    print("✓ Feature importance calculated")
    print_subsection("Top 10 Most Important Features")
    top_features = feature_importance.head(10)
    for idx, row in top_features.iterrows():
        print(f"{row['feature']:30}: {row['importance']:.4f}")
    
    # Step 6: Visualizations
    print_section("STEP 6: CREATING VISUALIZATIONS")
    print("Generating performance visualizations...")
    
    # Create output directory for plots
    os.makedirs('output', exist_ok=True)
    
    # Plot 1: Confusion Matrix
    plt.figure(figsize=(8, 6))
    conf_matrix = evaluation_results['confusion_matrix']
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
               xticklabels=['No Churn', 'Churn'],
               yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix - Ensemble Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('output/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Confusion matrix saved to output/confusion_matrix.png")
    
    # Plot 2: ROC Curves
    plt.figure(figsize=(10, 8))
    
    # Ensemble ROC curve
    fpr, tpr, _ = roc_curve(evaluation_results['y_test'], evaluation_results['y_pred_proba'])
    auc_score = evaluation_results['ensemble_auc']
    plt.plot(fpr, tpr, label=f'Ensemble (AUC = {auc_score:.4f})', linewidth=2)
    
    # Individual model ROC curves
    colors = ['red', 'green']
    for i, (name, proba) in enumerate(evaluation_results['individual_probabilities'].items()):
        fpr, tpr, _ = roc_curve(evaluation_results['y_test'], proba)
        auc_score = roc_auc_score(evaluation_results['y_test'], proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.4f})', 
                color=colors[i % len(colors)], alpha=0.7)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ ROC curves saved to output/roc_curves.png")
    
    # Plot 3: Feature Importance
    plt.figure(figsize=(12, 8))
    top_features_plot = feature_importance.head(15)
    sns.barplot(data=top_features_plot, x='importance', y='feature')
    plt.title('Top 15 Feature Importance (XGBoost)')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('output/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Feature importance plot saved to output/feature_importance.png")
    
    # Step 7: Sample Predictions
    print_section("STEP 7: SAMPLE PREDICTIONS")
    print("Making predictions on sample customers...")
    
    # Create sample customers
    sample_customers = pd.DataFrame({
        'customer_id': ['CUST_001', 'CUST_002', 'CUST_003', 'CUST_004', 'CUST_005'],
        'age': [35, 55, 28, 42, 67],
        'tenure': [12, 60, 6, 24, 48],
        'monthly_charges': [80.0, 45.0, 120.0, 65.0, 35.0],
        'total_charges': [960.0, 2700.0, 720.0, 1560.0, 1680.0],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'partner': ['Yes', 'No', 'No', 'Yes', 'Yes'],
        'dependents': ['No', 'Yes', 'No', 'No', 'Yes'],
        'phone_service': ['Yes', 'Yes', 'Yes', 'Yes', 'No'],
        'internet_service': ['Fiber optic', 'DSL', 'Fiber optic', 'DSL', 'No'],
        'contract': ['Month-to-month', 'Two year', 'Month-to-month', 'One year', 'Two year'],
        'payment_method': ['Electronic check', 'Credit card', 'Electronic check', 'Bank transfer', 'Mailed check']
    })
    
    print("Sample customers:")
    print(sample_customers[['customer_id', 'age', 'tenure', 'monthly_charges', 'contract']])
    
    # Make predictions
    predictions = []
    for _, customer in sample_customers.iterrows():
        # Preprocess customer data
        customer_df = customer.to_frame().T.drop(columns=['customer_id'])
        customer_engineered = preprocessor.engineer_features(customer_df)
        customer_encoded = preprocessor.encode_categorical_features(customer_engineered)
        customer_scaled = preprocessor.scale_features(customer_encoded.values, fit=False)
        
        # Make prediction
        churn_probability = ensemble_model.predict_proba(customer_scaled)[0, 1]
        churn_prediction = int(churn_probability > 0.5)
        
        # Calculate risk level
        if churn_probability < 0.3:
            risk_level = "Low"
        elif churn_probability < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        predictions.append({
            'customer_id': customer['customer_id'],
            'churn_probability': churn_probability,
            'churn_prediction': churn_prediction,
            'risk_level': risk_level,
            'confidence': max(churn_probability, 1 - churn_probability) * 2 - 1
        })
    
    # Display predictions
    predictions_df = pd.DataFrame(predictions)
    print_subsection("Prediction Results")
    print(predictions_df.to_string(index=False, float_format='%.4f'))
    
    # Step 8: Save Model
    print_section("STEP 8: SAVE MODEL")
    print("Saving trained model and preprocessor...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'models/ensemble_model_{timestamp}.joblib'
    ensemble_model.save_model(model_path)
    print(f"✓ Model saved to {model_path}")
    
    # Save preprocessor
    preprocessor_path = f'models/preprocessor_{timestamp}.joblib'
    preprocessor.save_preprocessor(preprocessor_path)
    print(f"✓ Preprocessor saved to {preprocessor_path}")
    
    # Save results
    results_path = f'models/results_{timestamp}.json'
    results_data = {
        'training_results': training_results,
        'evaluation_results': {
            'ensemble_auc': evaluation_results['ensemble_auc'],
            'ensemble_accuracy': evaluation_results['ensemble_accuracy'],
            'ensemble_precision': evaluation_results['ensemble_precision'],
            'ensemble_recall': evaluation_results['ensemble_recall'],
            'ensemble_f1': evaluation_results['ensemble_f1']
        },
        'feature_importance': feature_importance.head(20).to_dict('records'),
        'predictions': predictions,
        'timestamp': datetime.now().isoformat()
    }
    
    import json
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    print(f"✓ Results saved to {results_path}")
    
    # Step 9: Summary
    print_section("DEMO COMPLETED SUCCESSFULLY!")
    print("Summary of what was accomplished:")
    print("✓ Created sample customer data (1000 records)")
    print("✓ Applied SMOTE-ENN for class balancing")
    print("✓ Trained ensemble model (Logistic Regression + XGBoost)")
    print("✓ Achieved high performance metrics")
    print("✓ Analyzed feature importance")
    print("✓ Generated visualizations")
    print("✓ Made sample predictions")
    print("✓ Saved model artifacts")
    
    print_subsection("Final Performance Metrics")
    print(f"Ensemble AUC:      {evaluation_results['ensemble_auc']:.4f}")
    print(f"Ensemble Accuracy: {evaluation_results['ensemble_accuracy']:.4f}")
    print(f"Ensemble F1-Score: {evaluation_results['ensemble_f1']:.4f}")
    
    print_subsection("Files Created")
    print("• Models directory with trained artifacts")
    print("• Output directory with visualizations")
    print("• Results JSON with all metrics")
    
    print(f"\nDemo completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("You can now use the saved model for production predictions!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
