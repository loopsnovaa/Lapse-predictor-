#!/usr/bin/env python3
"""
Real Kaggle Insurance Dataset Integration Script

This script properly maps the real Kaggle insurance dataset to our 
insurance policy lapse prediction system.
"""

import pandas as pd
import numpy as np
import os
import sys
sys.path.append('src')

from data.preprocessing import DataPreprocessor
from models.ensemble import ChurnEnsembleModel

def load_and_clean_kaggle_data(file_path):
    """Load and clean the Kaggle insurance dataset."""
    print("="*60)
    print("LOADING AND CLEANING KAGGLE INSURANCE DATASET")
    print("="*60)
    
    # Load the dataset
    df = pd.read_csv(file_path, sep=';', encoding='utf-8')
    print(f"✓ Loaded dataset with shape: {df.shape}")
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Remove empty columns
    df = df.drop(columns=['Unnamed: 20', 'Unnamed: 21'])
    
    # Clean the data
    print("Cleaning data...")
    
    # Clean BENEFIT column (remove commas and convert to numeric)
    df['BENEFIT'] = df['BENEFIT'].str.replace(',', '').str.replace(' ', '').str.replace('-', '0')
    df['BENEFIT'] = pd.to_numeric(df['BENEFIT'], errors='coerce').fillna(0)
    
    # Clean Premium column (remove commas and convert to numeric)
    df['Premium'] = df['Premium'].str.replace(',', '').str.replace(' ', '').str.replace('-', '0')
    df['Premium'] = pd.to_numeric(df['Premium'], errors='coerce').fillna(0)
    
    # Clean INITIAL BENEFIT column
    df['INITIAL BENEFIT'] = df['INITIAL BENEFIT'].fillna(0)
    
    # Create policy_lapse target variable from POLICY STATUS
    df['policy_lapse'] = (df['POLICY STATUS'] == 'Lapse').astype(int)
    
    print(f"✓ Policy lapse distribution:")
    lapse_counts = df['policy_lapse'].value_counts()
    print(f"  - Active policies: {lapse_counts[0]:,} ({lapse_counts[0]/len(df)*100:.1f}%)")
    print(f"  - Lapsed policies: {lapse_counts[1]:,} ({lapse_counts[1]/len(df)*100:.1f}%)")
    
    # Create mapped columns for our system
    df_mapped = pd.DataFrame()
    
    # Map to our expected column names
    df_mapped['policy_id'] = range(1, len(df) + 1)
    df_mapped['age'] = df['ENTRY AGE']
    df_mapped['gender'] = df['SEX'].map({'M': 'Male', 'F': 'Female'})
    df_mapped['marital_status'] = 'Unknown'  # Not available in this dataset
    df_mapped['education'] = 'Unknown'  # Not available in this dataset
    df_mapped['income'] = df['BENEFIT'] * 0.1  # Estimate income as 10% of benefit
    df_mapped['policy_type'] = df['POLICY TYPE 1'].astype(str) + '_' + df['POLICY TYPE 2'].astype(str)
    df_mapped['policy_amount'] = df['BENEFIT']
    df_mapped['premium_amount'] = df['Premium']
    df_mapped['policy_tenure_months'] = df['Policy Year'] * 12  # Convert years to months
    df_mapped['payment_frequency'] = df['PAYMENT MODE']
    df_mapped['payment_method'] = 'Unknown'  # Not available in this dataset
    df_mapped['claims_history'] = 0  # Not available in this dataset
    df_mapped['credit_score'] = 700  # Default value
    df_mapped['employment_status'] = 'Unknown'  # Not available in this dataset
    df_mapped['smoking_status'] = 'Unknown'  # Not available in this dataset
    df_mapped['health_conditions'] = df['SUBSTANDARD RISK'].map({0: 'None', 1: 'High Risk'})
    df_mapped['policy_lapse'] = df['policy_lapse']
    
    # Add additional features from the original dataset
    df_mapped['channel1'] = df['CHANNEL1']
    df_mapped['channel2'] = df['CHANNEL2']
    df_mapped['channel3'] = df['CHANNEL3']
    df_mapped['policy_type_3'] = df['POLICY TYPE 3']
    df_mapped['non_lapse_guaranteed'] = df['NON LAPSE GUARANTEED']
    df_mapped['substandard_risk'] = df['SUBSTANDARD RISK']
    df_mapped['number_of_advance_premium'] = df['NUMBER OF ADVANCE PREMIUM']
    df_mapped['initial_benefit'] = df['INITIAL BENEFIT']
    df_mapped['full_benefit'] = df['Full Benefit?']
    df_mapped['policy_year_decimal'] = df['Policy Year (Decimal)']
    df_mapped['policy_year'] = df['Policy Year']
    df_mapped['issue_date'] = df['Issue Date']
    
    print(f"✓ Mapped dataset shape: {df_mapped.shape}")
    print(f"✓ Features: {len(df_mapped.columns)} columns")
    
    return df_mapped

def run_prediction_pipeline(data_path):
    """Run the complete prediction pipeline with the real dataset."""
    print("\n" + "="*60)
    print("RUNNING PREDICTION PIPELINE")
    print("="*60)
    
    # Load and clean data
    df = load_and_clean_kaggle_data(data_path)
    
    # Take a sample for faster processing (optional)
    if len(df) > 10000:
        print(f"Taking sample of 10,000 records for faster processing...")
        df_sample = df.sample(n=10000, random_state=42)
        print(f"Sample shape: {df_sample.shape}")
    else:
        df_sample = df
    
    # Initialize preprocessor
    print("\nInitializing preprocessor...")
    preprocessor = DataPreprocessor()
    
    # Prepare data
    print("Preparing data with SMOTE-ENN...")
    prepared_data = preprocessor.prepare_data(df_sample, 'policy_lapse')
    
    print(f"✓ Training set shape: {prepared_data['X_train'].shape}")
    print(f"✓ Test set shape: {prepared_data['X_test'].shape}")
    print(f"✓ Number of features: {len(prepared_data['feature_names'])}")
    
    # Initialize and train model
    print("\nTraining ensemble model...")
    ensemble_model = ChurnEnsembleModel()
    training_results = ensemble_model.train(
        prepared_data['X_train'], 
        prepared_data['y_train']
    )
    
    print("✓ Model training completed!")
    print("Cross-validation results:")
    for model_name, scores in training_results['individual_scores'].items():
        print(f"  {model_name:20}: AUC = {scores['mean_cv_score']:.4f} (+/- {scores['std_cv_score'] * 2:.4f})")
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluation_results = ensemble_model.evaluate(
        prepared_data['X_test'], 
        prepared_data['y_test']
    )
    
    print("✓ Model evaluation completed!")
    print(f"\nFinal Performance:")
    print(f"  AUC Score:      {evaluation_results['ensemble_auc']:.4f}")
    print(f"  Accuracy:      {evaluation_results['ensemble_accuracy']:.4f}")
    print(f"  Precision:     {evaluation_results['ensemble_precision']:.4f}")
    print(f"  Recall:        {evaluation_results['ensemble_recall']:.4f}")
    print(f"  F1-Score:      {evaluation_results['ensemble_f1']:.4f}")
    
    # Get feature importance
    print("\nAnalyzing feature importance...")
    feature_importance = ensemble_model.get_feature_importance(prepared_data['feature_names'])
    
    print("Top 10 Most Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:30}: {row['importance']:.4f}")
    
    # Save results
    print("\nSaving results...")
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model_path = 'models/kaggle_ensemble_model.joblib'
    ensemble_model.save_model(model_path)
    print(f"✓ Model saved to {model_path}")
    
    # Save preprocessor
    preprocessor_path = 'models/kaggle_preprocessor.joblib'
    preprocessor.save_preprocessor(preprocessor_path)
    print(f"✓ Preprocessor saved to {preprocessor_path}")
    
    # Save processed data
    processed_data_path = 'data/kaggle_processed_data.csv'
    df_sample.to_csv(processed_data_path, index=False)
    print(f"✓ Processed data saved to {processed_data_path}")
    
    return {
        'model': ensemble_model,
        'preprocessor': preprocessor,
        'evaluation_results': evaluation_results,
        'feature_importance': feature_importance,
        'training_results': training_results
    }

def make_sample_predictions(model, preprocessor, df_sample):
    """Make sample predictions on the dataset."""
    print("\n" + "="*60)
    print("MAKING SAMPLE PREDICTIONS")
    print("="*60)
    
    # Take a few sample records
    sample_records = df_sample.head(5)
    
    print("Sample policy records:")
    print(sample_records[['policy_id', 'age', 'gender', 'policy_type', 'policy_amount', 'premium_amount', 'policy_lapse']].to_string(index=False))
    
    predictions = []
    for _, record in sample_records.iterrows():
        # Prepare single record
        record_df = record.to_frame().T.drop(columns=['policy_lapse'])
        
        # Apply preprocessing
        record_engineered = preprocessor.engineer_features(record_df)
        record_encoded = preprocessor.encode_categorical_features(record_engineered)
        record_scaled = preprocessor.scale_features(record_encoded.values, fit=False)
        
        # Make prediction
        lapse_probability = model.predict_proba(record_scaled)[0, 1]
        lapse_prediction = int(lapse_probability > 0.5)
        
        # Calculate risk level
        if lapse_probability < 0.3:
            risk_level = "Low"
        elif lapse_probability < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        predictions.append({
            'policy_id': record['policy_id'],
            'actual_lapse': record['policy_lapse'],
            'predicted_lapse': lapse_prediction,
            'lapse_probability': lapse_probability,
            'risk_level': risk_level,
            'correct': record['policy_lapse'] == lapse_prediction
        })
    
    # Display predictions
    predictions_df = pd.DataFrame(predictions)
    print("\nPrediction Results:")
    print(predictions_df.to_string(index=False, float_format='%.4f'))
    
    accuracy = predictions_df['correct'].mean()
    print(f"\nSample Accuracy: {accuracy:.2%}")

def main():
    """Main function."""
    print("="*60)
    print("REAL KAGGLE INSURANCE DATASET INTEGRATION")
    print("="*60)
    
    # Check if processed data exists
    processed_data_path = 'data/kaggle_processed_data.csv'
    if os.path.exists(processed_data_path):
        print("Found existing processed data. Loading...")
        df_sample = pd.read_csv(processed_data_path)
        print(f"✓ Loaded processed data with shape: {df_sample.shape}")
        
        # Load saved model and preprocessor
        try:
            model = ChurnEnsembleModel()
            model.load_model('models/kaggle_ensemble_model.joblib')
            
            preprocessor = DataPreprocessor()
            preprocessor.load_preprocessor('models/kaggle_preprocessor.joblib')
            
            print("✓ Loaded saved model and preprocessor")
            
            # Make sample predictions
            make_sample_predictions(model, preprocessor, df_sample)
            
        except Exception as e:
            print(f"Could not load saved model: {e}")
            print("Running full pipeline...")
            run_prediction_pipeline('data/Kaggle.csv')
    else:
        print("Running full prediction pipeline...")
        results = run_prediction_pipeline('data/Kaggle.csv')
        
        # Make sample predictions
        make_sample_predictions(
            results['model'], 
            results['preprocessor'], 
            pd.read_csv('data/kaggle_processed_data.csv')
        )
    
    print("\n" + "="*60)
    print("INTEGRATION COMPLETE!")
    print("="*60)
    print("The real Kaggle insurance dataset has been successfully integrated!")
    print("You can now:")
    print("1. Use the trained model for predictions")
    print("2. Start the API: python src/api/app.py")
    print("3. Launch the dashboard: python src/dashboard/app.py")

if __name__ == "__main__":
    main()
