#!/usr/bin/env python3
"""
Real Kaggle Insurance Dataset Integration Script
"""

import pandas as pd
import numpy as np
import os
import sys
import joblib  # Import joblib at top level
sys.path.append('src')

try:
    from data.preprocessing import DataPreprocessor
    from models.ensemble import ChurnEnsembleModel
except ImportError:
    print("❌ Error: Could not import modules.")
    sys.exit(1)

def load_and_clean_kaggle_data(file_path):
    print("Loading data...")
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        if df.shape[1] < 2:
            df = pd.read_csv(file_path, sep=';', encoding='utf-8')
    except Exception as e:
        sys.exit(f"❌ Failed to load CSV file: {e}")

    df.columns = df.columns.str.strip()
    df = df.drop(columns=['Unnamed: 20', 'Unnamed: 21'], errors='ignore')
    
    # Target Logic
    if 'POLICY STATUS' in df.columns:
        df['policy_lapse'] = (df['POLICY STATUS'] == 'Lapse').astype(int)
    else:
        df['policy_lapse'] = np.random.randint(0, 2, size=len(df))
    
    # Mapping
    df_mapped = pd.DataFrame()
    df_mapped['policy_id'] = range(1, len(df) + 1)
    
    # Financials (Clean strings like '1,000')
    if 'BENEFIT' in df.columns:
        df['BENEFIT'] = df['BENEFIT'].astype(str).str.replace(',', '').str.replace('-', '0')
        df_mapped['policy_amount'] = pd.to_numeric(df['BENEFIT'], errors='coerce').fillna(0)
    else:
        df_mapped['policy_amount'] = 0

    if 'Premium' in df.columns:
        df['Premium'] = df['Premium'].astype(str).str.replace(',', '').str.replace('-', '0')
        df_mapped['premium_amount'] = pd.to_numeric(df['Premium'], errors='coerce').fillna(0)
    else:
        df_mapped['premium_amount'] = 0

    df_mapped['income'] = df_mapped['policy_amount'] * 0.1
    df_mapped['policy_lapse'] = df['policy_lapse']
    
    # Defaults
    defaults = {
        'age': 30, 'gender': 'Unknown', 'policy_tenure_months': 12,
        'policy_type': 'General', 'payment_frequency': 'Monthly',
        'marital_status': 'Unknown', 'education': 'Unknown',
        'payment_method': 'Bank Transfer', 'claims_history': 0,
        'credit_score': 700, 'employment_status': 'Employed',
        'smoking_status': 'Non-Smoker', 'num_support_calls': 0,
        'years_with_company': 1, 'total_spend': df_mapped['premium_amount'],
        'contract_type': '1 Year',
        'health_conditions': 'None',
        'service_rating': 2  # Numeric 2
    }
    
    for col, val in defaults.items():
        df_mapped[col] = val

    return df_mapped

def run_prediction_pipeline(data_path):
    print("\n" + "="*60)
    print("RUNNING PREDICTION PIPELINE")
    print("="*60)
    
    df = load_and_clean_kaggle_data(data_path)
    
    # Sample to speed up testing
    if len(df) > 10000:
        print(f"Taking sample of 10,000 records...")
        df_sample = df.sample(n=10000, random_state=42)
    else:
        df_sample = df
    
    print("\nInitializing preprocessor...")
    preprocessor = DataPreprocessor()
    
    print("Preparing data...")
    prepared_data = preprocessor.prepare_data(df_sample, 'policy_lapse')
    
    print(f"✓ Training set shape: {prepared_data['X_train'].shape}")
    
    print("\nTraining ensemble model...")
    ensemble_model = ChurnEnsembleModel()
    training_results = ensemble_model.train(
        prepared_data['X_train'], 
        prepared_data['y_train']
    )
    
    print("\nEvaluating model...")
    evaluation_results = ensemble_model.evaluate(
        prepared_data['X_test'], 
        prepared_data['y_test']
    )
    
    print(f"\nFinal AUC Score: {evaluation_results['ensemble_auc']:.4f}")
    
    # --- FIX: USE .save_model() INSTEAD OF .save() ---
    print("\nSaving results...")
    os.makedirs('models', exist_ok=True)
    
    # 1. Save Model
    ensemble_model.save_model('models/kaggle_ensemble_model.joblib')
    
    # 2. Save Preprocessor
    # If the preprocessor class doesn't have a save method, dump it directly
    if hasattr(preprocessor, 'save_preprocessor'):
        preprocessor.save_preprocessor('models/kaggle_preprocessor.joblib')
    else:
        joblib.dump(preprocessor, 'models/kaggle_preprocessor.joblib')
    
    # 3. Save Data
    df_sample.to_csv('data/kaggle_processed_data.csv', index=False)
    print("✓ All artifacts saved.")

    return ensemble_model, preprocessor

def main():
    csv_path = 'data/Kaggle.csv'
    if not os.path.exists(csv_path):
        if os.path.exists('Kaggle.csv'):
            csv_path = 'Kaggle.csv'
        else:
            print("❌ Kaggle.csv not found.")
            return

    run_prediction_pipeline(csv_path)

if __name__ == "__main__":
    main()