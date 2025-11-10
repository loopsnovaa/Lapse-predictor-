#!/usr/bin/env python3
"""
Script to examine the Kaggle dataset structure and integrate it with our insurance system.
"""

import pandas as pd
import numpy as np
import os

def examine_dataset(file_path):
    """Examine the dataset structure and content."""
    print("="*60)
    print("EXAMINING KAGGLE DATASET")
    print("="*60)
    
    try:
        # Try different approaches to read the CSV
        print("Attempting to read CSV file...")
        
        # First, let's see the first few lines to understand the format
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:10]
            print("First 10 lines of the file:")
            for i, line in enumerate(lines):
                print(f"Line {i+1}: {line.strip()}")
        
        print("\n" + "="*40)
        
        # Try reading with different parameters
        try:
            df = pd.read_csv(file_path, sep=';', encoding='utf-8')
            print("✓ Successfully read with semicolon separator and UTF-8 encoding")
        except:
            try:
                df = pd.read_csv(file_path, sep=';', encoding='latin-1')
                print("✓ Successfully read with semicolon separator and Latin-1 encoding")
            except:
                try:
                    df = pd.read_csv(file_path, sep=';', encoding='cp1252')
                    print("✓ Successfully read with semicolon separator and CP1252 encoding")
                except:
                    try:
                        df = pd.read_csv(file_path, encoding='utf-8')
                        print("✓ Successfully read with comma separator and UTF-8 encoding")
                    except Exception as e:
                        print(f"❌ Failed to read CSV: {e}")
                        return None
        
        print(f"\nDataset shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\nColumns:")
        for i, col in enumerate(df.columns):
            print(f"{i+1:2d}. {col}")
        
        print("\nData types:")
        print(df.dtypes)
        
        print("\nFirst 5 rows:")
        print(df.head())
        
        print("\nBasic statistics:")
        print(df.describe())
        
        print("\nMissing values:")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("No missing values found")
        
        return df
        
    except Exception as e:
        print(f"Error examining dataset: {e}")
        return None

def identify_dataset_type(df):
    """Identify what type of dataset this is."""
    print("\n" + "="*60)
    print("IDENTIFYING DATASET TYPE")
    print("="*60)
    
    columns = df.columns.tolist()
    
    # Check for common insurance-related columns
    insurance_keywords = [
        'policy', 'premium', 'claim', 'insurance', 'lapse', 'churn',
        'customer', 'client', 'age', 'gender', 'income', 'risk'
    ]
    
    insurance_columns = []
    for col in columns:
        if any(keyword in col.lower() for keyword in insurance_keywords):
            insurance_columns.append(col)
    
    print(f"Found {len(insurance_columns)} insurance-related columns:")
    for col in insurance_columns:
        print(f"  - {col}")
    
    # Check for target variable
    target_candidates = []
    for col in columns:
        if any(keyword in col.lower() for keyword in ['target', 'label', 'churn', 'lapse', 'response']):
            target_candidates.append(col)
    
    print(f"\nPotential target variables:")
    for col in target_candidates:
        print(f"  - {col}")
        if col in df.columns:
            print(f"    Values: {df[col].value_counts().to_dict()}")
    
    return insurance_columns, target_candidates

def create_insurance_mapping(df):
    """Create a mapping to adapt the dataset for insurance lapse prediction."""
    print("\n" + "="*60)
    print("CREATING INSURANCE MAPPING")
    print("="*60)
    
    # Create a copy for mapping
    df_mapped = df.copy()
    
    # Try to identify and map columns
    column_mapping = {}
    
    # Look for age-related columns
    age_cols = [col for col in df.columns if 'age' in col.lower()]
    if age_cols:
        column_mapping['age'] = age_cols[0]
        print(f"Mapped age: {age_cols[0]}")
    
    # Look for gender-related columns
    gender_cols = [col for col in df.columns if 'gender' in col.lower() or 'sex' in col.lower()]
    if gender_cols:
        column_mapping['gender'] = gender_cols[0]
        print(f"Mapped gender: {gender_cols[0]}")
    
    # Look for income-related columns
    income_cols = [col for col in df.columns if 'income' in col.lower() or 'salary' in col.lower()]
    if income_cols:
        column_mapping['income'] = income_cols[0]
        print(f"Mapped income: {income_cols[0]}")
    
    # Look for policy-related columns
    policy_cols = [col for col in df.columns if 'policy' in col.lower()]
    if policy_cols:
        column_mapping['policy_amount'] = policy_cols[0]
        print(f"Mapped policy amount: {policy_cols[0]}")
    
    # Look for premium-related columns
    premium_cols = [col for col in df.columns if 'premium' in col.lower()]
    if premium_cols:
        column_mapping['premium_amount'] = premium_cols[0]
        print(f"Mapped premium amount: {premium_cols[0]}")
    
    # Create a synthetic target if none exists
    if not any('target' in col.lower() or 'churn' in col.lower() or 'lapse' in col.lower() for col in df.columns):
        print("Creating synthetic policy_lapse target...")
        np.random.seed(42)
        df_mapped['policy_lapse'] = np.random.choice([0, 1], len(df_mapped), p=[0.85, 0.15])
        print("✓ Created synthetic policy_lapse target (15% lapse rate)")
    
    return df_mapped, column_mapping

def main():
    """Main function."""
    file_path = 'data/Kaggle.csv'
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    # Examine the dataset
    df = examine_dataset(file_path)
    if df is None:
        return
    
    # Identify dataset type
    insurance_cols, target_cols = identify_dataset_type(df)
    
    # Create insurance mapping
    df_mapped, mapping = create_insurance_mapping(df)
    
    # Save the mapped dataset
    output_path = 'data/kaggle_insurance_data.csv'
    df_mapped.to_csv(output_path, index=False)
    print(f"\n✓ Mapped dataset saved to: {output_path}")
    
    print("\n" + "="*60)
    print("INTEGRATION COMPLETE")
    print("="*60)
    print("You can now use this dataset with the insurance prediction system:")
    print("python demo.py --data_path data/kaggle_insurance_data.csv")

if __name__ == "__main__":
    main()
