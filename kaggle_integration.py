#!/usr/bin/env python3
"""
Kaggle Insurance Dataset Integration Script

This script downloads and processes real insurance datasets from Kaggle
for the policy lapse prediction system.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_kaggle_dataset(dataset_name: str, output_dir: str = "data") -> str:
    """
    Download a dataset from Kaggle (requires kaggle API setup).
    
    Args:
        dataset_name: Name of the Kaggle dataset
        output_dir: Directory to save the dataset
        
    Returns:
        Path to the downloaded dataset
    """
    try:
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        # Initialize Kaggle API
        api = KaggleApi()
        api.authenticate()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Download dataset
        logger.info(f"Downloading dataset: {dataset_name}")
        api.dataset_download_files(dataset_name, path=output_dir, unzip=True)
        
        logger.info(f"Dataset downloaded successfully to {output_dir}")
        return output_dir
        
    except ImportError:
        logger.error("Kaggle API not installed. Install with: pip install kaggle")
        return None
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        return None

def load_prudential_dataset(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess the Prudential Life Insurance dataset.
    
    Args:
        file_path: Path to the dataset file
        
    Returns:
        Preprocessed DataFrame
    """
    try:
        # Load the dataset
        df = pd.read_csv(file_path)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        
        # Basic preprocessing for Prudential dataset
        # The dataset has features like Product_Info_1-7, Ins_Age, Ht, Wt, etc.
        
        # Create a binary target for lapse prediction
        # For demonstration, we'll create a synthetic lapse target
        # In real scenarios, this would be based on actual lapse data
        np.random.seed(42)
        df['policy_lapse'] = np.random.choice([0, 1], len(df), p=[0.85, 0.15])
        
        # Select relevant features for insurance lapse prediction
        feature_columns = [
            'Product_Info_1', 'Product_Info_2', 'Product_Info_3', 'Product_Info_4',
            'Product_Info_5', 'Product_Info_6', 'Product_Info_7',
            'Ins_Age', 'Ht', 'Wt', 'BMI',
            'Employment_Info_1', 'Employment_Info_2', 'Employment_Info_3',
            'Employment_Info_4', 'Employment_Info_5', 'Employment_Info_6',
            'InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3', 'InsuredInfo_4',
            'InsuredInfo_5', 'InsuredInfo_6', 'InsuredInfo_7',
            'Insurance_History_1', 'Insurance_History_2', 'Insurance_History_3',
            'Insurance_History_4', 'Insurance_History_5', 'Insurance_History_7',
            'Insurance_History_8', 'Insurance_History_9',
            'Family_Hist_1', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4',
            'Family_Hist_5',
            'Medical_History_1', 'Medical_History_2', 'Medical_History_3',
            'Medical_History_4', 'Medical_History_5', 'Medical_History_6',
            'Medical_History_7', 'Medical_History_8', 'Medical_History_9',
            'Medical_History_10', 'Medical_History_11', 'Medical_History_12',
            'Medical_History_13', 'Medical_History_14', 'Medical_History_15',
            'Medical_History_16', 'Medical_History_17', 'Medical_History_18',
            'Medical_History_19', 'Medical_History_20', 'Medical_History_21',
            'Medical_History_22', 'Medical_History_23', 'Medical_History_24',
            'Medical_History_25', 'Medical_History_26', 'Medical_History_27',
            'Medical_History_28', 'Medical_History_29', 'Medical_History_30',
            'Medical_History_31', 'Medical_History_32', 'Medical_History_33',
            'Medical_History_34', 'Medical_History_35', 'Medical_History_36',
            'Medical_History_37', 'Medical_History_38', 'Medical_History_39',
            'Medical_History_40', 'Medical_History_41',
            'Medical_Keyword_1', 'Medical_Keyword_2', 'Medical_Keyword_3',
            'Medical_Keyword_4', 'Medical_Keyword_5', 'Medical_Keyword_6',
            'Medical_Keyword_7', 'Medical_Keyword_8', 'Medical_Keyword_9',
            'Medical_Keyword_10', 'Medical_Keyword_11', 'Medical_Keyword_12',
            'Medical_Keyword_13', 'Medical_Keyword_14', 'Medical_Keyword_15',
            'Medical_Keyword_16', 'Medical_Keyword_17', 'Medical_Keyword_18',
            'Medical_Keyword_19', 'Medical_Keyword_20', 'Medical_Keyword_21',
            'Medical_Keyword_22', 'Medical_Keyword_23', 'Medical_Keyword_24',
            'Medical_Keyword_25', 'Medical_Keyword_26', 'Medical_Keyword_27',
            'Medical_Keyword_28', 'Medical_Keyword_29', 'Medical_Keyword_30',
            'Medical_Keyword_31', 'Medical_Keyword_32', 'Medical_Keyword_33',
            'Medical_Keyword_34', 'Medical_Keyword_35', 'Medical_Keyword_36',
            'Medical_Keyword_37', 'Medical_Keyword_38', 'Medical_Keyword_39',
            'Medical_Keyword_40', 'Medical_Keyword_41', 'Medical_Keyword_42',
            'Medical_Keyword_43', 'Medical_Keyword_44', 'Medical_Keyword_45',
            'Medical_Keyword_46', 'Medical_Keyword_47', 'Medical_Keyword_48',
            'policy_lapse'
        ]
        
        # Select only available columns
        available_columns = [col for col in feature_columns if col in df.columns]
        df_selected = df[available_columns].copy()
        
        # Handle missing values
        df_selected = df_selected.fillna(df_selected.median())
        
        logger.info(f"Preprocessed dataset shape: {df_selected.shape}")
        logger.info(f"Policy lapse distribution: {df_selected['policy_lapse'].value_counts().to_dict()}")
        
        return df_selected
        
    except Exception as e:
        logger.error(f"Error loading Prudential dataset: {e}")
        return None

def create_insurance_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Create realistic insurance policy sample data.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Sample DataFrame
    """
    np.random.seed(42)
    
    data = {
        'policy_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 80, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'income': np.random.uniform(20000, 150000, n_samples),
        'policy_type': np.random.choice(['Life', 'Health', 'Auto', 'Home'], n_samples),
        'policy_amount': np.random.uniform(10000, 500000, n_samples),
        'premium_amount': np.random.uniform(50, 2000, n_samples),
        'policy_tenure_months': np.random.randint(1, 240, n_samples),
        'payment_frequency': np.random.choice(['Monthly', 'Quarterly', 'Semi-Annual', 'Annual'], n_samples),
        'payment_method': np.random.choice(['Bank Transfer', 'Credit Card', 'Check', 'Cash'], n_samples),
        'claims_history': np.random.randint(0, 5, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'employment_status': np.random.choice(['Employed', 'Self-Employed', 'Unemployed', 'Retired'], n_samples),
        'smoking_status': np.random.choice(['Non-Smoker', 'Smoker', 'Former Smoker'], n_samples),
        'health_conditions': np.random.choice(['None', 'Diabetes', 'Hypertension', 'Heart Disease'], n_samples),
        'policy_lapse': np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    }
    
    return pd.DataFrame(data)

def main():
    """Main function to demonstrate Kaggle dataset integration."""
    print("="*60)
    print("KAGGLE INSURANCE DATASET INTEGRATION")
    print("="*60)
    
    # Option 1: Use sample data (no Kaggle API required)
    print("Creating sample insurance data...")
    sample_data = create_insurance_sample_data(1000)
    print(f"✓ Sample data created with shape: {sample_data.shape}")
    print(f"✓ Policy lapse rate: {sample_data['policy_lapse'].mean():.2%}")
    
    # Save sample data
    os.makedirs('data', exist_ok=True)
    sample_data.to_csv('data/insurance_sample_data.csv', index=False)
    print("✓ Sample data saved to data/insurance_sample_data.csv")
    
    # Option 2: Download real Kaggle dataset (requires setup)
    print("\nTo use real Kaggle datasets:")
    print("1. Install Kaggle API: pip install kaggle")
    print("2. Setup Kaggle credentials: https://github.com/Kaggle/kaggle-api")
    print("3. Download dataset: kaggle competitions download -c prudential-life-insurance-assessment")
    
    print("\n" + "="*60)
    print("INTEGRATION COMPLETED")
    print("="*60)
    print("You can now use the insurance data with the prediction system:")
    print("python demo.py")

if __name__ == "__main__":
    main()



