"""
Data preprocessing and feature engineering module for churn prediction.
Includes SMOTE-ENN for handling class imbalance.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
import joblib
import logging
from typing import Tuple, Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles data preprocessing, feature engineering, and class imbalance correction.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.smote_enn = SMOTEENN(
            smote=SMOTE(random_state=42, k_neighbors=3),
            enn=EditedNearestNeighbours(n_neighbors=3)
        )
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        try:
            data = pd.read_csv(file_path)
            logger.info(f"Loaded data with shape: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting data cleaning...")
        
        # Handle missing values
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        
        # Fill numeric missing values with median
        for col in numeric_columns:
            if data[col].isnull().sum() > 0:
                data[col] = data[col].fillna(data[col].median())
        
        # Fill categorical missing values with mode
        for col in categorical_columns:
            if data[col].isnull().sum() > 0:
                data[col] = data[col].fillna(data[col].mode()[0])
        
        # Remove duplicates
        data = data.drop_duplicates()
        
        return data
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting feature engineering...")
        df = data.copy()
        
        # Insurance-specific feature engineering
        if 'policy_tenure_months' in df.columns and 'premium_amount' in df.columns:
            df['premium_to_tenure_ratio'] = df['premium_amount'] / (df['policy_tenure_months'] + 1)
        
        if 'policy_amount' in df.columns and 'premium_amount' in df.columns:
            df['premium_to_coverage_ratio'] = df['premium_amount'] / (df['policy_amount'] + 1)
        
        if 'income' in df.columns and 'premium_amount' in df.columns:
            df['premium_to_income_ratio'] = df['premium_amount'] / (df['income'] + 1)
        
        if 'age' in df.columns and 'policy_tenure_months' in df.columns:
            df['age_at_policy_start'] = df['age'] - (df['policy_tenure_months'] / 12)
        
        # Risk indicators
        if 'claims_history' in df.columns and 'policy_tenure_months' in df.columns:
            df['claims_per_year'] = df['claims_history'] / ((df['policy_tenure_months'] / 12) + 1)
        
        # --- FIX 1: Use Numeric Labels for Risk Category ---
        # This prevents the "could not convert string to float: 'Good'" error
        if 'credit_score' in df.columns:
            df['credit_risk_category'] = pd.cut(df['credit_score'], 
                                               bins=[0, 580, 670, 740, 850], 
                                               labels=[0, 1, 2, 3]) # Numeric labels (0=Poor, 3=Excellent)
            df['credit_risk_category'] = df['credit_risk_category'].astype(float)
        
        # Create interaction features
        insurance_features = ['age', 'income', 'policy_amount', 'premium_amount', 'credit_score']
        available_features = [col for col in insurance_features if col in df.columns]
        
        for i, col1 in enumerate(available_features):
            for col2 in available_features[i+1:]:
                if col1 != col2:
                    df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
        
        logger.info(f"Feature engineering completed. New shape: {df.shape}")
        return df
    
    def encode_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting categorical encoding...")
        df = data.copy()
        
        # --- FIX 2: Include 'category' dtype in encoding ---
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_columns:
            # Convert to string to ensure LabelEncoder works
            df[col] = df[col].astype(str)
            
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                # Handle unseen categories
                unique_values = df[col].unique()
                known_values = self.label_encoders[col].classes_
                unknown_values = set(unique_values) - set(known_values)
                
                if unknown_values:
                    # Map unknown categories to the most frequent category (mode)
                    most_frequent = self.label_encoders[col].classes_[0] 
                    df[col] = df[col].replace(list(unknown_values), most_frequent)
                
                df[col] = self.label_encoders[col].transform(df[col])
        
        logger.info("Categorical encoding completed")
        return df
    
    def apply_smote_enn(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        logger.info("Applying SMOTE-ENN for class balancing...")
        try:
            X_balanced, y_balanced = self.smote_enn.fit_resample(X, y)
            logger.info(f"SMOTE-ENN completed. New shape: {X_balanced.shape}")
            return X_balanced, y_balanced
        except Exception as e:
            logger.warning(f"SMOTE-ENN failed ({e}). Returning original data.")
            return X, y
    
    def scale_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)
    
    def prepare_data(self, data: pd.DataFrame, target_column: str, 
                     test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        logger.info("Starting complete data preparation pipeline...")
        
        # 1. Clean
        data_clean = self.clean_data(data)
        
        # 2. Engineer
        data_engineered = self.engineer_features(data_clean)
        
        # 3. Separate
        if target_column not in data_engineered.columns:
             raise ValueError(f"Target column {target_column} not found!")
             
        X = data_engineered.drop(columns=[target_column])
        y = data_engineered[target_column]
        
        self.feature_names = X.columns.tolist()
        
        # 4. Encode
        X_encoded = self.encode_categorical_features(X)
        
        # 5. Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # 6. Scale
        X_train_scaled = self.scale_features(X_train.values, fit=True)
        X_test_scaled = self.scale_features(X_test.values, fit=False)
        
        # 7. Balance (SMOTE)
        X_train_balanced, y_train_balanced = self.apply_smote_enn(X_train_scaled, y_train.values)
        
        logger.info("Data preparation pipeline completed")
        
        return {
            'X_train': X_train_balanced,
            'X_test': X_test_scaled,
            'y_train': y_train_balanced,
            'y_test': y_test.values,
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders
        }

    def save_preprocessor(self, file_path: str):
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'config': self.config
        }
        joblib.dump(preprocessor_data, file_path)
        logger.info(f"Preprocessor saved to {file_path}")
    
    def load_preprocessor(self, file_path: str):
        preprocessor_data = joblib.load(file_path)
        self.scaler = preprocessor_data['scaler']
        self.label_encoders = preprocessor_data['label_encoders']
        self.feature_names = preprocessor_data['feature_names']
        self.config = preprocessor_data['config']
        logger.info(f"Preprocessor loaded from {file_path}")


def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
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

# --- FIX 3: Correct Main Block Logic ---
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    sample_data = create_sample_data(1000)
    print("Sample data created.")
    
    # Use correct column name 'policy_lapse' instead of 'churn'
    prepared_data = preprocessor.prepare_data(sample_data, 'policy_lapse')
    
    print(f"X_train: {prepared_data['X_train'].shape}")
    print(f"y_train: {prepared_data['y_train'].shape}")