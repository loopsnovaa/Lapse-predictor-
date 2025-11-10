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
        """
        Initialize the data preprocessor.
        
        Args:
            config: Configuration dictionary with preprocessing parameters
        """
        self.config = config or {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.smote_enn = SMOTEENN(
            smote=SMOTE(random_state=42, k_neighbors=3),
            enn=EditedNearestNeighbours(n_neighbors=3)
        )
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Loaded DataFrame
        """
        try:
            data = pd.read_csv(file_path)
            logger.info(f"Loaded data with shape: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and outliers.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning...")
        
        # Handle missing values
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        categorical_columns = data.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        for col in numeric_columns:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].median(), inplace=True)
                logger.info(f"Filled missing values in {col} with median")
        
        # Fill categorical missing values with mode
        for col in categorical_columns:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].mode()[0], inplace=True)
                logger.info(f"Filled missing values in {col} with mode")
        
        # Remove duplicates
        initial_rows = len(data)
        data = data.drop_duplicates()
        removed_rows = initial_rows - len(data)
        if removed_rows > 0:
            logger.info(f"Removed {removed_rows} duplicate rows")
        
        logger.info(f"Data cleaning completed. Final shape: {data.shape}")
        return data
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features for better model performance.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering...")
        
        # Create a copy to avoid modifying original data
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
        
        if 'credit_score' in df.columns:
            df['credit_risk_category'] = pd.cut(df['credit_score'], 
                                               bins=[0, 580, 670, 740, 850], 
                                               labels=['Poor', 'Fair', 'Good', 'Excellent'])
        
        # Create interaction features for key insurance variables
        insurance_features = ['age', 'income', 'policy_amount', 'premium_amount', 'credit_score']
        available_features = [col for col in insurance_features if col in df.columns]
        
        for i, col1 in enumerate(available_features):
            for col2 in available_features[i+1:]:
                if col1 != col2:
                    df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
        
        # Create polynomial features for important insurance variables
        important_features = ['age', 'income', 'policy_amount', 'premium_amount', 'credit_score']
        for feature in important_features:
            if feature in df.columns:
                df[f'{feature}_squared'] = df[feature] ** 2
                df[f'{feature}_log'] = np.log1p(df[feature])
        
        logger.info(f"Feature engineering completed. New shape: {df.shape}")
        return df
    
    def encode_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using label encoding.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with encoded categorical features
        """
        logger.info("Starting categorical encoding...")
        
        df = data.copy()
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # Handle unseen categories
                unique_values = df[col].unique()
                known_values = self.label_encoders[col].classes_
                unknown_values = set(unique_values) - set(known_values)
                
                if unknown_values:
                    logger.warning(f"Unknown categories in {col}: {unknown_values}")
                    # Map unknown categories to the most frequent category
                    most_frequent = df[col].mode()[0]
                    df[col] = df[col].replace(list(unknown_values), most_frequent)
                
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        logger.info("Categorical encoding completed")
        return df
    
    def apply_smote_enn(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE-ENN to balance the dataset.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Balanced feature matrix and target vector
        """
        logger.info("Applying SMOTE-ENN for class balancing...")
        
        # Check class distribution before balancing
        unique, counts = np.unique(y, return_counts=True)
        logger.info(f"Class distribution before SMOTE-ENN: {dict(zip(unique, counts))}")
        
        # Apply SMOTE-ENN
        X_balanced, y_balanced = self.smote_enn.fit_resample(X, y)
        
        # Check class distribution after balancing
        unique, counts = np.unique(y_balanced, return_counts=True)
        logger.info(f"Class distribution after SMOTE-ENN: {dict(zip(unique, counts))}")
        
        logger.info(f"SMOTE-ENN completed. New shape: {X_balanced.shape}")
        return X_balanced, y_balanced
    
    def scale_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Scale features using StandardScaler.
        
        Args:
            X: Feature matrix
            fit: Whether to fit the scaler or use existing fit
            
        Returns:
            Scaled feature matrix
        """
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            logger.info("Features scaled (fitted)")
        else:
            X_scaled = self.scaler.transform(X)
            logger.info("Features scaled (transformed)")
        
        return X_scaled
    
    def prepare_data(self, data: pd.DataFrame, target_column: str, 
                    test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Complete data preparation pipeline.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            test_size: Proportion of data for testing
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary containing prepared data splits
        """
        logger.info("Starting complete data preparation pipeline...")
        
        # Clean data
        data_clean = self.clean_data(data)
        
        # Engineer features
        data_engineered = self.engineer_features(data_clean)
        
        # Separate features and target
        X = data_engineered.drop(columns=[target_column])
        y = data_engineered[target_column]
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Encode categorical features
        X_encoded = self.encode_categorical_features(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scale_features(X_train.values, fit=True)
        X_test_scaled = self.scale_features(X_test.values, fit=False)
        
        # Apply SMOTE-ENN to training data only
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
        """
        Save the preprocessor components.
        
        Args:
            file_path: Path to save the preprocessor
        """
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'config': self.config
        }
        joblib.dump(preprocessor_data, file_path)
        logger.info(f"Preprocessor saved to {file_path}")
    
    def load_preprocessor(self, file_path: str):
        """
        Load the preprocessor components.
        
        Args:
            file_path: Path to load the preprocessor from
        """
        preprocessor_data = joblib.load(file_path)
        self.scaler = preprocessor_data['scaler']
        self.label_encoders = preprocessor_data['label_encoders']
        self.feature_names = preprocessor_data['feature_names']
        self.config = preprocessor_data['config']
        logger.info(f"Preprocessor loaded from {file_path}")


def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Create sample insurance policy data for testing purposes.
    
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
        'policy_tenure_months': np.random.randint(1, 240, n_samples),  # 1 month to 20 years
        'payment_frequency': np.random.choice(['Monthly', 'Quarterly', 'Semi-Annual', 'Annual'], n_samples),
        'payment_method': np.random.choice(['Bank Transfer', 'Credit Card', 'Check', 'Cash'], n_samples),
        'claims_history': np.random.randint(0, 5, n_samples),  # Number of claims
        'credit_score': np.random.randint(300, 850, n_samples),
        'employment_status': np.random.choice(['Employed', 'Self-Employed', 'Unemployed', 'Retired'], n_samples),
        'smoking_status': np.random.choice(['Non-Smoker', 'Smoker', 'Former Smoker'], n_samples),
        'health_conditions': np.random.choice(['None', 'Diabetes', 'Hypertension', 'Heart Disease'], n_samples),
        'policy_lapse': np.random.choice([0, 1], n_samples, p=[0.85, 0.15])  # 15% lapse rate
    }
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    
    # Create sample data
    sample_data = create_sample_data(1000)
    print("Sample data created:")
    print(sample_data.head())
    print(f"\nData shape: {sample_data.shape}")
    print(f"Churn distribution:\n{sample_data['churn'].value_counts()}")
    
    # Prepare data
    prepared_data = preprocessor.prepare_data(sample_data, 'churn')
    
    print(f"\nPrepared data shapes:")
    print(f"X_train: {prepared_data['X_train'].shape}")
    print(f"X_test: {prepared_data['X_test'].shape}")
    print(f"y_train: {prepared_data['y_train'].shape}")
    print(f"y_test: {prepared_data['y_test'].shape}")
