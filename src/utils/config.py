"""
Utility functions and configurations for the churn prediction system.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('churn_prediction.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class Config:
    """Configuration class for the churn prediction system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            'data': {
                'test_size': 0.2,
                'random_state': 42,
                'target_column': 'churn',
                'sample_data_size': 1000
            },
            'preprocessing': {
                'apply_smote_enn': True,
                'feature_engineering': True,
                'scaling': True,
                'encoding': True
            },
            'model': {
                'enable_hyperparameter_tuning': False,
                'cv_folds': 5,
                'ensemble_method': 'stacking',
                'base_models': ['logistic_regression', 'xgboost']
            },
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'reload': False
            },
            'dashboard': {
                'host': '127.0.0.1',
                'port': 8050,
                'debug': False,
                'refresh_interval': 30000
            },
            'mlflow': {
                'experiment_name': 'churn_prediction',
                'tracking_uri': 'file:./mlruns'
            },
            'paths': {
                'data_dir': 'data',
                'models_dir': 'models',
                'logs_dir': 'logs',
                'reports_dir': 'reports'
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, file_path: str):
        """Save configuration to file."""
        with open(file_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def create_directories(self):
        """Create necessary directories."""
        directories = [
            self.get('paths.data_dir'),
            self.get('paths.models_dir'),
            self.get('paths.logs_dir'),
            self.get('paths.reports_dir')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")


def setup_logging(log_level: str = 'INFO', log_file: str = 'churn_prediction.log'):
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Log file path
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def create_sample_config(output_path: str = 'config.json'):
    """
    Create a sample configuration file.
    
    Args:
        output_path: Output path for configuration file
    """
    config = Config()
    config.save(output_path)
    logger.info(f"Sample configuration saved to {output_path}")


def validate_data_format(data: Dict[str, Any]) -> bool:
    """
    Validate data format for prediction.
    
    Args:
        data: Input data dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = [
        'age', 'tenure', 'monthly_charges', 'total_charges',
        'gender', 'partner', 'dependents', 'phone_service',
        'internet_service', 'contract', 'payment_method'
    ]
    
    for field in required_fields:
        if field not in data:
            logger.warning(f"Missing required field: {field}")
            return False
    
    return True


def format_prediction_response(prediction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format prediction response for API.
    
    Args:
        prediction: Raw prediction data
        
    Returns:
        Formatted prediction response
    """
    return {
        'customer_id': prediction.get('customer_id'),
        'churn_probability': round(prediction.get('churn_probability', 0), 4),
        'churn_prediction': prediction.get('churn_prediction', 0),
        'risk_level': prediction.get('risk_level', 'Unknown'),
        'confidence': round(prediction.get('confidence', 0), 4),
        'timestamp': datetime.now().isoformat(),
        'model_version': prediction.get('model_version', '1.0.0')
    }


def calculate_model_performance_metrics(y_true, y_pred, y_pred_proba):
    """
    Calculate comprehensive model performance metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities
        
    Returns:
        Dictionary of performance metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'classification_report': classification_report(y_true, y_pred, output_dict=True)
    }
    
    return metrics


def save_prediction_log(prediction_data: Dict[str, Any], log_file: str = 'predictions.log'):
    """
    Save prediction to log file.
    
    Args:
        prediction_data: Prediction data to log
        log_file: Log file path
    """
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'prediction': prediction_data
    }
    
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')


def load_prediction_logs(log_file: str = 'predictions.log', n_entries: int = 100):
    """
    Load recent prediction logs.
    
    Args:
        log_file: Log file path
        n_entries: Number of recent entries to load
        
    Returns:
        List of prediction log entries
    """
    if not os.path.exists(log_file):
        return []
    
    entries = []
    with open(log_file, 'r') as f:
        lines = f.readlines()
        
        # Get last n_entries
        for line in lines[-n_entries:]:
            try:
                entry = json.loads(line.strip())
                entries.append(entry)
            except json.JSONDecodeError:
                continue
    
    return entries


def generate_model_report(model_results: Dict[str, Any], output_path: str = 'model_report.json'):
    """
    Generate comprehensive model report.
    
    Args:
        model_results: Model training and evaluation results
        output_path: Output path for report
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'model_info': {
            'type': 'Ensemble (Logistic Regression + XGBoost)',
            'version': '1.0.0',
            'training_date': datetime.now().isoformat()
        },
        'performance_metrics': model_results.get('evaluation_results', {}),
        'training_info': model_results.get('training_results', {}),
        'data_info': {
            'training_samples': model_results.get('training_results', {}).get('training_samples', 0),
            'features_count': model_results.get('training_results', {}).get('features_count', 0)
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Model report saved to {output_path}")


def check_system_requirements():
    """
    Check system requirements and dependencies.
    
    Returns:
        Dictionary of requirement checks
    """
    requirements = {
        'python_version': True,  # Assume Python 3.7+
        'required_packages': [],
        'disk_space': True,  # Assume sufficient disk space
        'memory': True  # Assume sufficient memory
    }
    
    # Check required packages
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'xgboost',
        'fastapi', 'uvicorn', 'dash', 'plotly'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    requirements['required_packages'] = missing_packages
    
    return requirements


if __name__ == "__main__":
    # Example usage
    config = Config()
    print("Default configuration:")
    print(json.dumps(config.config, indent=2))
    
    # Create sample config file
    create_sample_config('sample_config.json')
    
    # Check system requirements
    requirements = check_system_requirements()
    print(f"\nSystem requirements check:")
    print(f"Missing packages: {requirements['required_packages']}")
    
    # Create directories
    config.create_directories()



