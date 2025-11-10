"""
Training pipeline for churn prediction ensemble model.
Includes hyperparameter tuning, cross-validation, and model evaluation.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from src.data.preprocessing import DataPreprocessor
from src.models.ensemble import ChurnEnsembleModel, hyperparameter_tuning

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Complete training pipeline for churn prediction model.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the training pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.preprocessor = DataPreprocessor(self.config.get('preprocessing', {}))
        self.model = None
        self.training_results = {}
        self.evaluation_results = {}
        
        # Setup MLflow
        self._setup_mlflow()
        
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
                'target_column': 'churn'
            },
            'preprocessing': {
                'apply_smote_enn': True,
                'feature_engineering': True
            },
            'model': {
                'enable_hyperparameter_tuning': False,
                'cv_folds': 5
            },
            'mlflow': {
                'experiment_name': 'churn_prediction',
                'tracking_uri': 'file:./mlruns'
            }
        }
    
    def _setup_mlflow(self):
        """Setup MLflow tracking."""
        try:
            mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
            mlflow.set_experiment(self.config['mlflow']['experiment_name'])
            logger.info("MLflow setup completed")
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}")
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load data from file.
        
        Args:
            data_path: Path to the data file
            
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading data from {data_path}")
        
        if data_path.endswith('.csv'):
            data = pd.read_csv(data_path)
        elif data_path.endswith('.parquet'):
            data = pd.read_parquet(data_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Parquet.")
        
        logger.info(f"Data loaded successfully. Shape: {data.shape}")
        return data
    
    def prepare_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Prepare data using the preprocessor.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Prepared data dictionary
        """
        logger.info("Preparing data...")
        
        target_column = self.config['data']['target_column']
        test_size = self.config['data']['test_size']
        random_state = self.config['data']['random_state']
        
        prepared_data = self.preprocessor.prepare_data(
            data, target_column, test_size, random_state
        )
        
        logger.info("Data preparation completed")
        return prepared_data
    
    def train_model(self, prepared_data: Dict[str, Any], 
                   tune_hyperparameters: bool = False) -> Dict[str, Any]:
        """
        Train the ensemble model.
        
        Args:
            prepared_data: Prepared data dictionary
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting model training...")
        
        # Initialize model
        self.model = ChurnEnsembleModel(self.config.get('model', {}))
        
        # Hyperparameter tuning if enabled
        if tune_hyperparameters and self.config['model']['enable_hyperparameter_tuning']:
            logger.info("Performing hyperparameter tuning...")
            
            # Tune XGBoost parameters
            xgb_params = hyperparameter_tuning(
                prepared_data['X_train'], 
                prepared_data['y_train'], 
                'xgboost'
            )
            
            # Update model with best parameters
            self.model.models['xgboost'].set_params(**xgb_params['best_params'])
            
            logger.info(f"Best XGBoost parameters: {xgb_params['best_params']}")
        
        # Train the model
        training_results = self.model.train(
            prepared_data['X_train'], 
            prepared_data['y_train']
        )
        
        self.training_results = training_results
        logger.info("Model training completed")
        
        return training_results
    
    def evaluate_model(self, prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the trained model.
        
        Args:
            prepared_data: Prepared data dictionary
            
        Returns:
            Evaluation results dictionary
        """
        logger.info("Evaluating model...")
        
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        evaluation_results = self.model.evaluate(
            prepared_data['X_test'], 
            prepared_data['y_test']
        )
        
        self.evaluation_results = evaluation_results
        logger.info("Model evaluation completed")
        
        return evaluation_results
    
    def log_to_mlflow(self, prepared_data: Dict[str, Any], 
                     training_results: Dict[str, Any], 
                     evaluation_results: Dict[str, Any]):
        """
        Log training results to MLflow.
        
        Args:
            prepared_data: Prepared data dictionary
            training_results: Training results dictionary
            evaluation_results: Evaluation results dictionary
        """
        try:
            with mlflow.start_run():
                # Log parameters
                mlflow.log_params({
                    'test_size': self.config['data']['test_size'],
                    'random_state': self.config['data']['random_state'],
                    'cv_folds': self.config['model']['cv_folds'],
                    'enable_smote_enn': self.config['preprocessing']['apply_smote_enn'],
                    'feature_engineering': self.config['preprocessing']['feature_engineering']
                })
                
                # Log metrics
                mlflow.log_metrics({
                    'ensemble_auc': evaluation_results['ensemble_auc'],
                    'ensemble_accuracy': evaluation_results['ensemble_accuracy'],
                    'ensemble_precision': evaluation_results['ensemble_precision'],
                    'ensemble_recall': evaluation_results['ensemble_recall'],
                    'ensemble_f1': evaluation_results['ensemble_f1'],
                    'training_samples': training_results['training_samples'],
                    'features_count': training_results['features_count']
                })
                
                # Log individual model scores
                for model_name, scores in training_results['individual_scores'].items():
                    mlflow.log_metrics({
                        f'{model_name}_cv_auc_mean': scores['mean_cv_score'],
                        f'{model_name}_cv_auc_std': scores['std_cv_score']
                    })
                
                # Log model
                mlflow.sklearn.log_model(
                    self.model.ensemble_model, 
                    "ensemble_model",
                    registered_model_name="churn_prediction_ensemble"
                )
                
                # Log preprocessor
                mlflow.sklearn.log_model(
                    self.preprocessor, 
                    "preprocessor",
                    registered_model_name="churn_preprocessor"
                )
                
                logger.info("Results logged to MLflow successfully")
                
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")
    
    def save_artifacts(self, output_dir: str = "models"):
        """
        Save trained model and preprocessor artifacts.
        
        Args:
            output_dir: Directory to save artifacts
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = os.path.join(output_dir, f"ensemble_model_{timestamp}.joblib")
        self.model.save_model(model_path)
        
        # Save preprocessor
        preprocessor_path = os.path.join(output_dir, f"preprocessor_{timestamp}.joblib")
        self.preprocessor.save_preprocessor(preprocessor_path)
        
        # Save training results
        results_path = os.path.join(output_dir, f"training_results_{timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump(self.training_results, f, indent=2, default=str)
        
        # Save evaluation results
        eval_path = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
        with open(eval_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2, default=str)
        
        # Save configuration
        config_path = os.path.join(output_dir, f"config_{timestamp}.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Artifacts saved to {output_dir}")
        
        return {
            'model_path': model_path,
            'preprocessor_path': preprocessor_path,
            'results_path': results_path,
            'eval_path': eval_path,
            'config_path': config_path
        }
    
    def generate_report(self, output_path: str = "training_report.html"):
        """
        Generate a comprehensive training report.
        
        Args:
            output_path: Path to save the HTML report
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import plotly.offline as pyo
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Model Performance Comparison', 'Confusion Matrix',
                              'Feature Importance', 'ROC Curves'),
                specs=[[{"type": "bar"}, {"type": "heatmap"}],
                       [{"type": "bar"}, {"type": "scatter"}]]
            )
            
            # Model performance comparison
            models = list(self.training_results['individual_scores'].keys())
            cv_scores = [self.training_results['individual_scores'][m]['mean_cv_score'] for m in models]
            
            fig.add_trace(
                go.Bar(x=models, y=cv_scores, name='CV AUC Score'),
                row=1, col=1
            )
            
            # Confusion matrix
            conf_matrix = self.evaluation_results['confusion_matrix']
            fig.add_trace(
                go.Heatmap(z=conf_matrix, 
                          x=['No Churn', 'Churn'],
                          y=['No Churn', 'Churn'],
                          showscale=False),
                row=1, col=2
            )
            
            # Feature importance (top 10)
            if self.model.feature_importance is not None:
                top_features = self.model.feature_importance.head(10)
                fig.add_trace(
                    go.Bar(x=top_features['importance'], 
                          y=top_features['feature'],
                          orientation='h',
                          name='Feature Importance'),
                    row=2, col=1
                )
            
            # ROC curve
            y_test = self.evaluation_results['y_test']
            y_pred_proba = self.evaluation_results['y_pred_proba']
            
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            
            fig.add_trace(
                go.Scatter(x=fpr, y=tpr, 
                          mode='lines',
                          name=f'ROC Curve (AUC = {self.evaluation_results["ensemble_auc"]:.4f})'),
                row=2, col=2
            )
            
            # Add diagonal line for ROC
            fig.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], 
                          mode='lines',
                          line=dict(dash='dash'),
                          name='Random Classifier'),
                row=2, col=2
            )
            
            fig.update_layout(
                title_text="Churn Prediction Model Training Report",
                showlegend=True,
                height=800
            )
            
            # Save as HTML
            pyo.plot(fig, filename=output_path, auto_open=False)
            logger.info(f"Training report saved to {output_path}")
            
        except ImportError:
            logger.warning("Plotly not available. Skipping report generation.")
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
    
    def run_full_pipeline(self, data_path: str, output_dir: str = "models",
                         tune_hyperparameters: bool = False) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Args:
            data_path: Path to the data file
            output_dir: Directory to save outputs
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Pipeline results dictionary
        """
        logger.info("Starting full training pipeline...")
        
        # Load data
        data = self.load_data(data_path)
        
        # Prepare data
        prepared_data = self.prepare_data(data)
        
        # Train model
        training_results = self.train_model(prepared_data, tune_hyperparameters)
        
        # Evaluate model
        evaluation_results = self.evaluate_model(prepared_data)
        
        # Log to MLflow
        self.log_to_mlflow(prepared_data, training_results, evaluation_results)
        
        # Save artifacts
        artifact_paths = self.save_artifacts(output_dir)
        
        # Generate report
        report_path = os.path.join(output_dir, "training_report.html")
        self.generate_report(report_path)
        
        logger.info("Full training pipeline completed successfully")
        
        return {
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'artifact_paths': artifact_paths,
            'report_path': report_path,
            'model': self.model,
            'preprocessor': self.preprocessor
        }


def main():
    """Main function to run the training pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train churn prediction model')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to the training data file')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Output directory for artifacts')
    parser.add_argument('--tune_hyperparameters', action='store_true',
                       help='Enable hyperparameter tuning')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = TrainingPipeline(args.config_path)
    
    # Run full pipeline
    results = pipeline.run_full_pipeline(
        args.data_path,
        args.output_dir,
        args.tune_hyperparameters
    )
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING PIPELINE SUMMARY")
    print("="*50)
    print(f"Training samples: {results['training_results']['training_samples']}")
    print(f"Features count: {results['training_results']['features_count']}")
    print(f"Ensemble AUC: {results['evaluation_results']['ensemble_auc']:.4f}")
    print(f"Ensemble Accuracy: {results['evaluation_results']['ensemble_accuracy']:.4f}")
    print(f"Ensemble Precision: {results['evaluation_results']['ensemble_precision']:.4f}")
    print(f"Ensemble Recall: {results['evaluation_results']['ensemble_recall']:.4f}")
    print(f"Ensemble F1-Score: {results['evaluation_results']['ensemble_f1']:.4f}")
    print(f"\nArtifacts saved to: {args.output_dir}")
    print(f"Training report: {results['report_path']}")


if __name__ == "__main__":
    main()



