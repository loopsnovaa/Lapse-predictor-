"""
Ensemble model implementation using Logistic Regression and XGBoost.
Includes stacking ensemble method for improved performance.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import joblib
import logging
from typing import Dict, Any, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnEnsembleModel:
    """
    Ensemble model combining Logistic Regression and XGBoost for churn prediction.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the ensemble model.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config or {}
        self.models = {}
        self.ensemble_model = None
        self.feature_importance = None
        self.training_history = {}
        
        # Initialize base models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize the base models with optimized parameters."""
        
        # Logistic Regression with regularization
        self.models['logistic_regression'] = LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=1.0,
            penalty='l2',
            solver='liblinear',
            class_weight='balanced'
        )
        
        # XGBoost with optimized parameters
        self.models['xgboost'] = xgb.XGBClassifier(
            random_state=42,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=1.0,
            eval_metric='logloss',
            early_stopping_rounds=10
        )
        
        # Create stacking ensemble
        self.ensemble_model = StackingClassifier(
            estimators=[
                ('lr', self.models['logistic_regression']),
                ('xgb', self.models['xgboost'])
            ],
            final_estimator=LogisticRegression(random_state=42, max_iter=1000),
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        logger.info("Models initialized successfully")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """
        Train the ensemble model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting model training...")
        
        # Train individual models
        individual_scores = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            # Cross-validation for individual models
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='roc_auc',
                n_jobs=-1
            )
            
            individual_scores[name] = {
                'mean_cv_score': cv_scores.mean(),
                'std_cv_score': cv_scores.std(),
                'cv_scores': cv_scores
            }
            
            # Fit the model
            model.fit(X_train, y_train)
            logger.info(f"{name} training completed. CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train ensemble model
        logger.info("Training ensemble model...")
        self.ensemble_model.fit(X_train, y_train)
        
        # Get ensemble cross-validation scores
        ensemble_cv_scores = cross_val_score(
            self.ensemble_model, X_train, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=-1
        )
        
        individual_scores['ensemble'] = {
            'mean_cv_score': ensemble_cv_scores.mean(),
            'std_cv_score': ensemble_cv_scores.std(),
            'cv_scores': ensemble_cv_scores
        }
        
        logger.info(f"Ensemble training completed. CV AUC: {ensemble_cv_scores.mean():.4f} (+/- {ensemble_cv_scores.std() * 2:.4f})")
        
        # Store training history
        self.training_history = {
            'individual_scores': individual_scores,
            'training_samples': len(X_train),
            'features_count': X_train.shape[1]
        }
        
        return self.training_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the ensemble model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        return self.ensemble_model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities using the ensemble model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Prediction probabilities
        """
        return self.ensemble_model.predict_proba(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the model performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Evaluation metrics dictionary
        """
        logger.info("Evaluating model performance...")
        
        # Get predictions
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Individual model predictions for comparison
        individual_predictions = {}
        individual_probabilities = {}
        
        for name, model in self.models.items():
            individual_predictions[name] = model.predict(X_test)
            individual_probabilities[name] = model.predict_proba(X_test)[:, 1]
            individual_auc = roc_auc_score(y_test, individual_probabilities[name])
            logger.info(f"{name} test AUC: {individual_auc:.4f}")
        
        evaluation_results = {
            'ensemble_auc': auc_score,
            'ensemble_accuracy': class_report['accuracy'],
            'ensemble_precision': class_report['1']['precision'],
            'ensemble_recall': class_report['1']['recall'],
            'ensemble_f1': class_report['1']['f1-score'],
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'individual_predictions': individual_predictions,
            'individual_probabilities': individual_probabilities,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        logger.info(f"Ensemble test AUC: {auc_score:.4f}")
        logger.info(f"Ensemble test accuracy: {class_report['accuracy']:.4f}")
        
        return evaluation_results
    
    def get_feature_importance(self, feature_names: List[str] = None) -> pd.DataFrame:
        """
        Get feature importance from XGBoost model.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(self.models['xgboost'].n_features_in_)]
        
        importance_scores = self.models['xgboost'].feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = importance_df
        return importance_df
    
    def plot_feature_importance(self, top_n: int = 20, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to display
            figsize: Figure size
        """
        if self.feature_importance is None:
            logger.warning("Feature importance not calculated. Call get_feature_importance() first.")
            return
        
        plt.figure(figsize=figsize)
        top_features = self.feature_importance.head(top_n)
        
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importance (XGBoost)')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self, evaluation_results: Dict[str, Any], figsize: Tuple[int, int] = (10, 8)):
        """
        Plot ROC curves for ensemble and individual models.
        
        Args:
            evaluation_results: Results from evaluate() method
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Plot ensemble ROC curve
        fpr, tpr, _ = roc_curve(evaluation_results['y_test'], evaluation_results['y_pred_proba'])
        auc_score = evaluation_results['ensemble_auc']
        plt.plot(fpr, tpr, label=f'Ensemble (AUC = {auc_score:.4f})', linewidth=2)
        
        # Plot individual model ROC curves
        colors = ['red', 'green', 'blue', 'orange']
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
        plt.show()
    
    def plot_confusion_matrix(self, evaluation_results: Dict[str, Any], 
                            figsize: Tuple[int, int] = (8, 6)):
        """
        Plot confusion matrix.
        
        Args:
            evaluation_results: Results from evaluate() method
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        conf_matrix = evaluation_results['confusion_matrix']
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        plt.title('Confusion Matrix - Ensemble Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    def save_model(self, file_path: str):
        """
        Save the trained model.
        
        Args:
            file_path: Path to save the model
        """
        model_data = {
            'ensemble_model': self.ensemble_model,
            'models': self.models,
            'feature_importance': self.feature_importance,
            'training_history': self.training_history,
            'config': self.config
        }
        joblib.dump(model_data, file_path)
        logger.info(f"Model saved to {file_path}")
    
    def load_model(self, file_path: str):
        """
        Load a trained model.
        
        Args:
            file_path: Path to load the model from
        """
        model_data = joblib.load(file_path)
        self.ensemble_model = model_data['ensemble_model']
        self.models = model_data['models']
        self.feature_importance = model_data['feature_importance']
        self.training_history = model_data['training_history']
        self.config = model_data['config']
        logger.info(f"Model loaded from {file_path}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model performance and configuration.
        
        Returns:
            Model summary dictionary
        """
        summary = {
            'model_type': 'Ensemble (Logistic Regression + XGBoost)',
            'base_models': list(self.models.keys()),
            'ensemble_method': 'Stacking',
            'config': self.config,
            'training_history': self.training_history
        }
        
        if self.feature_importance is not None:
            summary['top_features'] = self.feature_importance.head(10).to_dict('records')
        
        return summary


def hyperparameter_tuning(X_train: np.ndarray, y_train: np.ndarray, 
                         model_type: str = 'xgboost') -> Dict[str, Any]:
    """
    Perform hyperparameter tuning for the specified model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_type: Type of model to tune ('logistic_regression' or 'xgboost')
        
    Returns:
        Best parameters dictionary
    """
    from sklearn.model_selection import GridSearchCV
    
    logger.info(f"Starting hyperparameter tuning for {model_type}...")
    
    if model_type == 'logistic_regression':
        param_grid = {
            'C': [0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'class_weight': ['balanced', None]
        }
        model = LogisticRegression(random_state=42, max_iter=1000)
        
    elif model_type == 'xgboost':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1.0],
            'reg_lambda': [0, 0.1, 1.0]
        }
        model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='roc_auc', 
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Best parameters for {model_type}: {grid_search.best_params_}")
    logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_estimator': grid_search.best_estimator_
    }


if __name__ == "__main__":
    # Example usage
    from src.data.preprocessing import DataPreprocessor, create_sample_data
    
    # Create sample data
    sample_data = create_sample_data(1000)
    preprocessor = DataPreprocessor()
    prepared_data = preprocessor.prepare_data(sample_data, 'churn')
    
    # Initialize and train ensemble model
    ensemble_model = ChurnEnsembleModel()
    
    # Train the model
    training_results = ensemble_model.train(
        prepared_data['X_train'], 
        prepared_data['y_train']
    )
    
    # Evaluate the model
    evaluation_results = ensemble_model.evaluate(
        prepared_data['X_test'], 
        prepared_data['y_test']
    )
    
    # Get feature importance
    feature_importance = ensemble_model.get_feature_importance(prepared_data['feature_names'])
    print("Top 10 Features:")
    print(feature_importance.head(10))
    
    # Print evaluation results
    print(f"\nEnsemble Model Performance:")
    print(f"AUC: {evaluation_results['ensemble_auc']:.4f}")
    print(f"Accuracy: {evaluation_results['ensemble_accuracy']:.4f}")
    print(f"Precision: {evaluation_results['ensemble_precision']:.4f}")
    print(f"Recall: {evaluation_results['ensemble_recall']:.4f}")
    print(f"F1-Score: {evaluation_results['ensemble_f1']:.4f}")
    
    # Get model summary
    summary = ensemble_model.get_model_summary()
    print(f"\nModel Summary:")
    print(f"Model Type: {summary['model_type']}")
    print(f"Base Models: {summary['base_models']}")
    print(f"Training Samples: {summary['training_history']['training_samples']}")
    print(f"Features Count: {summary['training_history']['features_count']}")



