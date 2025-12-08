import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChurnEnsembleModel:
    def __init__(self):
        """Initialize the ensemble with models."""
        self.models = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            # CRITICAL FIX: Removed 'early_stopping_rounds' to prevent Cross-Validation crash
            'xgboost': XGBClassifier(
                n_estimators=100, 
                learning_rate=0.1, 
                max_depth=5, 
                eval_metric='logloss',
                random_state=42
            )
        }
        self.trained_models = {}
        self.feature_names = None
        logger.info("Models initialized successfully")

    def train(self, X_train, y_train):
        """
        Train models using Cross-Validation to verify stability, then fit on full data.
        """
        logger.info("Starting model training...")
        
        # Save feature names
        if hasattr(X_train, 'columns'):
            self.feature_names = X_train.columns.tolist()
        elif isinstance(X_train, np.ndarray):
             # Create generic names if numpy array
             self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

        training_results = {'individual_scores': {}}
        
        # Define CV strategy
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            try:
                # 1. Cross-Validation Score (Check for stability)
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=cv, scoring='roc_auc', n_jobs=-1
                )
                
                mean_score = cv_scores.mean()
                std_score = cv_scores.std()
                
                logger.info(f"{name} training completed. CV AUC: {mean_score:.4f} (+/- {std_score:.4f})")
                
                training_results['individual_scores'][name] = {
                    'mean_cv_score': mean_score,
                    'std_cv_score': std_score
                }
                
                # 2. Final Fit on All Data
                model.fit(X_train, y_train)
                self.trained_models[name] = model
                
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                # We catch the error so one failed model doesn't kill the whole pipeline
                continue
                
        return training_results

    def predict_proba(self, X):
        """
        Soft Voting: Average the probabilities of all models.
        """
        if not self.trained_models:
            raise ValueError("Models not trained yet!")
            
        # Collect probabilities from each model
        all_probs = []
        
        for name, model in self.trained_models.items():
            try:
                # Get probability for class 1 (Lapse)
                probs = model.predict_proba(X)[:, 1]
                all_probs.append(probs)
            except Exception as e:
                logger.warning(f"Model {name} failed prediction: {e}")
            
        if not all_probs:
            raise ValueError("No models successfully returned probabilities.")

        # Average them
        avg_probs = np.mean(all_probs, axis=0)
        
        # Return in sklearn format (n_samples, 2) -> [prob_0, prob_1]
        return np.vstack((1 - avg_probs, avg_probs)).T

    def predict(self, X, threshold=0.5):
        """
        Predict class based on averaged probability.
        """
        probs = self.predict_proba(X)[:, 1]
        return (probs > threshold).astype(int)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the Ensemble Performance.
        """
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)[:, 1]
        
        return {
            'ensemble_accuracy': accuracy_score(y_test, y_pred),
            'ensemble_precision': precision_score(y_test, y_pred, zero_division=0),
            'ensemble_recall': recall_score(y_test, y_pred, zero_division=0),
            'ensemble_f1': f1_score(y_test, y_pred, zero_division=0),
            'ensemble_auc': roc_auc_score(y_test, y_prob)
        }

    def get_feature_importance(self, feature_names=None):
        """
        Average feature importance across Tree-based models.
        """
        if feature_names is None:
            feature_names = self.feature_names
            
        if feature_names is None:
            return pd.DataFrame()

        # Dataframe to store importances
        fi_df = pd.DataFrame({'feature': feature_names})
        
        valid_models = 0
        fi_df['importance'] = 0.0
        
        for name, model in self.trained_models.items():
            if hasattr(model, 'feature_importances_'):
                # Ensure shapes match before adding
                if len(model.feature_importances_) == len(fi_df):
                    fi_df['importance'] += model.feature_importances_
                    valid_models += 1
                
        if valid_models > 0:
            fi_df['importance'] /= valid_models
            
        return fi_df.sort_values(by='importance', ascending=False)

    def save_model(self, path):
        joblib.dump(self, path)
        
    def load_model(self, path):
        loaded = joblib.load(path)
        self.__dict__.update(loaded.__dict__)