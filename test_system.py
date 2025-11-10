#!/usr/bin/env python3
"""
Simple test script for the churn prediction system.
Tests basic functionality without extensive output.
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from data.preprocessing import DataPreprocessor, create_sample_data
from models.ensemble import ChurnEnsembleModel

def test_data_preprocessing():
    """Test data preprocessing functionality."""
    print("Testing data preprocessing...")
    
    # Create sample data
    data = create_sample_data(100)
    preprocessor = DataPreprocessor()
    prepared_data = preprocessor.prepare_data(data, 'policy_lapse')
    
    assert prepared_data['X_train'].shape[0] > 0, "Training data should not be empty"
    assert prepared_data['X_test'].shape[0] > 0, "Test data should not be empty"
    assert len(prepared_data['feature_names']) > 0, "Should have feature names"
    
    print("✓ Data preprocessing test passed")
    return prepared_data

def test_model_training(prepared_data):
    """Test model training functionality."""
    print("Testing model training...")
    
    model = ChurnEnsembleModel()
    training_results = model.train(
        prepared_data['X_train'], 
        prepared_data['y_train']
    )
    
    assert 'individual_scores' in training_results, "Should have individual scores"
    assert 'training_samples' in training_results, "Should have training samples count"
    
    print("✓ Model training test passed")
    return model

def test_model_evaluation(model, prepared_data):
    """Test model evaluation functionality."""
    print("Testing model evaluation...")
    
    evaluation_results = model.evaluate(
        prepared_data['X_test'], 
        prepared_data['y_test']
    )
    
    assert 'ensemble_auc' in evaluation_results, "Should have ensemble AUC"
    assert 'ensemble_accuracy' in evaluation_results, "Should have ensemble accuracy"
    assert evaluation_results['ensemble_auc'] > 0.5, "AUC should be better than random"
    
    print("✓ Model evaluation test passed")
    return evaluation_results

def test_predictions(model, prepared_data):
    """Test prediction functionality."""
    print("Testing predictions...")
    
    # Test single prediction
    X_test = prepared_data['X_test'][:1]  # Take first sample
    prediction = model.predict(X_test)
    probability = model.predict_proba(X_test)
    
    assert len(prediction) == 1, "Should return single prediction"
    assert len(probability) == 1, "Should return single probability"
    assert prediction[0] in [0, 1], "Prediction should be 0 or 1"
    assert 0 <= probability[0][1] <= 1, "Probability should be between 0 and 1"
    
    print("✓ Predictions test passed")

def test_feature_importance(model, prepared_data):
    """Test feature importance functionality."""
    print("Testing feature importance...")
    
    feature_importance = model.get_feature_importance(prepared_data['feature_names'])
    
    assert len(feature_importance) > 0, "Should have feature importance"
    assert 'feature' in feature_importance.columns, "Should have feature column"
    assert 'importance' in feature_importance.columns, "Should have importance column"
    
    print("✓ Feature importance test passed")

def main():
    """Run all tests."""
    print("="*50)
    print("CHURN PREDICTION SYSTEM - QUICK TESTS")
    print("="*50)
    
    try:
        # Test data preprocessing
        prepared_data = test_data_preprocessing()
        
        # Test model training
        model = test_model_training(prepared_data)
        
        # Test model evaluation
        evaluation_results = test_model_evaluation(model, prepared_data)
        
        # Test predictions
        test_predictions(model, prepared_data)
        
        # Test feature importance
        test_feature_importance(model, prepared_data)
        
        print("\n" + "="*50)
        print("ALL TESTS PASSED! ✓")
        print("="*50)
        print(f"Final AUC Score: {evaluation_results['ensemble_auc']:.4f}")
        print(f"Final Accuracy: {evaluation_results['ensemble_accuracy']:.4f}")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
