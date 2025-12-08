import sys
import os

# Add 'src' to python path so we can import modules
sys.path.append('src')

print("--- STARTING TEST SYSTEM ---")  # debug print

try:
    from data.preprocessing import DataPreprocessor, create_sample_data
    from models.ensemble import ChurnEnsembleModel
    print("✓ Imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def main():
    print("="*50)
    print("CHURN PREDICTION SYSTEM - QUICK TESTS")
    print("="*50)
    
    # 1. Test Data Gen
    print("\n1. Generating Data...")
    try:
        data = create_sample_data(100)
        print(f"   Created {len(data)} samples.")
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return

    # 2. Test Preprocessing
    print("\n2. Testing Preprocessing...")
    try:
        preprocessor = DataPreprocessor()
        prepared_data = preprocessor.prepare_data(data, 'policy_lapse')
        X_train = prepared_data['X_train']
        print(f"   Preprocessing successful. Training shape: {X_train.shape}")
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        # Print the error traceback to see exactly where it died
        import traceback
        traceback.print_exc()
        return

    # 3. Test Training
    print("\n3. Testing Model Training...")
    try:
        model = ChurnEnsembleModel()
        results = model.train(prepared_data['X_train'], prepared_data['y_train'])
        print(f"   Training successful. Models: {list(results['individual_scores'].keys())}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return

    print("\n" + "="*50)
    print("ALL SYSTEMS GO! ✓")
    print("="*50)

if __name__ == "__main__":
    main()