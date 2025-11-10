#!/usr/bin/env python3
"""
Quick start script for the churn prediction system.
This script provides a simple way to get started with the system.
"""

import os
import sys
import subprocess

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        return False
    print(f"✓ Python {sys.version.split()[0]} detected")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def create_directories():
    """Create necessary directories."""
    directories = ['data', 'models', 'output', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def run_tests():
    """Run system tests."""
    print("Running system tests...")
    try:
        result = subprocess.run([sys.executable, "test_system.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ All tests passed")
            return True
        else:
            print("❌ Some tests failed")
            print(result.stdout)
            return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def main():
    """Main setup function."""
    print("="*60)
    print("CHURN PREDICTION SYSTEM - QUICK SETUP")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Create directories
    create_directories()
    
    # Run tests
    if not run_tests():
        print("\n⚠️  Tests failed, but you can still try running the demo")
    
    print("\n" + "="*60)
    print("SETUP COMPLETED!")
    print("="*60)
    print("Next steps:")
    print("1. Run the demo: python demo.py")
    print("2. Start the API: python src/api/app.py")
    print("3. Launch dashboard: python src/dashboard/app.py")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)



