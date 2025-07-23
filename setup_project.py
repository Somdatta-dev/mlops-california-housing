#!/usr/bin/env python3
"""
Project setup script for MLOps California Housing project.
Run this after cloning the repository to set up the environment.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"   Error: {e.stderr.strip()}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required!")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def setup_project():
    """Set up the MLOps project."""
    print("🚀 MLOps California Housing Project Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        return False
    
    # Create remote storage directory
    remote_dir = Path("../dvc_remote_storage")
    if not remote_dir.exists():
        print("🔄 Creating DVC remote storage directory...")
        remote_dir.mkdir(parents=True, exist_ok=True)
        print("✅ DVC remote storage directory created!")
    else:
        print("✅ DVC remote storage directory already exists!")
    
    # Check DVC status
    if not run_command("dvc status", "Checking DVC status"):
        return False
    
    # Pull data from DVC remote
    if not run_command("dvc pull", "Pulling data from DVC remote"):
        return False
    
    # Verify data files exist
    data_files = [
        "data/raw/california_housing_features.csv",
        "data/raw/california_housing_targets.csv",
        "data/raw/dataset_metadata.json",
        "data/raw/validation_report.json"
    ]
    
    print("🔍 Verifying data files...")
    all_files_exist = True
    for file_path in data_files:
        if Path(file_path).exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - MISSING!")
            all_files_exist = False
    
    if not all_files_exist:
        print("❌ Some data files are missing!")
        return False
    
    # Test data loading
    print("🧪 Testing data loading...")
    try:
        from src.data_loader import CaliforniaHousingDataLoader
        loader = CaliforniaHousingDataLoader()
        features, targets = loader.load_dataset()
        print(f"✅ Successfully loaded {len(features):,} samples with {len(features.columns)} features")
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False
    
    # Success!
    print("\n" + "🎉" * 20)
    print("🎉 PROJECT SETUP COMPLETED SUCCESSFULLY! 🎉")
    print("🎉" * 20)
    print("\n📋 What's ready:")
    print("   ✅ All dependencies installed")
    print("   ✅ DVC configured and data pulled")
    print("   ✅ California Housing dataset loaded")
    print("   ✅ Data validation completed")
    print("\n🚀 You can now:")
    print("   • Run data validation: python src/data_validation.py")
    print("   • Start model training: python src/models/train_models.py")
    print("   • Launch API server: uvicorn src.api.main:app --reload")
    print("   • View MLflow UI: mlflow ui")
    
    return True

def main():
    """Main function."""
    if not setup_project():
        print("\n❌ Setup failed! Please check the errors above.")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())