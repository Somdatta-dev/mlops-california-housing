"""
Quick test to verify mlops environment is working correctly
"""

import sys
print(f"🐍 Python: {sys.version}")
print(f"📍 Executable: {sys.executable}")

try:
    import xgboost as xgb
    print(f"✅ XGBoost: {xgb.__version__}")
except ImportError as e:
    print(f"❌ XGBoost: {e}")

try:
    import mlflow
    print(f"✅ MLflow: {mlflow.__version__}")
except ImportError as e:
    print(f"❌ MLflow: {e}")

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"🔥 CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"❌ PyTorch: {e}")

try:
    import pandas as pd
    import numpy as np
    from sklearn.datasets import make_regression
    print(f"✅ Data Science Stack: pandas {pd.__version__}, numpy {np.__version__}")
except ImportError as e:
    print(f"❌ Data Science Stack: {e}")

# Test project imports
try:
    sys.path.append('.')
    from src.gpu_model_trainer import XGBoostConfig, ModelConfig
    print("✅ Project imports: Working")
    
    # Quick XGBoost test
    config = XGBoostConfig(n_estimators=100, max_depth=3)
    print(f"✅ XGBoost Config: {config.n_estimators} estimators, depth {config.max_depth}")
    
except Exception as e:
    print(f"❌ Project imports: {e}")

print("\n🎯 mlops environment is ready for XGBoost development!")