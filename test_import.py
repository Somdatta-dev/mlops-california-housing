#!/usr/bin/env python3

print("Starting import test...")

try:
    print("Importing basic modules...")
    import torch
    import numpy as np
    from dataclasses import dataclass
    from datetime import datetime
    print("✓ Basic modules imported")
    
    print("Testing dataclass creation...")
    @dataclass
    class TestMetrics:
        value: float
        timestamp: datetime
    
    test_obj = TestMetrics(value=1.0, timestamp=datetime.now())
    print(f"✓ Dataclass created: {test_obj}")
    
    print("Importing from gpu_model_trainer...")
    import src.gpu_model_trainer as gmt
    print(f"✓ Module imported, attributes: {[attr for attr in dir(gmt) if not attr.startswith('_')]}")
    
    print("Testing specific imports...")
    from src.gpu_model_trainer import GPUMemoryManager
    print("✓ GPUMemoryManager imported")
    
    from src.gpu_model_trainer import GPUMetrics
    print("✓ GPUMetrics imported")
    
    print("All imports successful!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()