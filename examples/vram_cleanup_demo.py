"""
VRAM Cleanup Demonstration

This script demonstrates the GPU memory management and VRAM cleanup functionality
for the MLOps platform GPU training infrastructure.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import gc
import time
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPUMemoryManager:
    """GPU memory management utilities for VRAM cleanup and monitoring."""
    
    @staticmethod
    def clear_gpu_memory():
        """Comprehensive GPU memory cleanup."""
        if torch.cuda.is_available():
            try:
                # Clear PyTorch cache
                torch.cuda.empty_cache()
                
                # Force garbage collection
                gc.collect()
                
                # Synchronize CUDA operations
                torch.cuda.synchronize()
                
                logger.info("GPU memory cleared successfully")
                
            except Exception as e:
                logger.warning(f"Failed to clear GPU memory: {e}")
    
    @staticmethod
    def get_gpu_memory_info():
        """Get current GPU memory usage information."""
        if not torch.cuda.is_available():
            return {"available": False}
        
        try:
            # Get memory info from PyTorch
            allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved() / (1024**3)   # GB
            max_allocated = torch.cuda.max_memory_allocated() / (1024**3)  # GB
            max_reserved = torch.cuda.max_memory_reserved() / (1024**3)    # GB
            
            return {
                "available": True,
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "max_allocated_gb": max_allocated,
                "max_reserved_gb": max_reserved,
                "free_gb": reserved - allocated
            }
            
        except Exception as e:
            logger.error(f"Failed to get GPU memory info: {e}")
            return {"available": False, "error": str(e)}
    
    @staticmethod
    def reset_peak_memory_stats():
        """Reset peak memory statistics."""
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
                logger.info("GPU peak memory stats reset")
            except Exception as e:
                logger.warning(f"Failed to reset peak memory stats: {e}")


def create_large_model_and_data():
    """Create a large model and data to consume GPU memory."""
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    
    # Create a large neural network
    model = nn.Sequential(
        nn.Linear(1000, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1000),
        nn.ReLU(),
        nn.Linear(1000, 1)
    ).to(device)
    
    # Create large tensors
    batch_size = 1024
    input_size = 1000
    
    data = torch.randn(batch_size, input_size).to(device)
    targets = torch.randn(batch_size, 1).to(device)
    
    return model, data, targets, device


def demonstrate_memory_leak():
    """Demonstrate memory consumption without cleanup."""
    print("\n" + "="*60)
    print("DEMONSTRATING MEMORY CONSUMPTION WITHOUT CLEANUP")
    print("="*60)
    
    models = []
    data_tensors = []
    
    for i in range(3):
        print(f"\nIteration {i+1}:")
        
        # Get initial memory
        initial_memory = GPUMemoryManager.get_gpu_memory_info()
        if initial_memory.get('available'):
            print(f"  Before: {initial_memory['allocated_gb']:.3f} GB allocated, {initial_memory['reserved_gb']:.3f} GB reserved")
        
        # Create model and data (this will consume GPU memory)
        model, data, targets, device = create_large_model_and_data()
        
        # Store references (preventing garbage collection)
        models.append(model)
        data_tensors.append((data, targets))
        
        # Get memory after creation
        after_memory = GPUMemoryManager.get_gpu_memory_info()
        if after_memory.get('available'):
            print(f"  After:  {after_memory['allocated_gb']:.3f} GB allocated, {after_memory['reserved_gb']:.3f} GB reserved")
            memory_increase = after_memory['allocated_gb'] - initial_memory.get('allocated_gb', 0)
            print(f"  Memory increase: {memory_increase:.3f} GB")
    
    final_memory = GPUMemoryManager.get_gpu_memory_info()
    if final_memory.get('available'):
        print(f"\nFinal memory usage: {final_memory['allocated_gb']:.3f} GB allocated")
        print("⚠️  Memory not cleaned up - references still held!")
    
    return models, data_tensors


def demonstrate_proper_cleanup():
    """Demonstrate proper memory cleanup."""
    print("\n" + "="*60)
    print("DEMONSTRATING PROPER MEMORY CLEANUP")
    print("="*60)
    
    for i in range(3):
        print(f"\nIteration {i+1}:")
        
        # Get initial memory
        initial_memory = GPUMemoryManager.get_gpu_memory_info()
        if initial_memory.get('available'):
            print(f"  Before: {initial_memory['allocated_gb']:.3f} GB allocated, {initial_memory['reserved_gb']:.3f} GB reserved")
        
        # Create model and data
        model, data, targets, device = create_large_model_and_data()
        
        # Get memory after creation
        after_memory = GPUMemoryManager.get_gpu_memory_info()
        if after_memory.get('available'):
            print(f"  After:  {after_memory['allocated_gb']:.3f} GB allocated, {after_memory['reserved_gb']:.3f} GB reserved")
            memory_increase = after_memory['allocated_gb'] - initial_memory.get('allocated_gb', 0)
            print(f"  Memory increase: {memory_increase:.3f} GB")
        
        # Proper cleanup
        print("  Cleaning up...")
        
        # Move to CPU and delete references
        if torch.cuda.is_available():
            model = model.cpu()
            data = data.cpu()
            targets = targets.cpu()
        
        del model, data, targets
        
        # Comprehensive cleanup
        GPUMemoryManager.clear_gpu_memory()
        
        # Get memory after cleanup
        cleanup_memory = GPUMemoryManager.get_gpu_memory_info()
        if cleanup_memory.get('available'):
            print(f"  Cleaned: {cleanup_memory['allocated_gb']:.3f} GB allocated, {cleanup_memory['reserved_gb']:.3f} GB reserved")
            memory_freed = after_memory.get('allocated_gb', 0) - cleanup_memory.get('allocated_gb', 0)
            print(f"  Memory freed: {memory_freed:.3f} GB")
            
            if memory_freed > 0.01:  # More than 10MB
                print("  ✅ Cleanup successful!")
            else:
                print("  ℹ️  Minimal cleanup (may be expected)")


def demonstrate_comprehensive_cleanup(models, data_tensors):
    """Demonstrate comprehensive cleanup of accumulated memory."""
    print("\n" + "="*60)
    print("DEMONSTRATING COMPREHENSIVE CLEANUP")
    print("="*60)
    
    # Get memory before cleanup
    before_memory = GPUMemoryManager.get_gpu_memory_info()
    if before_memory.get('available'):
        print(f"Before comprehensive cleanup: {before_memory['allocated_gb']:.3f} GB allocated")
    
    # Clean up all models
    print("Cleaning up models...")
    for model in models:
        if torch.cuda.is_available() and hasattr(model, 'cpu'):
            model.cpu()
    models.clear()
    
    # Clean up all data tensors
    print("Cleaning up data tensors...")
    for data, targets in data_tensors:
        if torch.cuda.is_available():
            data = data.cpu()
            targets = targets.cpu()
        del data, targets
    data_tensors.clear()
    
    # Multiple cleanup passes
    print("Performing comprehensive cleanup...")
    for i in range(3):
        gc.collect()
        GPUMemoryManager.clear_gpu_memory()
        time.sleep(0.1)
    
    # Get memory after cleanup
    after_memory = GPUMemoryManager.get_gpu_memory_info()
    if after_memory.get('available'):
        print(f"After comprehensive cleanup: {after_memory['allocated_gb']:.3f} GB allocated")
        memory_freed = before_memory.get('allocated_gb', 0) - after_memory.get('allocated_gb', 0)
        print(f"Total memory freed: {memory_freed:.3f} GB ({memory_freed * 1024:.1f} MB)")
        
        if memory_freed > 0.1:  # More than 100MB
            print("🎉 Comprehensive cleanup very successful!")
        elif memory_freed > 0.01:  # More than 10MB
            print("✅ Comprehensive cleanup successful!")
        else:
            print("ℹ️  Minimal cleanup (GPU may not have been heavily used)")


def main():
    """Main demonstration function."""
    print("GPU Memory Management and VRAM Cleanup Demonstration")
    print("=" * 60)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")
        
        # Reset memory stats for clean start
        GPUMemoryManager.reset_peak_memory_stats()
        
        initial_memory = GPUMemoryManager.get_gpu_memory_info()
        print(f"Initial GPU memory: {initial_memory['allocated_gb']:.3f} GB allocated")
    else:
        print("GPU not available - demonstration will use CPU")
        print("(VRAM cleanup features will be simulated)")
    
    # Demonstrate memory leak scenario
    models, data_tensors = demonstrate_memory_leak()
    
    # Demonstrate proper cleanup
    demonstrate_proper_cleanup()
    
    # Demonstrate comprehensive cleanup of accumulated memory
    demonstrate_comprehensive_cleanup(models, data_tensors)
    
    # Final memory report
    print("\n" + "="*60)
    print("FINAL MEMORY REPORT")
    print("="*60)
    
    final_memory = GPUMemoryManager.get_gpu_memory_info()
    if final_memory.get('available'):
        print(f"Final allocated memory: {final_memory['allocated_gb']:.3f} GB")
        print(f"Final reserved memory: {final_memory['reserved_gb']:.3f} GB")
        print(f"Peak allocated memory: {final_memory['max_allocated_gb']:.3f} GB")
        print(f"Peak reserved memory: {final_memory['max_reserved_gb']:.3f} GB")
        
        if final_memory['allocated_gb'] < 0.1:  # Less than 100MB
            print("🎉 Excellent! GPU memory successfully cleaned up!")
        elif final_memory['allocated_gb'] < 0.5:  # Less than 500MB
            print("✅ Good! Most GPU memory cleaned up!")
        else:
            print("⚠️  Some GPU memory still allocated - may need additional cleanup")
    else:
        print("GPU memory monitoring not available")
    
    print("\n" + "="*60)
    print("KEY TAKEAWAYS FOR VRAM CLEANUP:")
    print("• Always move models to CPU before deletion: model.cpu()")
    print("• Delete tensor references explicitly: del tensor")
    print("• Call torch.cuda.empty_cache() to free cached memory")
    print("• Use gc.collect() to force garbage collection")
    print("• Call torch.cuda.synchronize() to ensure operations complete")
    print("• Implement context managers for automatic cleanup")
    print("• Monitor memory usage with torch.cuda.memory_allocated()")
    print("• Use multiple cleanup passes for thorough cleaning")
    print("="*60)


if __name__ == "__main__":
    main()