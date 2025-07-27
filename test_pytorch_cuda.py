#!/usr/bin/env python3
"""
Test PyTorch 2.7.0 with CUDA 12.8 in Docker container
"""

import subprocess
import sys

def test_pytorch_cuda_build():
    """Test building a simple PyTorch CUDA container."""
    
    # Create a simple test Dockerfile
    dockerfile_content = """
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Install Python and pip
RUN apt-get update && apt-get install -y \\
    python3.10 \\
    python3-pip \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create symlinks
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \\
    ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch 2.7.0 with CUDA 12.8
RUN pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# Test script
RUN echo 'import torch; print(f"PyTorch version: {torch.__version__}"); print(f"CUDA available: {torch.cuda.is_available()}"); print(f"CUDA version: {torch.version.cuda}"); print(f"GPU count: {torch.cuda.device_count()}")' > /test_pytorch.py

CMD ["python", "/test_pytorch.py"]
"""
    
    # Write test Dockerfile
    with open("Dockerfile.test", "w") as f:
        f.write(dockerfile_content)
    
    print("Building test PyTorch CUDA container...")
    
    # Build test image
    result = subprocess.run([
        "docker", "build", "-f", "Dockerfile.test", "-t", "pytorch-cuda-test", "."
    ], capture_output=True, text=True, timeout=600)
    
    if result.returncode != 0:
        print(f"❌ Build failed: {result.stderr}")
        return False
    
    print("✅ Build successful")
    
    # Test PyTorch without GPU
    print("Testing PyTorch (CPU mode)...")
    result = subprocess.run([
        "docker", "run", "--rm", "pytorch-cuda-test"
    ], capture_output=True, text=True, timeout=60)
    
    if result.returncode == 0:
        print(f"✅ PyTorch CPU test: {result.stdout.strip()}")
    else:
        print(f"❌ PyTorch CPU test failed: {result.stderr}")
        return False
    
    # Test PyTorch with GPU
    print("Testing PyTorch (GPU mode)...")
    result = subprocess.run([
        "docker", "run", "--rm", "--gpus", "all", "pytorch-cuda-test"
    ], capture_output=True, text=True, timeout=60)
    
    if result.returncode == 0:
        print(f"✅ PyTorch GPU test: {result.stdout.strip()}")
        
        # Check if CUDA is actually available
        if "CUDA available: True" in result.stdout:
            print("🎉 CUDA is working in container!")
            return True
        else:
            print("⚠️  CUDA not available in container")
            return False
    else:
        print(f"❌ PyTorch GPU test failed: {result.stderr}")
        return False

def cleanup():
    """Clean up test resources."""
    subprocess.run(["docker", "rmi", "pytorch-cuda-test"], capture_output=True)
    try:
        import os
        os.remove("Dockerfile.test")
    except:
        pass

if __name__ == "__main__":
    try:
        success = test_pytorch_cuda_build()
        cleanup()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted")
        cleanup()
        sys.exit(1)
    except Exception as e:
        print(f"Test failed with exception: {e}")
        cleanup()
        sys.exit(1)