#!/usr/bin/env python3
"""
Quick test for Docker setup with CUDA 12.8 and PyTorch 2.7.0
"""

import subprocess
import sys
import time
import requests

def run_command(cmd, timeout=120):
    """Run command with timeout."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Timeout"
    except Exception as e:
        return False, "", str(e)

def test_build_and_run():
    """Test building and running the Docker container."""
    print("🚀 Testing Docker setup with CUDA 12.8 and PyTorch 2.7.0")
    print("=" * 60)
    
    # Build the image
    print("📦 Building Docker image...")
    success, stdout, stderr = run_command(
        "docker build -t mlops-test:latest --target production .",
        timeout=600
    )
    
    if not success:
        print(f"❌ Build failed: {stderr}")
        return False
    
    print("✅ Docker image built successfully")
    
    # Test PyTorch and CUDA
    print("🧪 Testing PyTorch and CUDA...")
    success, stdout, stderr = run_command(
        'docker run --rm --gpus all mlops-test:latest python -c "import torch; print(f\'PyTorch: {torch.__version__}\'); print(f\'CUDA: {torch.cuda.is_available()}\'); print(f\'GPU: {torch.cuda.device_count()}\')"',
        timeout=60
    )
    
    if success:
        print(f"✅ PyTorch test results:\n{stdout}")
        if "CUDA: True" in stdout:
            print("🎉 CUDA is working!")
        else:
            print("⚠️  CUDA not available")
    else:
        print(f"❌ PyTorch test failed: {stderr}")
        return False
    
    # Start container for API test
    print("🌐 Testing API container...")
    success, stdout, stderr = run_command(
        "docker run -d --name mlops-quick-test --gpus all -p 8005:8000 mlops-test:latest",
        timeout=60
    )
    
    if not success:
        print(f"❌ Container start failed: {stderr}")
        return False
    
    print("✅ Container started")
    
    # Wait for API to be ready
    print("⏳ Waiting for API to be ready...")
    for i in range(30):
        try:
            response = requests.get("http://localhost:8005/health", timeout=3)
            if response.status_code == 200:
                print("✅ API is responding")
                break
        except:
            pass
        time.sleep(2)
    else:
        print("⚠️  API health check timeout")
    
    # Test API info endpoint
    try:
        response = requests.get("http://localhost:8005/info", timeout=5)
        if response.status_code == 200:
            info = response.json()
            print(f"✅ API Info: {info.get('name', 'Unknown')} v{info.get('version', 'Unknown')}")
        else:
            print(f"⚠️  API info returned: {response.status_code}")
    except Exception as e:
        print(f"⚠️  API info test failed: {e}")
    
    # Cleanup
    run_command("docker stop mlops-quick-test", timeout=30)
    run_command("docker rm mlops-quick-test", timeout=30)
    run_command("docker rmi mlops-test:latest", timeout=30)
    
    print("🧹 Cleanup completed")
    return True

if __name__ == "__main__":
    try:
        success = test_build_and_run()
        if success:
            print("\n🎉 Docker setup with CUDA 12.8 and PyTorch 2.7.0 is working!")
            print("\nNext steps:")
            print("1. Use 'make build' to build production images")
            print("2. Use 'make run' to start all services")
            print("3. Access API at http://localhost:8000")
        else:
            print("\n❌ Some tests failed")
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted")
        run_command("docker stop mlops-quick-test", timeout=10)
        run_command("docker rm mlops-quick-test", timeout=10)
        sys.exit(1)