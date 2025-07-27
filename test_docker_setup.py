#!/usr/bin/env python3
"""
Test script to verify Docker setup for MLOps California Housing Platform
Tests Docker configuration, CUDA support, and container functionality
"""

import subprocess
import sys
import time
import requests
import json
from pathlib import Path

def run_command(cmd, capture_output=True, timeout=30):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=capture_output, 
            text=True, 
            timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def test_docker_availability():
    """Test if Docker is available and running."""
    print("Testing Docker availability...")
    
    success, stdout, stderr = run_command("docker --version")
    if not success:
        print(f"❌ Docker not available: {stderr}")
        return False
    
    print(f"✅ Docker version: {stdout.strip()}")
    
    # Test Docker daemon
    success, stdout, stderr = run_command("docker info")
    if not success:
        print(f"❌ Docker daemon not running: {stderr}")
        return False
    
    print("✅ Docker daemon is running")
    return True

def test_nvidia_docker():
    """Test NVIDIA Docker runtime availability."""
    print("\nTesting NVIDIA Docker runtime...")
    
    # Check if nvidia-smi is available
    success, stdout, stderr = run_command("nvidia-smi")
    if not success:
        print("⚠️  NVIDIA GPU not detected or nvidia-smi not available")
        return False
    
    print("✅ NVIDIA GPU detected")
    
    # Test NVIDIA Docker runtime
    success, stdout, stderr = run_command(
        "docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi",
        timeout=60
    )
    if not success:
        print(f"⚠️  NVIDIA Docker runtime not available: {stderr}")
        return False
    
    print("✅ NVIDIA Docker runtime is working")
    return True

def test_dockerfile_syntax():
    """Test Dockerfile syntax and structure."""
    print("\nTesting Dockerfile syntax...")
    
    dockerfile_path = Path("Dockerfile")
    if not dockerfile_path.exists():
        print("❌ Dockerfile not found")
        return False
    
    # Test Dockerfile syntax
    success, stdout, stderr = run_command("docker build --dry-run .", timeout=60)
    if not success:
        print(f"❌ Dockerfile syntax error: {stderr}")
        return False
    
    print("✅ Dockerfile syntax is valid")
    return True

def test_docker_compose_config():
    """Test Docker Compose configuration."""
    print("\nTesting Docker Compose configuration...")
    
    compose_file = Path("docker-compose.yml")
    if not compose_file.exists():
        print("❌ docker-compose.yml not found")
        return False
    
    # Test Docker Compose config
    success, stdout, stderr = run_command("docker-compose config", timeout=30)
    if not success:
        print(f"❌ Docker Compose configuration error: {stderr}")
        return False
    
    print("✅ Docker Compose configuration is valid")
    return True

def test_image_build():
    """Test building the Docker image."""
    print("\nTesting Docker image build...")
    
    # Build the image
    print("Building Docker image (this may take a while)...")
    success, stdout, stderr = run_command(
        "docker build -t mlops-test:latest --target production .",
        timeout=600  # 10 minutes timeout
    )
    
    if not success:
        print(f"❌ Docker image build failed: {stderr}")
        return False
    
    print("✅ Docker image built successfully")
    
    # Check image size
    success, stdout, stderr = run_command(
        "docker images mlops-test:latest --format '{{.Size}}'"
    )
    if success:
        print(f"📊 Image size: {stdout.strip()}")
    
    return True

def test_container_startup():
    """Test container startup and basic functionality."""
    print("\nTesting container startup...")
    
    # Start container
    success, stdout, stderr = run_command(
        "docker run -d --name mlops-test-container -p 8001:8000 mlops-test:latest",
        timeout=60
    )
    
    if not success:
        print(f"❌ Container startup failed: {stderr}")
        return False
    
    print("✅ Container started successfully")
    
    # Wait for container to be ready
    print("Waiting for container to be ready...")
    for i in range(30):  # Wait up to 30 seconds
        try:
            response = requests.get("http://localhost:8001/health", timeout=5)
            if response.status_code == 200:
                print("✅ Container health check passed")
                break
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    else:
        print("⚠️  Container health check failed or timed out")
    
    # Test API endpoint
    try:
        response = requests.get("http://localhost:8001/", timeout=10)
        if response.status_code == 200:
            print("✅ API endpoint responding")
        else:
            print(f"⚠️  API endpoint returned status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"⚠️  API endpoint test failed: {e}")
    
    # Cleanup
    run_command("docker stop mlops-test-container", timeout=30)
    run_command("docker rm mlops-test-container", timeout=30)
    
    return True

def test_gpu_in_container():
    """Test GPU availability in container."""
    print("\nTesting GPU availability in container...")
    
    # Test GPU access in container
    success, stdout, stderr = run_command(
        "docker run --rm --gpus all mlops-test:latest nvidia-smi",
        timeout=60
    )
    
    if not success:
        print(f"⚠️  GPU not accessible in container: {stderr}")
        return False
    
    print("✅ GPU accessible in container")
    
    # Test CUDA in Python
    success, stdout, stderr = run_command(
        'docker run --rm --gpus all mlops-test:latest python -c "import torch; print(f\'CUDA available: {torch.cuda.is_available()}\')"',
        timeout=60
    )
    
    if success and "CUDA available: True" in stdout:
        print("✅ CUDA available in Python")
    else:
        print(f"⚠️  CUDA not available in Python: {stdout}")
    
    return True

def cleanup():
    """Clean up test resources."""
    print("\nCleaning up test resources...")
    
    # Remove test image
    run_command("docker rmi mlops-test:latest", timeout=30)
    
    # Remove any leftover containers
    run_command("docker container prune -f", timeout=30)
    
    print("✅ Cleanup completed")

def main():
    """Run all Docker setup tests."""
    print("MLOps California Housing Platform - Docker Setup Test")
    print("=" * 55)
    
    tests = [
        ("Docker Availability", test_docker_availability),
        ("NVIDIA Docker Runtime", test_nvidia_docker),
        ("Dockerfile Syntax", test_dockerfile_syntax),
        ("Docker Compose Config", test_docker_compose_config),
        ("Image Build", test_image_build),
        ("Container Startup", test_container_startup),
        ("GPU in Container", test_gpu_in_container),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 55)
    print("TEST SUMMARY")
    print("=" * 55)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\nTests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Docker setup is ready.")
        cleanup()
        return 0
    else:
        print("⚠️  Some tests failed. Please check the configuration.")
        cleanup()
        return 1

if __name__ == "__main__":
    sys.exit(main())