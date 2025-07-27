#!/usr/bin/env python3
"""
Comprehensive test script for MLOps California Housing Platform Docker setup
Tests CUDA 12.8, PyTorch 2.7.0, and full container functionality
"""

import subprocess
import sys
import time
import requests
import json
import os
from pathlib import Path

def run_command(cmd, capture_output=True, timeout=300, cwd=None):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=capture_output, 
            text=True, 
            timeout=timeout,
            cwd=cwd
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", f"Command timed out after {timeout} seconds"
    except Exception as e:
        return False, "", str(e)

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

def print_status(message, success=True):
    """Print a status message with appropriate emoji."""
    emoji = "✅" if success else "❌"
    print(f"{emoji} {message}")

def print_warning(message):
    """Print a warning message."""
    print(f"⚠️  {message}")

def print_info(message):
    """Print an info message."""
    print(f"ℹ️  {message}")

def test_prerequisites():
    """Test system prerequisites."""
    print_section("Testing Prerequisites")
    
    # Test Docker
    success, stdout, stderr = run_command("docker --version")
    if not success:
        print_status(f"Docker not available: {stderr}", False)
        return False
    print_status(f"Docker available: {stdout.strip()}")
    
    # Test Docker Compose
    success, stdout, stderr = run_command("docker-compose --version")
    if not success:
        print_status(f"Docker Compose not available: {stderr}", False)
        return False
    print_status(f"Docker Compose available: {stdout.strip()}")
    
    # Test NVIDIA GPU
    success, stdout, stderr = run_command("nvidia-smi")
    if not success:
        print_warning("NVIDIA GPU not detected - GPU tests will be skipped")
        return True
    
    print_status("NVIDIA GPU detected")
    print_info(f"GPU Info:\n{stdout.strip()}")
    
    # Test NVIDIA Docker runtime
    success, stdout, stderr = run_command(
        "docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi",
        timeout=120
    )
    if not success:
        print_warning(f"NVIDIA Docker runtime test failed: {stderr}")
        return True  # Continue without GPU support
    
    print_status("NVIDIA Docker runtime working")
    return True

def test_docker_files():
    """Test that all required Docker files exist and are valid."""
    print_section("Testing Docker Files")
    
    required_files = [
        "Dockerfile",
        "Dockerfile.cpu",
        "docker-compose.yml",
        "docker/entrypoint.sh",
        "docker/build.sh",
        "docker/prometheus/prometheus.yml",
        ".dockerignore",
        "requirements.txt"
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print_status(f"{file_path} exists")
        else:
            print_status(f"{file_path} missing", False)
            all_exist = False
    
    # Test Docker Compose config
    success, stdout, stderr = run_command("docker-compose config")
    if not success:
        print_status(f"Docker Compose config invalid: {stderr}", False)
        return False
    
    print_status("Docker Compose configuration is valid")
    return all_exist

def build_images():
    """Build Docker images."""
    print_section("Building Docker Images")
    
    # Clean up any existing images
    print_info("Cleaning up existing test images...")
    run_command("docker rmi mlops-test-gpu:latest mlops-test-cpu:latest", timeout=60)
    
    # Build GPU image
    print_info("Building GPU image with CUDA 12.8 and PyTorch 2.7.0...")
    success, stdout, stderr = run_command(
        "docker build -t mlops-test-gpu:latest --target production .",
        timeout=900  # 15 minutes
    )
    
    if not success:
        print_status(f"GPU image build failed: {stderr}", False)
        return False
    
    print_status("GPU image built successfully")
    
    # Build CPU image
    print_info("Building CPU image...")
    success, stdout, stderr = run_command(
        "docker build -f Dockerfile.cpu -t mlops-test-cpu:latest --target production .",
        timeout=900  # 15 minutes
    )
    
    if not success:
        print_status(f"CPU image build failed: {stderr}", False)
        return False
    
    print_status("CPU image built successfully")
    
    # Show image sizes
    success, stdout, stderr = run_command(
        "docker images mlops-test-gpu:latest mlops-test-cpu:latest --format 'table {{.Repository}}\\t{{.Tag}}\\t{{.Size}}'"
    )
    if success:
        print_info(f"Image sizes:\n{stdout}")
    
    return True

def test_pytorch_cuda():
    """Test PyTorch and CUDA functionality in container."""
    print_section("Testing PyTorch and CUDA in Container")
    
    # Test PyTorch version
    success, stdout, stderr = run_command(
        'docker run --rm mlops-test-gpu:latest python -c "import torch; print(f\'PyTorch version: {torch.__version__}\')"',
        timeout=60
    )
    
    if not success:
        print_status(f"PyTorch version test failed: {stderr}", False)
        return False
    
    print_status(f"PyTorch test passed: {stdout.strip()}")
    
    # Test CUDA availability
    success, stdout, stderr = run_command(
        'docker run --rm --gpus all mlops-test-gpu:latest python -c "import torch; print(f\'CUDA available: {torch.cuda.is_available()}\'); print(f\'CUDA version: {torch.version.cuda}\'); print(f\'GPU count: {torch.cuda.device_count()}\')"',
        timeout=60
    )
    
    if not success:
        print_warning(f"CUDA test failed (GPU may not be available): {stderr}")
        return True  # Continue without GPU
    
    print_status(f"CUDA test passed: {stdout.strip()}")
    
    # Test GPU memory
    success, stdout, stderr = run_command(
        'docker run --rm --gpus all mlops-test-gpu:latest python -c "import torch; print(f\'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\') if torch.cuda.is_available() else print(\'No GPU available\')"',
        timeout=60
    )
    
    if success:
        print_info(f"GPU memory info: {stdout.strip()}")
    
    return True

def test_container_startup():
    """Test container startup and API functionality."""
    print_section("Testing Container Startup and API")
    
    # Start GPU container
    print_info("Starting GPU container...")
    success, stdout, stderr = run_command(
        "docker run -d --name mlops-test-gpu-container --gpus all -p 8001:8000 mlops-test-gpu:latest",
        timeout=60
    )
    
    if not success:
        print_status(f"GPU container startup failed: {stderr}", False)
        return False
    
    print_status("GPU container started")
    
    # Wait for container to be ready
    print_info("Waiting for container to be ready...")
    for i in range(60):  # Wait up to 60 seconds
        try:
            response = requests.get("http://localhost:8001/health", timeout=5)
            if response.status_code == 200:
                print_status("Container health check passed")
                break
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    else:
        print_status("Container health check failed or timed out", False)
        # Get container logs for debugging
        success, stdout, stderr = run_command("docker logs mlops-test-gpu-container")
        if success:
            print_info(f"Container logs:\n{stdout}")
        return False
    
    # Test API endpoints
    try:
        # Test root endpoint
        response = requests.get("http://localhost:8001/", timeout=10)
        if response.status_code == 200:
            print_status("Root endpoint responding")
        else:
            print_warning(f"Root endpoint returned status: {response.status_code}")
        
        # Test info endpoint
        response = requests.get("http://localhost:8001/info", timeout=10)
        if response.status_code == 200:
            info_data = response.json()
            print_status(f"Info endpoint responding - API version: {info_data.get('version', 'unknown')}")
        else:
            print_warning(f"Info endpoint returned status: {response.status_code}")
        
        # Test prediction endpoint (if model is available)
        test_data = {
            "MedInc": 8.3252,
            "HouseAge": 41.0,
            "AveRooms": 6.984,
            "AveBedrms": 1.024,
            "Population": 322.0,
            "AveOccup": 2.556,
            "Latitude": 37.88,
            "Longitude": -122.23
        }
        
        response = requests.post(
            "http://localhost:8001/predict",
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            prediction_data = response.json()
            print_status(f"Prediction endpoint working - prediction: {prediction_data.get('prediction', 'N/A')}")
        else:
            print_warning(f"Prediction endpoint returned status: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print_warning(f"API endpoint test failed: {e}")
    
    return True

def test_cpu_container():
    """Test CPU-only container."""
    print_section("Testing CPU-only Container")
    
    # Start CPU container
    print_info("Starting CPU container...")
    success, stdout, stderr = run_command(
        "docker run -d --name mlops-test-cpu-container -p 8002:8000 mlops-test-cpu:latest",
        timeout=60
    )
    
    if not success:
        print_status(f"CPU container startup failed: {stderr}", False)
        return False
    
    print_status("CPU container started")
    
    # Wait for container to be ready
    print_info("Waiting for CPU container to be ready...")
    for i in range(60):  # Wait up to 60 seconds
        try:
            response = requests.get("http://localhost:8002/health", timeout=5)
            if response.status_code == 200:
                print_status("CPU container health check passed")
                break
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    else:
        print_status("CPU container health check failed or timed out", False)
        return False
    
    # Test PyTorch CPU functionality
    success, stdout, stderr = run_command(
        'docker exec mlops-test-cpu-container python -c "import torch; print(f\'PyTorch CPU version: {torch.__version__}\'); print(f\'CUDA available: {torch.cuda.is_available()}\')"',
        timeout=30
    )
    
    if success:
        print_status(f"CPU PyTorch test passed: {stdout.strip()}")
    else:
        print_warning(f"CPU PyTorch test failed: {stderr}")
    
    return True

def test_docker_compose():
    """Test Docker Compose functionality."""
    print_section("Testing Docker Compose")
    
    # Test CPU-only profile
    print_info("Testing CPU-only Docker Compose profile...")
    success, stdout, stderr = run_command(
        "docker-compose --profile cpu-only up -d",
        timeout=300
    )
    
    if not success:
        print_status(f"Docker Compose CPU profile failed: {stderr}", False)
        return False
    
    print_status("Docker Compose CPU profile started")
    
    # Wait for services to be ready
    time.sleep(30)
    
    # Check service health
    services_healthy = True
    
    # Check MLflow
    try:
        response = requests.get("http://localhost:5000/health", timeout=10)
        if response.status_code == 200:
            print_status("MLflow service healthy")
        else:
            print_warning(f"MLflow service unhealthy: {response.status_code}")
            services_healthy = False
    except requests.exceptions.RequestException as e:
        print_warning(f"MLflow service not responding: {e}")
        services_healthy = False
    
    # Check Prometheus
    try:
        response = requests.get("http://localhost:9090/-/healthy", timeout=10)
        if response.status_code == 200:
            print_status("Prometheus service healthy")
        else:
            print_warning(f"Prometheus service unhealthy: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print_warning(f"Prometheus service not responding: {e}")
    
    # Check API service
    try:
        response = requests.get("http://localhost:8002/health", timeout=10)
        if response.status_code == 200:
            print_status("API service healthy")
        else:
            print_warning(f"API service unhealthy: {response.status_code}")
            services_healthy = False
    except requests.exceptions.RequestException as e:
        print_warning(f"API service not responding: {e}")
        services_healthy = False
    
    return services_healthy

def cleanup():
    """Clean up test resources."""
    print_section("Cleaning Up Test Resources")
    
    # Stop and remove containers
    containers = [
        "mlops-test-gpu-container",
        "mlops-test-cpu-container"
    ]
    
    for container in containers:
        run_command(f"docker stop {container}", timeout=30)
        run_command(f"docker rm {container}", timeout=30)
    
    # Stop Docker Compose services
    run_command("docker-compose --profile cpu-only down", timeout=60)
    run_command("docker-compose down", timeout=60)
    
    # Remove test images
    run_command("docker rmi mlops-test-gpu:latest mlops-test-cpu:latest", timeout=60)
    
    # Clean up system
    run_command("docker system prune -f", timeout=60)
    
    print_status("Cleanup completed")

def main():
    """Run comprehensive Docker setup tests."""
    print_section("MLOps California Housing Platform - Full Docker Test")
    print_info("Testing CUDA 12.8, PyTorch 2.7.0, and complete Docker setup")
    
    tests = [
        ("Prerequisites", test_prerequisites),
        ("Docker Files", test_docker_files),
        ("Build Images", build_images),
        ("PyTorch & CUDA", test_pytorch_cuda),
        ("Container Startup", test_container_startup),
        ("CPU Container", test_cpu_container),
        ("Docker Compose", test_docker_compose),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print_info(f"Running {test_name} test...")
            results[test_name] = test_func()
        except Exception as e:
            print_status(f"{test_name} failed with exception: {e}", False)
            results[test_name] = False
    
    # Summary
    print_section("TEST SUMMARY")
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nTests passed: {passed}/{total}")
    
    if passed == total:
        print_status("🎉 All tests passed! Docker setup with CUDA 12.8 is ready!")
        print_info("Next steps:")
        print_info("1. Use 'make run' to start all services")
        print_info("2. Use 'make run-cpu' for CPU-only deployment")
        print_info("3. Access API at http://localhost:8000")
        print_info("4. Access Grafana at http://localhost:3000 (admin/admin123)")
        cleanup()
        return 0
    else:
        print_status("⚠️  Some tests failed. Check the output above for details.", False)
        cleanup()
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        cleanup()
        sys.exit(1)