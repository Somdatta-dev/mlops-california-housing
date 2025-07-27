#!/usr/bin/env python3
"""
Minimal test to verify basic Docker functionality
Tests basic Docker commands and configuration validation
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd):
    """Run a shell command and return success status."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def test_docker_basic():
    """Test basic Docker functionality."""
    print("Testing basic Docker functionality...")
    
    # Test Docker version
    success, stdout, stderr = run_command("docker --version")
    if not success:
        print(f"❌ Docker not available: {stderr}")
        return False
    print(f"✅ Docker available: {stdout.strip()}")
    
    # Test Docker daemon
    success, stdout, stderr = run_command("docker info")
    if not success:
        print(f"❌ Docker daemon not running: {stderr}")
        return False
    print("✅ Docker daemon is running")
    
    return True

def test_files_exist():
    """Test that required Docker files exist."""
    print("\nTesting Docker files...")
    
    required_files = [
        "Dockerfile",
        "Dockerfile.cpu", 
        "docker-compose.yml",
        "docker/entrypoint.sh",
        "docker/build.sh",
        "docker/prometheus/prometheus.yml",
        ".dockerignore"
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist

def test_compose_config():
    """Test Docker Compose configuration."""
    print("\nTesting Docker Compose configuration...")
    
    success, stdout, stderr = run_command("docker-compose config")
    if not success:
        print(f"❌ Docker Compose config invalid: {stderr}")
        return False
    
    print("✅ Docker Compose configuration is valid")
    return True

def test_dockerfile_syntax():
    """Test basic Dockerfile syntax."""
    print("\nTesting Dockerfile syntax...")
    
    # Test main Dockerfile
    success, stdout, stderr = run_command("docker build --help")
    if not success:
        print("❌ Docker build command not available")
        return False
    
    print("✅ Docker build command available")
    
    # Check if Dockerfile exists and has basic structure
    dockerfile_path = Path("Dockerfile")
    if not dockerfile_path.exists():
        print("❌ Dockerfile not found")
        return False
    
    content = dockerfile_path.read_text()
    if "FROM" in content and "WORKDIR" in content:
        print("✅ Dockerfile has basic structure")
        return True
    else:
        print("❌ Dockerfile missing basic structure")
        return False

def main():
    """Run minimal Docker tests."""
    print("MLOps Platform - Minimal Docker Test")
    print("=" * 40)
    
    tests = [
        ("Docker Basic", test_docker_basic),
        ("Required Files", test_files_exist),
        ("Compose Config", test_compose_config),
        ("Dockerfile Syntax", test_dockerfile_syntax),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 40)
    print("TEST SUMMARY")
    print("=" * 40)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nTests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 Basic Docker setup is ready!")
        print("\nNext steps:")
        print("1. Build CPU image: make build-cpu")
        print("2. Run CPU service: make run-cpu")
        print("3. Test API: curl http://localhost:8002/health")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the configuration.")
        return 1

if __name__ == "__main__":
    sys.exit(main())