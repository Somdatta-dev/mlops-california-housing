"""
Locust performance testing file for MLOps California Housing API
Used by GitHub Actions workflows for performance benchmarking
"""

from locust import HttpUser, task, between
import json
import random


class MLOpsAPIUser(HttpUser):
    """
    Locust user class for testing the MLOps API endpoints
    """
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Called when a user starts"""
        # Test health endpoint on start
        response = self.client.get("/health")
        if response.status_code != 200:
            print(f"Health check failed: {response.status_code}")

    @task(3)
    def health_check(self):
        """Test health endpoint - most frequent task"""
        self.client.get("/health")

    @task(2)
    def model_info(self):
        """Test model info endpoint"""
        self.client.get("/model/info")

    @task(5)
    def predict_single(self):
        """Test single prediction endpoint - main functionality"""
        # Generate realistic California housing data
        payload = {
            "MedInc": round(random.uniform(0.5, 15.0), 2),
            "HouseAge": round(random.uniform(1.0, 52.0), 1),
            "AveRooms": round(random.uniform(2.0, 20.0), 2),
            "AveBedrms": round(random.uniform(0.5, 5.0), 2),
            "Population": round(random.uniform(100.0, 35000.0), 0),
            "AveOccup": round(random.uniform(1.0, 10.0), 2),
            "Latitude": round(random.uniform(32.5, 42.0), 2),
            "Longitude": round(random.uniform(-124.3, -114.3), 2)
        }
        
        with self.client.post("/predict", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if "prediction" in result:
                        response.success()
                    else:
                        response.failure("No prediction in response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def predict_batch(self):
        """Test batch prediction endpoint - less frequent"""
        # Generate batch of predictions
        batch_size = random.randint(2, 5)
        batch_payload = []
        
        for _ in range(batch_size):
            payload = {
                "MedInc": round(random.uniform(0.5, 15.0), 2),
                "HouseAge": round(random.uniform(1.0, 52.0), 1),
                "AveRooms": round(random.uniform(2.0, 20.0), 2),
                "AveBedrms": round(random.uniform(0.5, 5.0), 2),
                "Population": round(random.uniform(100.0, 35000.0), 0),
                "AveOccup": round(random.uniform(1.0, 10.0), 2),
                "Latitude": round(random.uniform(32.5, 42.0), 2),
                "Longitude": round(random.uniform(-124.3, -114.3), 2)
            }
            batch_payload.append(payload)
        
        with self.client.post("/predict/batch", json=batch_payload, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if "predictions" in result and len(result["predictions"]) == batch_size:
                        response.success()
                    else:
                        response.failure("Invalid batch response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def metrics_endpoint(self):
        """Test Prometheus metrics endpoint"""
        with self.client.get("/metrics", catch_response=True) as response:
            if response.status_code == 200:
                # Check if it looks like Prometheus metrics
                if "# HELP" in response.text or "# TYPE" in response.text:
                    response.success()
                else:
                    response.failure("Invalid metrics format")
            else:
                response.failure(f"HTTP {response.status_code}")

    def on_stop(self):
        """Called when a user stops"""
        pass


class HighLoadUser(HttpUser):
    """
    High-load user for stress testing
    """
    wait_time = between(0.1, 0.5)  # Very short wait time for stress testing
    
    @task
    def rapid_predictions(self):
        """Rapid fire predictions for stress testing"""
        payload = {
            "MedInc": 5.0,
            "HouseAge": 10.0,
            "AveRooms": 6.0,
            "AveBedrms": 1.2,
            "Population": 3000.0,
            "AveOccup": 3.0,
            "Latitude": 34.0,
            "Longitude": -118.0
        }
        
        self.client.post("/predict", json=payload)


class EdgeCaseUser(HttpUser):
    """
    User that tests edge cases and error conditions
    """
    wait_time = between(2, 5)
    
    @task
    def test_invalid_data(self):
        """Test with invalid data to check error handling"""
        invalid_payloads = [
            # Missing fields
            {"MedInc": 5.0},
            # Invalid ranges
            {
                "MedInc": -1.0,  # Negative income
                "HouseAge": 10.0,
                "AveRooms": 6.0,
                "AveBedrms": 1.2,
                "Population": 3000.0,
                "AveOccup": 3.0,
                "Latitude": 34.0,
                "Longitude": -118.0
            },
            # Extreme values
            {
                "MedInc": 1000.0,  # Very high income
                "HouseAge": 10.0,
                "AveRooms": 6.0,
                "AveBedrms": 1.2,
                "Population": 3000.0,
                "AveOccup": 3.0,
                "Latitude": 34.0,
                "Longitude": -118.0
            }
        ]
        
        payload = random.choice(invalid_payloads)
        
        with self.client.post("/predict", json=payload, catch_response=True) as response:
            # We expect these to fail with 422 (validation error)
            if response.status_code == 422:
                response.success()
            elif response.status_code == 200:
                response.failure("Expected validation error but got success")
            else:
                response.failure(f"Unexpected status code: {response.status_code}")

    @task
    def test_malformed_json(self):
        """Test with malformed JSON"""
        with self.client.post("/predict", 
                            data="invalid json", 
                            headers={"Content-Type": "application/json"},
                            catch_response=True) as response:
            if response.status_code == 422:
                response.success()
            else:
                response.failure(f"Expected 422 but got {response.status_code}")