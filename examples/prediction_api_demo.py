#!/usr/bin/env python3
"""
Prediction API Endpoints Demonstration

This script demonstrates the complete Prediction API endpoints functionality
including single predictions, batch processing, and model information retrieval.
"""

import json
import time
import requests
from typing import Dict, List, Any
from datetime import datetime


class PredictionAPIDemo:
    """Demonstration of Prediction API endpoints."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the demo with API base URL.
        
        Args:
            base_url: Base URL of the FastAPI service
        """
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    def print_section(self, title: str) -> None:
        """Print a formatted section header."""
        print(f"\n{'='*60}")
        print(f"🔮 {title}")
        print(f"{'='*60}")
    
    def print_subsection(self, title: str) -> None:
        """Print a formatted subsection header."""
        print(f"\n{'─'*40}")
        print(f"📊 {title}")
        print(f"{'─'*40}")
    
    def check_api_health(self) -> bool:
        """Check if the API is running and healthy."""
        try:
            response = self.session.get(f"{self.base_url}/health/")
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ API is healthy: {health_data['status']}")
                print(f"   Version: {health_data['version']}")
                print(f"   Uptime: {health_data['uptime_seconds']:.2f}s")
                return True
            else:
                print(f"❌ API health check failed: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Cannot connect to API: {e}")
            print(f"   Make sure the API server is running at {self.base_url}")
            print(f"   Start it with: python src/api/run_server.py")
            return False
    
    def get_sample_housing_data(self) -> List[Dict[str, Any]]:
        """Get sample California Housing data for testing."""
        return [
            {
                "MedInc": 8.3252,      # High income area (Bay Area)
                "HouseAge": 41.0,
                "AveRooms": 6.984127,
                "AveBedrms": 1.023810,
                "Population": 322.0,
                "AveOccup": 2.555556,
                "Latitude": 37.88,     # San Francisco Bay Area
                "Longitude": -122.23
            },
            {
                "MedInc": 5.6431,      # Moderate income area (LA)
                "HouseAge": 25.0,
                "AveRooms": 5.817352,
                "AveBedrms": 1.073446,
                "Population": 2401.0,
                "AveOccup": 2.109842,
                "Latitude": 34.03,     # Los Angeles area
                "Longitude": -118.38
            },
            {
                "MedInc": 3.2596,      # Lower income area
                "HouseAge": 52.0,
                "AveRooms": 4.192308,
                "AveBedrms": 1.061538,
                "Population": 496.0,
                "AveOccup": 2.802260,
                "Latitude": 33.69,     # San Diego area
                "Longitude": -117.39
            }
        ]
    
    def demo_single_prediction(self) -> None:
        """Demonstrate single prediction endpoint."""
        self.print_subsection("Single Prediction Endpoint")
        
        # Get sample data
        sample_data = self.get_sample_housing_data()[0]
        
        print("📝 Sample Input Data:")
        for key, value in sample_data.items():
            print(f"   {key}: {value}")
        
        # Make prediction request
        print(f"\n🚀 Making prediction request to {self.base_url}/predict/")
        start_time = time.time()
        
        try:
            response = self.session.post(f"{self.base_url}/predict/", json=sample_data)
            request_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"✅ Prediction successful!")
                print(f"   🏠 Predicted Price: ${result['prediction'] * 100000:,.2f}")
                print(f"   📊 Model Version: {result['model_version']}")
                print(f"   🎯 Model Stage: {result['model_stage']}")
                print(f"   ⏱️  Processing Time: {result['processing_time_ms']:.2f}ms")
                print(f"   🌐 Request Time: {request_time:.2f}ms")
                print(f"   🆔 Request ID: {result['request_id']}")
                print(f"   📅 Timestamp: {result['timestamp']}")
                print(f"   🔧 Features Used: {result['features_used']}")
                
                if result.get('confidence_interval'):
                    lower, upper = result['confidence_interval']
                    print(f"   📈 Confidence Interval: ${lower * 100000:,.2f} - ${upper * 100000:,.2f}")
                
                if result.get('model_info'):
                    model_info = result['model_info']
                    print(f"   🤖 Algorithm: {model_info['algorithm']}")
                    if 'performance_metrics' in model_info:
                        metrics = model_info['performance_metrics']
                        print(f"   📊 R² Score: {metrics.get('r2_score', 'N/A')}")
                        print(f"   📊 RMSE: {metrics.get('rmse', 'N/A')}")
                
            else:
                print(f"❌ Prediction failed: {response.status_code}")
                print(f"   Error: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
    
    def demo_batch_prediction(self) -> None:
        """Demonstrate batch prediction endpoint."""
        self.print_subsection("Batch Prediction Endpoint")
        
        # Get sample data
        sample_data = self.get_sample_housing_data()
        
        batch_request = {
            "predictions": sample_data,
            "return_confidence": True,
            "batch_id": f"demo_batch_{int(time.time())}"
        }
        
        print(f"📝 Batch Request:")
        print(f"   Predictions: {len(batch_request['predictions'])}")
        print(f"   Return Confidence: {batch_request['return_confidence']}")
        print(f"   Batch ID: {batch_request['batch_id']}")
        
        # Make batch prediction request
        print(f"\n🚀 Making batch prediction request to {self.base_url}/predict/batch")
        start_time = time.time()
        
        try:
            response = self.session.post(f"{self.base_url}/predict/batch", json=batch_request)
            request_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"✅ Batch prediction successful!")
                print(f"   📊 Total Predictions: {result['total_predictions']}")
                print(f"   ✅ Successful: {result['successful_predictions']}")
                print(f"   ❌ Failed: {result['failed_predictions']}")
                print(f"   📈 Status: {result['status']}")
                print(f"   ⏱️  Total Processing Time: {result['total_processing_time_ms']:.2f}ms")
                print(f"   ⏱️  Average Processing Time: {result['average_processing_time_ms']:.2f}ms")
                print(f"   🌐 Request Time: {request_time:.2f}ms")
                print(f"   🆔 Batch ID: {result['batch_id']}")
                
                print(f"\n📋 Individual Predictions:")
                for i, prediction in enumerate(result['predictions']):
                    if 'prediction' in prediction:
                        price = prediction['prediction'] * 100000
                        processing_time = prediction['processing_time_ms']
                        print(f"   {i+1}. ${price:,.2f} ({processing_time:.2f}ms)")
                        
                        if prediction.get('confidence_interval'):
                            lower, upper = prediction['confidence_interval']
                            print(f"      Confidence: ${lower * 100000:,.2f} - ${upper * 100000:,.2f}")
                    else:
                        print(f"   {i+1}. ❌ Error: {prediction.get('message', 'Unknown error')}")
                
            else:
                print(f"❌ Batch prediction failed: {response.status_code}")
                print(f"   Error: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
    
    def demo_model_info(self) -> None:
        """Demonstrate model information endpoint."""
        self.print_subsection("Model Information Endpoint")
        
        print(f"🚀 Getting model information from {self.base_url}/predict/model/info")
        
        try:
            response = self.session.get(f"{self.base_url}/predict/model/info")
            
            if response.status_code == 200:
                model_info = response.json()
                
                print(f"✅ Model information retrieved!")
                print(f"   🤖 Name: {model_info['name']}")
                print(f"   📊 Version: {model_info['version']}")
                print(f"   🎯 Stage: {model_info['stage']}")
                print(f"   🔧 Algorithm: {model_info['algorithm']}")
                print(f"   🏗️  Framework: {model_info['framework']}")
                print(f"   📅 Training Date: {model_info['training_date']}")
                print(f"   📈 GPU Accelerated: {model_info['gpu_accelerated']}")
                
                if model_info.get('model_size_mb'):
                    print(f"   💾 Model Size: {model_info['model_size_mb']:.1f} MB")
                
                print(f"\n📊 Features ({len(model_info['features'])}):")
                for i, feature in enumerate(model_info['features'], 1):
                    print(f"   {i}. {feature}")
                
                if model_info.get('performance_metrics'):
                    metrics = model_info['performance_metrics']
                    print(f"\n📈 Performance Metrics:")
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            print(f"   {metric}: {value:.4f}")
                        else:
                            print(f"   {metric}: {value}")
                
                if model_info.get('description'):
                    print(f"\n📝 Description: {model_info['description']}")
                
                if model_info.get('tags'):
                    print(f"\n🏷️  Tags:")
                    for tag, value in model_info['tags'].items():
                        print(f"   {tag}: {value}")
                
            else:
                print(f"❌ Model info request failed: {response.status_code}")
                print(f"   Error: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
    
    def demo_error_handling(self) -> None:
        """Demonstrate error handling capabilities."""
        self.print_subsection("Error Handling Demonstration")
        
        # Test validation error
        print("🧪 Testing validation error with invalid data:")
        invalid_data = {
            "MedInc": -1.0,        # Invalid negative income
            "HouseAge": 41.0,
            "AveRooms": 6.984127,
            "AveBedrms": 1.023810,
            "Population": 322.0,
            "AveOccup": 2.555556,
            "Latitude": 50.0,      # Invalid latitude (outside California)
            "Longitude": -122.23
        }
        
        try:
            response = self.session.post(f"{self.base_url}/predict/", json=invalid_data)
            
            if response.status_code == 422:
                error_data = response.json()
                print(f"✅ Validation error handled correctly!")
                print(f"   Status Code: {response.status_code}")
                print(f"   Error Type: {error_data['error']}")
                print(f"   Message: {error_data['message']}")
                
                if 'details' in error_data:
                    print(f"   Validation Details:")
                    for detail in error_data['details'][:3]:  # Show first 3 errors
                        field = detail.get('loc', ['unknown'])[-1]
                        message = detail.get('msg', 'Unknown error')
                        print(f"     - {field}: {message}")
            else:
                print(f"❌ Unexpected response: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
        
        # Test batch size limit
        print(f"\n🧪 Testing batch size limit (>100 predictions):")
        large_batch = {
            "predictions": [self.get_sample_housing_data()[0]] * 101  # Exceed limit
        }
        
        try:
            response = self.session.post(f"{self.base_url}/predict/batch", json=large_batch)
            
            if response.status_code == 422:
                error_data = response.json()
                print(f"✅ Batch size limit enforced correctly!")
                print(f"   Status Code: {response.status_code}")
                print(f"   Error Type: {error_data['error']}")
                print(f"   Message: {error_data['message']}")
            else:
                print(f"❌ Unexpected response: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
    
    def demo_performance_comparison(self) -> None:
        """Demonstrate performance comparison between single and batch predictions."""
        self.print_subsection("Performance Comparison")
        
        sample_data = self.get_sample_housing_data()[0]
        num_predictions = 10
        
        # Single predictions
        print(f"🚀 Testing {num_predictions} single predictions:")
        single_times = []
        
        for i in range(num_predictions):
            start_time = time.time()
            try:
                response = self.session.post(f"{self.base_url}/predict/", json=sample_data)
                if response.status_code == 200:
                    request_time = (time.time() - start_time) * 1000
                    single_times.append(request_time)
                    result = response.json()
                    processing_time = result['processing_time_ms']
                    print(f"   {i+1}. Request: {request_time:.2f}ms, Processing: {processing_time:.2f}ms")
            except requests.exceptions.RequestException:
                print(f"   {i+1}. ❌ Failed")
        
        # Batch prediction
        print(f"\n🚀 Testing 1 batch prediction with {num_predictions} items:")
        batch_request = {
            "predictions": [sample_data] * num_predictions
        }
        
        start_time = time.time()
        try:
            response = self.session.post(f"{self.base_url}/predict/batch", json=batch_request)
            if response.status_code == 200:
                batch_request_time = (time.time() - start_time) * 1000
                result = response.json()
                batch_processing_time = result['total_processing_time_ms']
                avg_processing_time = result['average_processing_time_ms']
                
                print(f"   Request: {batch_request_time:.2f}ms")
                print(f"   Total Processing: {batch_processing_time:.2f}ms")
                print(f"   Average Processing: {avg_processing_time:.2f}ms")
                
                # Performance comparison
                if single_times:
                    avg_single_request = sum(single_times) / len(single_times)
                    total_single_time = sum(single_times)
                    
                    print(f"\n📊 Performance Comparison:")
                    print(f"   Single Predictions:")
                    print(f"     Average Request Time: {avg_single_request:.2f}ms")
                    print(f"     Total Time: {total_single_time:.2f}ms")
                    print(f"   Batch Prediction:")
                    print(f"     Request Time: {batch_request_time:.2f}ms")
                    print(f"     Average per Item: {batch_request_time / num_predictions:.2f}ms")
                    
                    speedup = total_single_time / batch_request_time
                    print(f"   🚀 Batch Speedup: {speedup:.2f}x faster")
        
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Batch request failed: {e}")
    
    def run_demo(self) -> None:
        """Run the complete prediction API demonstration."""
        self.print_section("Prediction API Endpoints Demonstration")
        
        print("This demo showcases the complete Prediction API endpoints functionality")
        print("including single predictions, batch processing, model information,")
        print("error handling, and performance comparisons.")
        
        # Check API health
        if not self.check_api_health():
            return
        
        # Run demonstrations
        self.demo_single_prediction()
        self.demo_batch_prediction()
        self.demo_model_info()
        self.demo_error_handling()
        self.demo_performance_comparison()
        
        # Summary
        self.print_section("Demo Summary")
        print("✅ Single Prediction: Advanced validation with real-time inference")
        print("✅ Batch Processing: Efficient processing with partial success handling")
        print("✅ Model Information: Comprehensive metadata and performance metrics")
        print("✅ Error Handling: Proper validation and error reporting")
        print("✅ Performance: Optimized processing with batch speedup")
        print("\n🎉 Prediction API Endpoints demonstration completed successfully!")
        print("\n📚 For more information, see:")
        print("   - PREDICTION_API_ENDPOINTS_SUMMARY.md")
        print("   - src/api/README.md")
        print("   - API Documentation: http://localhost:8000/docs (debug mode)")


def main():
    """Main function to run the demonstration."""
    demo = PredictionAPIDemo()
    demo.run_demo()


if __name__ == "__main__":
    main()