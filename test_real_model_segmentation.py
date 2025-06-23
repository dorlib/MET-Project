#!/usr/bin/env python3
# Script to test the model service with real segmentation

import requests
import json
import os
import time
import sys

# Configure API endpoint
API_URL = "http://localhost:5001"  # If port forwarded directly, or through API gateway

def check_health():
    """Check if the model service is healthy"""
    try:
        response = requests.get(f"{API_URL}/health")
        response.raise_for_status()
        print(f"Health check result: {response.json()}")
        return True
    except Exception as e:
        print(f"Health check failed: {str(e)}")
        return False

def test_prediction(file_path):
    """Test model prediction with a file"""
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return False
    
    print(f"Testing prediction with file: {file_path}")
    
    try:
        # Start a prediction job
        payload = {
            "file_path": file_path,
            "job_id": f"test_real_model_{int(time.time())}"
        }
        
        response = requests.post(
            f"{API_URL}/predict", 
            json=payload
        )
        response.raise_for_status()
        result = response.json()
        
        job_id = result.get("job_id")
        print(f"Job started with ID: {job_id}")
        print(f"Initial status: {result.get('status')}")
        
        # Poll for job completion
        max_tries = 30
        tries = 0
        while tries < max_tries:
            tries += 1
            
            try:
                status_response = requests.get(f"{API_URL}/status/{job_id}")
                status_response.raise_for_status()
                status_data = status_response.json()
                
                print(f"Poll {tries}/{max_tries}: Status = {status_data.get('status')}")
                
                if status_data.get("status") == "completed":
                    print("Job completed successfully!")
                    print(f"Results saved at: {status_data.get('prediction_path')}")
                    print(f"Visualization saved at: {status_data.get('visualization_path')}")
                    return True
                elif status_data.get("status") == "failed":
                    print(f"Job failed: {status_data.get('error')}")
                    return False
                    
                time.sleep(2)  # Wait 2 seconds between polls
                
            except Exception as e:
                print(f"Error checking status: {str(e)}")
                time.sleep(5)  # Wait longer on error
        
        print("Timeout waiting for job completion")
        return False
        
    except Exception as e:
        print(f"Error starting prediction job: {str(e)}")
        return False

if __name__ == "__main__":
    # Check if model service is healthy
    if not check_health():
        print("Model service health check failed. Please ensure the service is running.")
        sys.exit(1)
    
    # Default test file
    test_file = "test_data/sample_nifti.nii.gz"
    
    # Allow overriding test file from command line
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    
    # Get absolute path if not already
    if not os.path.isabs(test_file):
        # Try to find the file relative to project root
        test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), test_file)
    
    # Test prediction
    if test_prediction(test_file):
        print("Test completed successfully!")
        sys.exit(0)
    else:
        print("Test failed.")
        sys.exit(1)
