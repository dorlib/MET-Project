#!/usr/bin/env python3
# test_services.py - Script to test the microservice APIs

import requests
import time
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from requests.exceptions import RequestException, Timeout, ConnectionError

# Default configuration
API_URL = "http://localhost:5000"
MODEL_SERVICE_URL = "http://localhost:5001" 
IMAGE_PROCESSING_URL = "http://localhost:5002"
USER_SERVICE_URL = "http://localhost:5003"

# Test user credentials
TEST_USER = {
    "name": "Test User",
    "email": "test@example.com",
    "password": "test_password123"
}

def check_health(service_name, url):
    """Check if a service is healthy by calling its health endpoint."""
    try:
        response = requests.get(f"{url}/health", timeout=5)
        if response.status_code == 200:
            print(f"✅ {service_name} is healthy")
            return True
        else:
            print(f"❌ {service_name} returned status code {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ {service_name} is not accessible: {e}")
        return False

def test_upload_file(file_path, auth_header=None):
    """Test file upload to the API gateway."""
    if not os.path.exists(file_path):
        print(f"❌ Test file not found: {file_path}")
        return None
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            # Include auth header if available for user association
            headers = auth_header if auth_header else {}
            response = requests.post(f"{API_URL}/upload", files=files, headers=headers)
        
        if response.status_code == 200:
            job_id = response.json().get('job_id')
            user_associated = response.json().get('user_associated', False)
            print(f"✅ File upload successful. Job ID: {job_id}")
            if user_associated:
                print("   ✅ File associated with user account")
            return job_id
        else:
            print(f"❌ File upload failed: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Error uploading file: {e}")
        return None

def poll_for_results(job_id, max_attempts=30, delay=5):
    """Poll for job results with a timeout."""
    print(f"Polling for results (job_id: {job_id})...")
    
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{API_URL}/results/{job_id}")
            data = response.json()
            
            if data.get('status') == 'completed':
                print(f"✅ Processing completed after {attempt+1} attempts")
                return data
            else:
                print(f"⏳ Status: {data.get('status')} ({attempt+1}/{max_attempts})")
        except requests.exceptions.RequestException as e:
            print(f"❌ Error checking status: {e}")
        
        time.sleep(delay)
    
    print(f"❌ Processing timed out after {max_attempts * delay} seconds")
    return None

def create_test_file(output_path, size=128):
    """Create a simple test .npy file with a cube inside for testing."""
    print(f"Creating test file at {output_path}...")
    
    # Create a blank 3D volume
    volume = np.zeros((size, size, size), dtype=np.float32)
    
    # Add a cube in the middle
    center = size // 2
    cube_size = size // 4
    
    volume[
        center-cube_size:center+cube_size,
        center-cube_size:center+cube_size, 
        center-cube_size:center+cube_size
    ] = 1.0
    
    # Add a smaller sphere (potential "metastasis")
    x, y, z = np.mgrid[:size, :size, :size]
    sphere_mask = (x - center*1.5)**2 + (y - center*0.8)**2 + (z - center*1.2)**2 <= (cube_size/2)**2
    volume[sphere_mask] = 0.8
    
    # Save the volume
    np.save(output_path, volume)
    print(f"✅ Created test file with shape {volume.shape}")
    return output_path

def test_authentication():
    """Test user authentication flow (register, login, profile, logout)"""
    print("\n==== Testing Authentication Flow ====")
    
    # 1. Register new user
    print("Testing user registration...")
    try:
        response = requests.post(f"{API_URL}/auth/register", json=TEST_USER)
        if response.status_code in (201, 409):  # Created or already exists
            print("✅ Registration test passed")
            
            # If user already exists, we should try to login instead
            if response.status_code == 409:
                print("Note: Test user already exists, will use it for login test")
        else:
            print(f"❌ Registration failed with status {response.status_code}: {response.text}")
            return None
    except RequestException as e:
        print(f"❌ Registration request error: {e}")
        return None
    
    # 2. Login with user
    print("Testing user login...")
    try:
        login_data = {
            "email": TEST_USER["email"],
            "password": TEST_USER["password"]
        }
        response = requests.post(f"{API_URL}/auth/login", json=login_data)
        
        if response.status_code == 200:
            token = response.json().get('token')
            if token:
                print("✅ Login test passed")
                auth_header = {"Authorization": f"Bearer {token}"}
                
                # 3. Test getting user profile
                print("Testing user profile...")
                profile_response = requests.get(f"{API_URL}/user/profile", headers=auth_header)
                if profile_response.status_code == 200:
                    print("✅ Profile retrieval test passed")
                else:
                    print(f"❌ Profile retrieval failed: {profile_response.text}")
                
                # 4. Test getting user's scan history
                print("Testing scan history retrieval...")
                scans_response = requests.get(f"{API_URL}/user/scans", headers=auth_header)
                if scans_response.status_code == 200:
                    print("✅ Scan history retrieval test passed")
                    print(f"  Found {len(scans_response.json().get('scans', []))} scans")
                else:
                    print(f"❌ Scan history retrieval failed: {scans_response.text}")
                
                # 5. Test logout
                print("Testing logout...")
                logout_response = requests.post(f"{API_URL}/auth/logout", headers=auth_header)
                if logout_response.status_code == 200:
                    print("✅ Logout test passed")
                else:
                    print(f"❌ Logout failed: {logout_response.text}")
                
                return auth_header
            else:
                print("❌ Login response missing token")
                return None
        else:
            print(f"❌ Login failed with status {response.status_code}: {response.text}")
            return None
    except RequestException as e:
        print(f"❌ Login request error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Test the MET analysis microservices")
    parser.add_argument("--test-file", help="Path to a test .npy file to use")
    parser.add_argument("--create-test-file", action="store_true", help="Create a test file")
    parser.add_argument("--skip-auth", action="store_true", help="Skip authentication tests")
    parser.add_argument("--test-mode", choices=["all", "auth", "upload", "results"], 
                      default="all", help="Select which tests to run")
    args = parser.parse_args()
    
    print("==== Testing MET Analysis Services ====")
    
    # Check service health
    api_healthy = check_health("API Gateway", API_URL)
    model_healthy = check_health("Model Service", MODEL_SERVICE_URL)
    processing_healthy = check_health("Image Processing Service", IMAGE_PROCESSING_URL)
    user_healthy = check_health("User Service", USER_SERVICE_URL)
    
    all_services_healthy = api_healthy and model_healthy and processing_healthy and user_healthy
    
    if not all_services_healthy:
        print("\n❌ One or more services are not healthy. Make sure all services are running.")
        if not args.skip_auth and not user_healthy:
            print("Use --skip-auth to bypass authentication tests")
        return
    
    # Run auth tests if not skipped
    auth_header = None
    if not args.skip_auth and (args.test_mode in ["all", "auth"]):
        auth_header = test_authentication()
    
    # Skip further tests if only testing auth
    if args.test_mode == "auth":
        return
        
    # Get or create test file
    if args.test_mode in ["all", "upload", "results"]:
        test_file = args.test_file
        
        if args.create_test_file or not test_file:
            test_dir = "./test_data"
            os.makedirs(test_dir, exist_ok=True)
            test_file = create_test_file(os.path.join(test_dir, "test_volume.npy"))
        
        # Test upload
        job_id = test_upload_file(test_file)
        if not job_id:
            return
            
        # Skip results test if only testing upload
        if args.test_mode == "upload":
            return
            
        # Test results
        results = poll_for_results(job_id)
        
        if results:
            print("\n==== Results Summary ====")
            print(f"Metastasis Count: {results.get('metastasis_count', 'N/A')}")
            print(f"Total Volume: {results.get('total_volume', 'N/A')} mm³")
            if results.get('metastasis_volumes'):
                print("Individual Volumes:")
                for i, vol in enumerate(results.get('metastasis_volumes')):
                    print(f"  - Metastasis {i+1}: {vol:.2f} mm³")
            
            print("\n✅ End-to-end test completed successfully!")

if __name__ == "__main__":
    main()
