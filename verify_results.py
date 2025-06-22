#!/usr/bin/env python3
import requests
import json
import sys

# Configuration
API_URL = "http://localhost:8000"
AUTH_URL = f"{API_URL}/auth/login"
JOB_ID = "752eeb92-bc3a-499e-bcaf-a63225b62660"  # Use your actual job_id here

# Credentials
EMAIL = "test@example.com"
PASSWORD = "test123"

def main():
    print(f"Starting verification with API URL: {API_URL} and Job ID: {JOB_ID}")
    sys.stdout.flush()
    
    # Login
    auth_response = requests.post(AUTH_URL, json={
        "email": EMAIL,
        "password": PASSWORD
    })
    
    if not auth_response.ok:
        print(f"Authentication failed: {auth_response.status_code}, {auth_response.text}")
        return
    
    print("Authentication successful")
    sys.stdout.flush()
    
    token = auth_response.json().get("access_token")
    headers = {"Authorization": f"Bearer {token}"}
    
    # Get results
    result_url = f"{API_URL}/results/{JOB_ID}"
    result_response = requests.get(result_url, headers=headers)
    
    if result_response.ok:
        result_data = result_response.json()
        print("Results API Response:")
        print(json.dumps(result_data, indent=2))
        
        # Check segmentation path
        segmentation_path = result_data.get("segmentation_path")
        print(f"\nSegmentation path: {segmentation_path}")
        
        # Test visualization
        if segmentation_path:
            vis_url = f"{API_URL}/visualization/{JOB_ID}?slice_idx=64&axis=0"
            print(f"\nVisualization URL: {vis_url}")
            vis_response = requests.get(vis_url, headers=headers)
            print(f"Visualization response status: {vis_response.status_code}")
            print(f"Visualization content type: {vis_response.headers.get('Content-Type')}")
            
            # Try advanced visualization
            adv_vis_url = f"{API_URL}/advanced-visualization/{JOB_ID}"
            print(f"\nAdvanced visualization URL: {adv_vis_url}")
            try:
                adv_vis_response = requests.get(adv_vis_url, headers=headers, timeout=10)
                print(f"Advanced visualization response status: {adv_vis_response.status_code}")
                print(f"Advanced visualization content type: {adv_vis_response.headers.get('Content-Type')}")
            except Exception as e:
                print(f"Exception during advanced visualization request: {e}")
    else:
        print(f"Failed to get results: {result_response.status_code}, {result_response.text}")

if __name__ == "__main__":
    main()
