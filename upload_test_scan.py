#!/usr/bin/env python3
import requests
import os
import json
import time

# Configuration
API_URL = "http://localhost:8000"
AUTH_URL = f"{API_URL}/auth/login"
UPLOAD_URL = f"{API_URL}/upload"
NPY_FILE_PATH = "/tmp/brain_mri_test.npy"

# Credentials (for demo purposes - in production use environment variables)
EMAIL = "test@example.com"
PASSWORD = "test123"

def main():
    print("Starting test scan upload...")
    
    # Step 1: Login to get token
    print("\n1. Authenticating...")
    auth_response = requests.post(AUTH_URL, json={
        "email": EMAIL,
        "password": PASSWORD
    })
    
    if not auth_response.ok:
        print(f"Authentication failed: {auth_response.status_code}, {auth_response.text}")
        return
    
    auth_data = auth_response.json()
    token = auth_data.get("access_token")
    print(f"Authentication successful!")
    
    # Step 2: Upload the NPY file
    headers = {"Authorization": f"Bearer {token}"}
    
    print("\n2. Uploading NPY file...")
    with open(NPY_FILE_PATH, "rb") as f:
        files = {"file": (os.path.basename(NPY_FILE_PATH), f, "application/octet-stream")}
        upload_response = requests.post(
            UPLOAD_URL, 
            files=files,
            data={"scan_type": "t1ce", "patient_id": "test_patient"},
            headers=headers
        )
    
    if not upload_response.ok:
        print(f"Upload failed: {upload_response.status_code}, {upload_response.text}")
        return
    
    upload_data = upload_response.json()
    job_id = upload_data.get("job_id")
    print(f"Upload successful! Job ID: {job_id}")
    
    # Step 3: Poll for results (optional)
    print("\n3. Waiting for processing...")
    result_url = f"{API_URL}/results/{job_id}"
    max_attempts = 10
    attempt = 0
    
    while attempt < max_attempts:
        time.sleep(2)
        attempt += 1
        print(f"Checking status (attempt {attempt}/{max_attempts})...")
        
        result_response = requests.get(result_url, headers=headers)
        if result_response.ok:
            result_data = result_response.json()
            status = result_data.get("status")
            print(f"Status: {status}")
            print(f"Result data keys: {list(result_data.keys() if result_data else {})}")
            
            if status == "COMPLETED":
                print(f"\nProcessing complete! Result data:")
                print(json.dumps(result_data, indent=2))
                
                # Check for visualization URL
                if "segmentation_path" in result_data:
                    print(f"\nSegmentation path: {result_data.get('segmentation_path')}")
                    vis_url = f"{API_URL}/visualization/{job_id}?slice_idx=64&axis=0"
                    print(f"Trying to access visualization URL: {vis_url}")
                    vis_response = requests.get(vis_url, headers=headers)
                    print(f"Visualization response: {vis_response.status_code}")
                    
                    # Try advanced visualization too
                    adv_vis_url = f"{API_URL}/advanced-visualization/{job_id}"
                    print(f"Trying to access advanced visualization URL: {adv_vis_url}")
                    adv_vis_response = requests.get(adv_vis_url, headers=headers)
                    print(f"Advanced visualization response: {adv_vis_response.status_code}")
                break
        else:
            print(f"Failed to get status: {result_response.status_code}, {result_response.text}")
    
    print("\nTest complete!")

if __name__ == "__main__":
    main()
