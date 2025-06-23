#!/usr/bin/env python3
import requests
import time
import os
import sys
from pathlib import Path

def test_new_visualizations():
    # Path to test file
    test_file_path = "resources/image_7_00009.npy"
    
    if not os.path.exists(test_file_path):
        print(f"Test file not found: {test_file_path}")
        sys.exit(1)
    
    # Upload the scan
    print(f"Uploading test scan {test_file_path}...")
    
    files = {'file': (Path(test_file_path).name, open(test_file_path, 'rb'), 'application/octet-stream')}
    response = requests.post('http://localhost:8000/upload', files=files)
    
    if not response.ok:
        print(f"Upload failed with status {response.status_code}: {response.text}")
        sys.exit(1)
    
    # Get job ID
    job_id = response.json().get("job_id")
    print(f"Upload successful. Job ID: {job_id}")
    
    # Poll for status until job is completed
    max_attempts = 60
    attempts = 0
    
    while attempts < max_attempts:
        attempts += 1
        status_response = requests.get(f'http://localhost:8000/status/{job_id}')
        
        if not status_response.ok:
            print(f"Failed to check status: {status_response.status_code}")
            time.sleep(1)
            continue
        
        status = status_response.json().get("status")
        print(f"Attempt {attempts}: Status = {status}")
        
        if status == "completed":
            break
            
        if status == "failed" or status == "not_found":
            print(f"Job processing failed or not found")
            sys.exit(1)
            
        time.sleep(1)
    
    if attempts >= max_attempts:
        print("Timed out waiting for job to complete")
        sys.exit(1)
    
    # Get results to validate metastasis detection
    results_response = requests.get(f'http://localhost:8000/results/{job_id}')
    
    if not results_response.ok:
        print(f"Failed to get results: {results_response.status_code}: {results_response.text}")
        sys.exit(1)
        
    results_data = results_response.json()
    print(f"Results: {results_data}")
    
    # Test the new visualization endpoints
    print("\nTesting traditional overlay visualization:")
    overlay_url = f"http://localhost:8000/visualization/{job_id}?type=slice&quality=high&slice_idx=64"
    overlay_response = requests.get(overlay_url)
    print(f"Overlay visualization response: Status {overlay_response.status_code}")
    
    print("\nTesting new side-by-side visualization:")
    side_by_side_url = f"http://localhost:8000/side-by-side-visualization/{job_id}?slice_idx=64"
    side_by_side_response = requests.get(side_by_side_url)
    print(f"Side-by-side visualization response: Status {side_by_side_response.status_code}")
    
    print("\nTesting new three-plane visualization:")
    three_plane_url = f"http://localhost:8000/three-plane-visualization/{job_id}"
    three_plane_response = requests.get(three_plane_url)
    print(f"Three-plane visualization response: Status {three_plane_response.status_code}")
    
    print("\nComplete URLs for browser testing:")
    print(f"1. Original overlay: {overlay_url}")
    print(f"2. Side-by-side: {side_by_side_url}")
    print(f"3. Three-plane: {three_plane_url}")
    
    print("\nTest completed!")
    
if __name__ == "__main__":
    test_new_visualizations()
