#!/usr/bin/env python3
import requests
import time
import os
import sys
from pathlib import Path
import json

def test_fix():
    # Create a new job
    print("Testing with a new job...")
    job_id = create_new_job()
    
    if not job_id:
        print("Failed to create a new job.")
        sys.exit(1)
    
    print(f"Successfully created job with ID: {job_id}")
    print("Waiting for job to complete...")
    
    # Poll for completion - increased timeout
    max_attempts = 120  # Wait up to 2 minutes
    attempts = 0
    
    while attempts < max_attempts:
        attempts += 1
        time.sleep(1)
        
        try:
            results_response = requests.get(f'http://localhost:8000/results/{job_id}')
            if results_response.ok:
                results = results_response.json()
                if results.get("status") == "completed":
                    print(f"Job completed! Results: {json.dumps(results, indent=2)}")
                    break
            else:
                print(f"Attempt {attempts}: Status check returned {results_response.status_code}")
        except Exception as e:
            print(f"Attempt {attempts}: Error checking status: {str(e)}")
    
    if attempts >= max_attempts:
        print("Timed out waiting for job to complete.")
        sys.exit(1)
    
    # Now test all visualization endpoints
    print("\nTesting visualization endpoints...")
    
    # Test standard visualization
    print("1. Testing standard visualization...")
    viz_response = requests.get(f'http://localhost:8000/visualization/{job_id}', timeout=60)
    if viz_response.ok:
        with open("standard_viz.png", "wb") as f:
            f.write(viz_response.content)
        print("  ✓ Standard visualization saved to standard_viz.png")
    else:
        print(f"  ✗ Standard visualization failed: {viz_response.status_code}")
    
    # Test side-by-side visualization
    print("2. Testing side-by-side visualization...")
    side_by_side_response = requests.get(f'http://localhost:8000/side-by-side-visualization/{job_id}', timeout=60)
    if side_by_side_response.ok:
        with open("side_by_side.png", "wb") as f:
            f.write(side_by_side_response.content)
        print("  ✓ Side-by-side visualization saved to side_by_side.png")
    else:
        print(f"  ✗ Side-by-side visualization failed: {side_by_side_response.status_code}")
    
    # Test three-plane visualization
    print("3. Testing three-plane visualization...")
    three_plane_response = requests.get(f'http://localhost:8000/three-plane-visualization/{job_id}', timeout=60)
    if three_plane_response.ok:
        with open("three_plane.png", "wb") as f:
            f.write(three_plane_response.content)
        print("  ✓ Three-plane visualization saved to three_plane.png")
    else:
        print(f"  ✗ Three-plane visualization failed: {three_plane_response.status_code}")
    
    print("\nTest completed!")

def create_new_job():
    """Create a new job by uploading a test scan"""
    # Path to test file
    test_file_path = "resources/image_7_00009.npy"
    
    if not os.path.exists(test_file_path):
        print(f"Test file not found: {test_file_path}")
        return None
    
    # Upload the scan
    print(f"Uploading test scan {test_file_path}...")
    
    try:
        files = {'file': (Path(test_file_path).name, open(test_file_path, 'rb'), 'application/octet-stream')}
        response = requests.post('http://localhost:8000/upload', files=files)
        
        if not response.ok:
            print(f"Upload failed with status {response.status_code}: {response.text}")
            return None
        
        # Get job ID
        job_id = response.json().get("job_id")
        return job_id
    except Exception as e:
        print(f"Error creating new job: {str(e)}")
        return None

if __name__ == "__main__":
    test_fix()
