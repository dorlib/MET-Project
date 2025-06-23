#!/usr/bin/env python3
import requests
import time
import os
import sys
from pathlib import Path

def test_upload_and_visualization():
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
        print(f"Failed to get results: {results_response.status_code}")
        sys.exit(1)
        
    results_data = results_response.json()
    print(f"Results: {results_data}")
    
    # Get visualization URLs for different slices to verify the enhanced overlay
    print("\nVisualization URLs to test the enhanced overlay (alpha=0.8):")
    
    # Middle slice with high quality
    viz_url_middle = f"http://localhost:8000/visualization/{job_id}?type=slice&quality=high&slice_idx=64&upscale=2&enhance_contrast=true&enhance_edges=true"
    print(f"Middle slice visualization: {viz_url_middle}")
    
    # Additional slices to test overlay visibility
    slice_indices = [32, 48, 80, 96]
    for idx in slice_indices:
        slice_url = f"http://localhost:8000/visualization/{job_id}?type=slice&quality=high&slice_idx={idx}&upscale=2&enhance_contrast=true&enhance_edges=true"
        print(f"Slice {idx} visualization: {slice_url}")
        
    print("\nThese visualizations should show improved color overlay with alpha=0.8 for better visibility.")
    print("Open these URLs in your browser to verify the segmentation overlay.")
    
    print("\nTest completed successfully!")
    
if __name__ == "__main__":
    test_upload_and_visualization()
