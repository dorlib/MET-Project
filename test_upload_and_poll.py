#!/usr/bin/env python3
"""
Test script to verify the upload, polling, and results visualization flow
"""
import os
import sys
import time
import requests
import json
import argparse
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Test MET project upload and polling flow')
    parser.add_argument('--file', '-f', type=str, default='resources/image_7_00009.npy',
                        help='Path to the .npy file to upload')
    parser.add_argument('--api-url', '-u', type=str, default='http://localhost:8000',
                        help='API Gateway URL')
    parser.add_argument('--poll-interval', '-i', type=float, default=2.0,
                        help='Polling interval in seconds')
    parser.add_argument('--timeout', '-t', type=int, default=300,
                        help='Maximum time to wait for processing (in seconds)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    return parser.parse_args()

def log(msg, verbose=True):
    timestamp = datetime.now().strftime('%H:%M:%S')
    if verbose:
        print(f"[{timestamp}] {msg}")
    sys.stdout.flush()

def upload_file(api_url, file_path, verbose=True):
    """Upload a file to the API gateway"""
    log(f"Uploading file: {file_path}", verbose)
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'application/octet-stream')}
            response = requests.post(f"{api_url}/upload", files=files)
            
            if not response.ok:
                log(f"Upload failed with status {response.status_code}: {response.text}", verbose)
                return None
            
            result = response.json()
            log(f"Upload successful. Job ID: {result.get('job_id')}", verbose)
            log(f"Response: {json.dumps(result, indent=2)}", verbose)
            return result
    except Exception as e:
        log(f"Error uploading file: {str(e)}", verbose)
        return None

def poll_status(api_url, job_id, poll_interval=2.0, timeout=300, verbose=True):
    """Poll for job status until completion or timeout"""
    log(f"Polling for status of job {job_id} (timeout: {timeout}s)", verbose)
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{api_url}/status/{job_id}")
            
            if not response.ok:
                log(f"Status check failed: {response.status_code}", verbose)
                time.sleep(poll_interval)
                continue
                
            result = response.json()
            status = result.get('status')
            log(f"Current status: {status}", verbose)
            
            if status in ['completed', 'failed', 'error']:
                return result
                
            # If still processing, wait and try again
            time.sleep(poll_interval)
            
        except Exception as e:
            log(f"Error checking status: {str(e)}", verbose)
            time.sleep(poll_interval)
    
    log(f"Polling timed out after {timeout} seconds", verbose)
    return {'status': 'timeout', 'error': f"Polling timed out after {timeout} seconds"}

def get_results(api_url, job_id, verbose=True):
    """Get results for a completed job"""
    log(f"Retrieving results for job {job_id}", verbose)
    
    try:
        response = requests.get(f"{api_url}/results/{job_id}")
        
        if not response.ok:
            log(f"Results retrieval failed: {response.status_code}", verbose)
            return None
            
        result = response.json()
        log(f"Results retrieved: {json.dumps(result, indent=2)}", verbose)
        return result
    except Exception as e:
        log(f"Error retrieving results: {str(e)}", verbose)
        return None

def verify_health(api_url, verbose=True):
    """Check API health endpoints"""
    log("Checking API health...", verbose)
    
    endpoints = [
        "/health",
        "/model/health"
    ]
    
    all_healthy = True
    for endpoint in endpoints:
        try:
            url = f"{api_url}{endpoint}"
            log(f"Checking endpoint: {url}", verbose)
            response = requests.get(url)
            
            if response.ok:
                log(f"✅ {endpoint} is healthy: {response.json()}", verbose)
            else:
                log(f"❌ {endpoint} returned status {response.status_code}: {response.text}", verbose)
                all_healthy = False
                
        except Exception as e:
            log(f"❌ Error checking {endpoint}: {str(e)}", verbose)
            all_healthy = False
            
    return all_healthy

def main():
    args = parse_args()
    
    # Verify API health
    api_health = verify_health(args.api_url, args.verbose)
    if not api_health:
        log("WARNING: One or more API health checks failed", True)
    
    # Upload a file
    upload_result = upload_file(args.api_url, args.file, args.verbose)
    if not upload_result:
        log("ERROR: File upload failed", True)
        return 1
    
    job_id = upload_result.get('job_id')
    if not job_id:
        log("ERROR: No job ID returned from upload", True)
        return 1
    
    # Poll for status
    status_result = poll_status(
        args.api_url, 
        job_id, 
        args.poll_interval, 
        args.timeout,
        args.verbose
    )
    
    if status_result.get('status') == 'completed':
        log("✅ Job completed successfully!", True)
        
        # Get results
        results = get_results(args.api_url, job_id, args.verbose)
        if results:
            log("✅ Results retrieved successfully!", True)
            
            # Check if visualization URL is included
            if 'visualization_url' in results:
                log(f"Visualization URL: {results['visualization_url']}", True)
            
            return 0
        else:
            log("❌ Failed to retrieve results", True)
            return 1
    else:
        log(f"❌ Job did not complete successfully. Final status: {status_result.get('status')}", True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
