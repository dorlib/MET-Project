#!/usr/bin/env python3
"""
Test script to upload a scan and verify confidence metrics are included
"""
import requests
import time
import json

def test_upload_with_confidence():
    # Upload the real scan file
    scan_file_path = "resources/image_7_00009.npy"
    
    try:
        # Upload scan
        with open(scan_file_path, 'rb') as f:
            files = {'file': f}
            data = {
                'patient_name': 'Test Patient Confidence',
                'scan_date': '2024-01-15'
            }
            
            print("Uploading scan...")
            response = requests.post('http://localhost:8000/upload', files=files, data=data)
            print(f"Upload status: {response.status_code}")
            print(f"Upload response: {response.json()}")
            
            if response.status_code != 200:
                print("Upload failed!")
                return
                
            job_id = response.json().get('job_id')
            print(f"Job ID: {job_id}")
            
            # Wait for processing to complete
            print("Waiting for processing...")
            max_wait = 60  # Maximum wait time in seconds
            wait_time = 0
            
            while wait_time < max_wait:
                time.sleep(5)
                wait_time += 5
                
                # Check results
                result_response = requests.get(f'http://localhost:8000/results/{job_id}')
                result_data = result_response.json()
                
                print(f"Status after {wait_time}s: {result_data.get('status', 'unknown')}")
                
                if result_data.get('status') == 'completed':
                    print("Processing completed!")
                    print("=== SCAN RESULTS WITH CONFIDENCE METRICS ===")
                    print(json.dumps(result_data, indent=2))
                    
                    # Check if confidence metrics are included
                    confidence_metrics = result_data.get('confidence_metrics', {})
                    if confidence_metrics:
                        print("\n=== CONFIDENCE METRICS FOUND ===")
                        print(f"Overall Confidence: {confidence_metrics.get('overall_confidence', 'N/A')}")
                        print(f"Volume Consistency: {confidence_metrics.get('volume_consistency', 'N/A')}")
                        
                        class_confidence = confidence_metrics.get('class_confidence', {})
                        if class_confidence:
                            print("Class-specific confidence:")
                            for class_id, confidence in class_confidence.items():
                                print(f"  Class {class_id}: {confidence:.3f}")
                    else:
                        print("\n❌ NO CONFIDENCE METRICS FOUND")
                    
                    return
                    
                elif result_data.get('status') in ['failed', 'error']:
                    print(f"Processing failed: {result_data}")
                    return
                    
            print("Processing timed out")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_upload_with_confidence()
