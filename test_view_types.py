#!/usr/bin/env python3
"""
Test script to verify that view type changes work correctly
"""
import requests
import json

# Test the visualization endpoint with different view types
base_url = "http://localhost:8000"
job_id = "90c4604c-43e2-4c2d-b899-3c5c87f89905"  # From the logs

def test_view_type(view_type, slice_idx=63):
    """Test a specific view type"""
    url = f"{base_url}/visualization/{job_id}"
    params = {
        "quality": "high",
        "upscale": "1.2",
        "enhance_contrast": "true",
        "enhance_edges": "true",
        "type": "slice",
        "view_type": view_type,
        "slice_idx": slice_idx,
        "brightness": "1.4",
        "contrast": "1.6",
        "show_original": "true"
    }
    
    print(f"\nTesting view_type: {view_type}")
    print(f"URL: {url}")
    print(f"Params: {params}")
    
    try:
        response = requests.get(url, params=params, timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            content_type = response.headers.get('content-type', '')
            print(f"Content-Type: {content_type}")
            print(f"Response Size: {len(response.content)} bytes")
            
            # Save the image for verification
            filename = f"test_view_{view_type}_slice_{slice_idx}.png"
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Saved image: {filename}")
            
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    print("Testing different view types...")
    
    # Test all three view types
    view_types = ["axial", "coronal", "sagittal"]
    
    for view_type in view_types:
        test_view_type(view_type, slice_idx=63)
    
    print("\nTest completed!")
