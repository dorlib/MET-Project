#!/usr/bin/env python3
"""
Test script to verify slice bounds are correct for each view type
"""
import requests

base_url = "http://localhost:8000"
job_id = "90c4604c-43e2-4c2d-b899-3c5c87f89905"

def test_slice_bounds(view_type, slice_indices):
    """Test slice bounds for a specific view type"""
    print(f"\n=== Testing {view_type.upper()} view bounds ===")
    
    for slice_idx in slice_indices:
        url = f"{base_url}/visualization/{job_id}"
        params = {
            "quality": "high",
            "type": "slice",
            "view_type": view_type,
            "slice_idx": slice_idx
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            status = "✓ SUCCESS" if response.status_code == 200 else f"✗ ERROR {response.status_code}"
            size = len(response.content) if response.status_code == 200 else 0
            print(f"  Slice {slice_idx:3d}: {status} ({size:,} bytes)")
            
        except Exception as e:
            print(f"  Slice {slice_idx:3d}: ✗ EXCEPTION - {e}")

if __name__ == "__main__":
    print("Testing slice bounds for different view types...")
    print("Expected: 128x128x128 volume, so valid range should be 0-127 for all axes")
    
    # Test boundary slices for each view type
    test_indices = [0, 1, 63, 126, 127, 128]  # Include invalid slice 128
    
    test_slice_bounds("axial", test_indices)
    test_slice_bounds("coronal", test_indices)  
    test_slice_bounds("sagittal", test_indices)
    
    print("\nTesting completed!")
