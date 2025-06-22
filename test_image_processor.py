#!/usr/bin/env python3
# test_image_processor.py - Standalone test for enhanced image processing features

import os
import sys
import numpy as np
import requests
import matplotlib.pyplot as plt
from PIL import Image
import io
import argparse
import json
import nibabel as nib

def load_volume(file_path):
    """Load a volume from .npy or .nii/.nii.gz file"""
    if file_path.endswith('.npy'):
        return np.load(file_path)
    elif file_path.endswith('.nii.gz') or file_path.endswith('.nii'):
        nii_img = nib.load(file_path)
        return np.asarray(nii_img.dataobj)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def simulate_segmentation(input_volume, num_classes=3):
    """
    Simulate a segmentation mask for testing
    For real cases, this would be the output of a model prediction
    """
    print("Generating simulated segmentation mask...")
    
    # Create an empty segmentation mask
    segmentation = np.zeros_like(input_volume, dtype=np.uint8)
    
    # Generate random blobs for each class
    for class_id in range(1, num_classes + 1):
        # Number of random spheres to create
        num_spheres = np.random.randint(1, 5)
        
        for i in range(num_spheres):
            # Random center and radius
            center_z = np.random.randint(10, segmentation.shape[0] - 10)
            center_y = np.random.randint(10, segmentation.shape[1] - 10)
            center_x = np.random.randint(10, segmentation.shape[2] - 10)
            
            radius = np.random.randint(3, 10)
            
            # Create a sphere
            z, y, x = np.ogrid[:segmentation.shape[0], :segmentation.shape[1], :segmentation.shape[2]]
            distance = np.sqrt((z - center_z)**2 + (y - center_y)**2 + (x - center_x)**2)
            mask = distance <= radius
            
            # Assign class_id to the sphere
            segmentation[mask] = class_id
    
    print(f"Generated segmentation with {num_classes} classes")
    return segmentation

def save_visualization(image, title, output_file):
    """Save an image with a title"""
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Saved visualization to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Test enhanced image processing features')
    parser.add_argument('input_file', help='Input volume file (.npy, .nii, or .nii.gz)')
    parser.add_argument('--output_dir', default='./test_outputs', help='Output directory for results')
    parser.add_argument('--service_url', default='http://localhost:5002', help='URL for the image processing service')
    parser.add_argument('--local_only', action='store_true', help='Run local processing only, no HTTP requests')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load the input volume
        print(f"Loading input volume: {args.input_file}")
        input_volume = load_volume(args.input_file)
        print(f"Input shape: {input_volume.shape}, min: {input_volume.min()}, max: {input_volume.max()}")
        
        # Generate a simulated segmentation mask
        segmentation = simulate_segmentation(input_volume)
        
        # Save the segmentation for reference
        seg_path = os.path.join(args.output_dir, 'simulated_segmentation.npy')
        np.save(seg_path, segmentation)
        print(f"Saved simulated segmentation to {seg_path}")
        
        # Create a simple visualization of the input and segmentation
        slice_idx = segmentation.shape[0] // 2  # Middle slice
        
        # Save a visualization of the input
        save_visualization(
            input_volume[slice_idx], 
            f"Input Volume (Slice {slice_idx})",
            os.path.join(args.output_dir, 'input_slice.png')
        )
        
        # Create a colored segmentation visualization
        colors = np.zeros((*segmentation[slice_idx].shape, 3))
        tissue_colors = {
            1: (1.0, 0.0, 0.0),  # Red
            2: (0.0, 1.0, 0.0),  # Green
            3: (0.0, 0.0, 1.0)   # Blue
        }
        
        for class_id, color in tissue_colors.items():
            mask = segmentation[slice_idx] == class_id
            if np.any(mask):
                for i, c in enumerate(color):
                    colors[mask, i] = c
        
        # Save the segmentation visualization
        save_visualization(
            colors,
            f"Segmentation (Slice {slice_idx})",
            os.path.join(args.output_dir, 'segmentation_slice.png')
        )
        
        if not args.local_only:
            # Test the image processing service endpoints
            print("\nTesting image processing service endpoints...")
            
            # First, check if the service is available
            try:
                health_response = requests.get(f"{args.service_url}/health", timeout=5)
                if health_response.status_code != 200:
                    print(f"Warning: Service health check failed with status {health_response.status_code}")
                    print("Continuing with local tests only...")
                    args.local_only = True
                else:
                    print("Service health check passed")
            except requests.exceptions.RequestException as e:
                print(f"Warning: Service not available ({str(e)})")
                print("Continuing with local tests only...")
                args.local_only = True
        
        if not args.local_only:
            # Use a temporary job ID for testing
            job_id = "test_job_" + os.path.basename(args.input_file).split('.')[0]
            
            # Save the input and segmentation to the expected locations for the service
            # Note: In a real scenario, these paths would be determined by the service
            service_results_dir = os.path.join(args.output_dir, 'service_results')
            os.makedirs(service_results_dir, exist_ok=True)
            
            np.save(os.path.join(service_results_dir, f"{job_id}_original.npy"), input_volume)
            np.save(os.path.join(service_results_dir, f"{job_id}_prediction.npy"), segmentation)
            
            # Test advanced analysis endpoint
            print("\nTesting advanced analysis...")
            analysis_data = {
                'segmentation': segmentation.tolist(),
                'original_image': input_volume.tolist(),
                'voxel_volume_mm3': 1.0
            }
            headers = {'Content-Type': 'application/json'}
            
            try:
                response = requests.post(
                    f"{args.service_url}/advanced-analysis/{job_id}",
                    json=analysis_data,
                    headers=headers
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print("Advanced analysis success!")
                    print(f"Classes found: {result.get('total_classes_found')}")
                    print(f"Total regions: {result.get('overall_summary', {}).get('total_regions')}")
                    
                    # Save the result
                    with open(os.path.join(args.output_dir, 'advanced_analysis.json'), 'w') as f:
                        json.dump(result, f, indent=2)
                else:
                    print(f"Advanced analysis failed: {response.status_code}")
                    print(response.text)
            except requests.exceptions.RequestException as e:
                print(f"Request error: {str(e)}")
            
            # Test visualization endpoint with various types
            for viz_type in ['slice', 'projection', 'multi-slice', 'lesions']:
                print(f"\nTesting {viz_type} visualization...")
                try:
                    response = requests.get(
                        f"{args.service_url}/visualization/{job_id}?type={viz_type}"
                    )
                    
                    if response.status_code == 200:
                        # Save the visualization image
                        img = Image.open(io.BytesIO(response.content))
                        img.save(os.path.join(args.output_dir, f"{viz_type}_visualization.png"))
                        print(f"Saved {viz_type} visualization")
                    else:
                        print(f"{viz_type.capitalize()} visualization failed: {response.status_code}")
                        print(response.text)
                except requests.exceptions.RequestException as e:
                    print(f"Request error: {str(e)}")
        
        print("\nTest completed successfully!")
        print(f"Output files are in: {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
