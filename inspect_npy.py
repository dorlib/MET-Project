#!/usr/bin/env python3
"""
NPY File Inspector - A tool to inspect and visualize NPY files used in the MET project
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

def inspect_npy_file(file_path):
    """Inspect and display information about an NPY file"""
    try:
        # Load the NPY file
        data = np.load(file_path)
        
        # Display basic info
        print(f"\n=== NPY File Information: {os.path.basename(file_path)} ===")
        print(f"Shape: {data.shape}")
        print(f"Data Type: {data.dtype}")
        print(f"Min Value: {data.min()}")
        print(f"Max Value: {data.max()}")
        print(f"Mean Value: {data.mean():.4f}")
        print(f"Standard Deviation: {data.std():.4f}")
        
        # Handle different types of data
        if data.ndim == 3:  # Single-channel 3D volume
            print("Type: 3D Volume (single channel)")
            unique_vals = np.unique(data)
            print(f"Unique Values: {unique_vals}")
            
            # If it's a mask/segmentation (few unique integers)
            if len(unique_vals) < 10 and np.issubdtype(data.dtype, np.integer):
                print("This appears to be a segmentation mask")
                for val in unique_vals:
                    count = np.sum(data == val)
                    percentage = 100.0 * count / data.size
                    print(f"  Class {val}: {count} voxels ({percentage:.2f}%)")
            
        elif data.ndim == 4:  # Multi-channel 3D volume
            print(f"Type: Multi-channel 3D Volume ({data.shape[-1]} channels)")
            
            # Check if it's one-hot encoded mask
            if data.shape[-1] <= 10:
                sums = np.sum(data, axis=-1)
                if np.allclose(sums[sums > 0], 1.0):
                    print("This appears to be a one-hot encoded segmentation mask")
                    for i in range(data.shape[-1]):
                        count = np.sum(data[..., i] > 0.5)
                        percentage = 100.0 * count / data[..., i].size
                        print(f"  Class {i}: {count} voxels ({percentage:.2f}%)")
        
        # Visualize middle slices
        plt.figure(figsize=(15, 5))
        
        # For 3D volume
        if data.ndim == 3:
            z_mid = data.shape[2] // 2
            y_mid = data.shape[1] // 2
            x_mid = data.shape[0] // 2
            
            plt.subplot(1, 3, 1)
            plt.imshow(data[:, :, z_mid], cmap='gray')
            plt.title(f'Axial (z={z_mid})')
            plt.colorbar()
            
            plt.subplot(1, 3, 2)
            plt.imshow(data[:, y_mid, :], cmap='gray')
            plt.title(f'Coronal (y={y_mid})')
            plt.colorbar()
            
            plt.subplot(1, 3, 3)
            plt.imshow(data[x_mid, :, :], cmap='gray')
            plt.title(f'Sagittal (x={x_mid})')
            plt.colorbar()
        
        # For multi-channel 3D volume
        elif data.ndim == 4:
            z_mid = data.shape[2] // 2
            
            # Show middle slice for each channel (up to 3)
            channels_to_show = min(3, data.shape[3])
            for i in range(channels_to_show):
                plt.subplot(1, 3, i+1)
                plt.imshow(data[:, :, z_mid, i], cmap='gray')
                plt.title(f'Channel {i}, Axial z={z_mid}')
                plt.colorbar()
        
        plt.tight_layout()
        output_path = os.path.splitext(file_path)[0] + "_inspection.png"
        plt.savefig(output_path)
        print(f"\nVisualization saved to: {output_path}")
        plt.close()
        
        return True
    
    except Exception as e:
        print(f"Error inspecting file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="NPY File Inspector for MET Project")
    parser.add_argument("file_path", help="Path to the NPY file to inspect")
    args = parser.parse_args()
    
    if not os.path.exists(args.file_path):
        print(f"Error: File not found: {args.file_path}")
        return 1
    
    if not args.file_path.endswith('.npy'):
        print(f"Error: File does not appear to be an NPY file: {args.file_path}")
        return 1
    
    success = inspect_npy_file(args.file_path)
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
