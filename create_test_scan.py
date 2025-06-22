#!/usr/bin/env python3
import numpy as np

# Create a simple 3D array to simulate an MRI scan
# Size: 128x128x128
scan_data = np.random.rand(128, 128, 128)

# Add some structure to make it look like brain tissue
# Create a sphere in the middle
x, y, z = np.ogrid[:128, :128, :128]
center_x, center_y, center_z = 64, 64, 64
radius = 50
sphere = (x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2 <= radius**2
scan_data[sphere] += 0.5

# Normalize to 0-1 range
scan_data = (scan_data - scan_data.min()) / (scan_data.max() - scan_data.min())

# Save as NPY file
output_path = '/tmp/brain_mri_test.npy'
np.save(output_path, scan_data)

print(f"Created test MRI file: {output_path}")
print(f"Shape: {scan_data.shape}, Min: {scan_data.min()}, Max: {scan_data.max()}")
