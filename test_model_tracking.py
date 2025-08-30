#!/usr/bin/env python3
"""
Test script to create a simple NPY file for testing model tracking
"""
import numpy as np
import os

# Create a simple 3D array that simulates a brain scan
test_data = np.random.rand(64, 64, 32).astype(np.float32)

# Save as NPY file
output_file = "test_model_tracking.npy"
np.save(output_file, test_data)

print(f"Created test file: {output_file}")
print(f"Shape: {test_data.shape}")
print(f"Size: {os.path.getsize(output_file)} bytes")
