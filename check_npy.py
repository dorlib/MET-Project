import numpy as np
import os
import sys

def check_npy_files(directory):
    print(f"Checking NPY files in {directory}")
    files = [f for f in os.listdir(directory) if f.endswith(".npy")]
    print(f"Found {len(files)} NPY files")
    if not files:
        print("No NPY files found")
        return
    
    print(f"First 5 files: {files[:5]}")
    for file in files[:3]:  # Check first 3 files
        path = os.path.join(directory, file)
        try:
            data = np.load(path)
            print(f"File: {file}, Shape: {data.shape}, Type: {data.dtype}")
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")

if __name__ == "__main__":
    check_npy_files("/app/uploads")
    print("\nAlso checking results directory:")
    check_npy_files("/app/results")

