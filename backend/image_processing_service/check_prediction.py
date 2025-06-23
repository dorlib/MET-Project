#!/usr/bin/env python3
import numpy as np
import glob

def analyze_prediction_files():
    files = glob.glob('/app/results/*_prediction.npy')
    print(f'Found {len(files)} prediction files')
    
    for f in files[-3:]:
        try:
            data = np.load(f)
            print(f'File: {f}')
            print(f'Shape: {data.shape}')
            print(f'Data type: {data.dtype}')
            print(f'Unique values: {np.unique(data)}')
            print(f'Class 1 count: {np.sum(data==1)}')
            print(f'Class 2 count: {np.sum(data==2)}')
            print(f'Percentage of voxels with class 1: {(np.sum(data==1) / data.size) * 100:.4f}%')
            print(f'Percentage of voxels with class 2: {(np.sum(data==2) / data.size) * 100:.4f}%')
            print('-' * 50)
        except Exception as e:
            print(f'Error loading {f}: {e}')

if __name__ == "__main__":
    analyze_prediction_files()
