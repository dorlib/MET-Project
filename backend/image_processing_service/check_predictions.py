import numpy as np
import os
import glob

# Find the latest prediction file
latest_file = max(glob.glob('/app/results/*_prediction.npy'), key=os.path.getctime)
print(f'Examining file: {latest_file}')

# Load the data
data = np.load(latest_file)
print(f'Shape: {data.shape}')
print(f'Unique values: {np.unique(data)}')
print(f'Value counts: {np.unique(data, return_counts=True)[1]}')

# Check presence of classes in slices at 10-slice intervals
print('Classes present by slice:')
for i in range(0, data.shape[0], 10):
    present = np.unique(data[i])
    print(f'  Slice {i}: {present}')

# Sample specific slices where we expect to find class 1 (metastasis)
print('Checking slices with potential metastases:')
for i in [40, 50, 60, 70, 80]:  # Sample slices to check
    classes_in_slice = np.unique(data[i])
    if 1 in classes_in_slice:
        # Count pixels of each class
        class_counts = [(c, np.sum(data[i] == c)) for c in classes_in_slice if c > 0]
        print(f'  Slice {i}: Classes {classes_in_slice}, counts: {class_counts}')

