import numpy as np

def get_npy_dimensions(file_path):
    # Load the .npy file
    data = np.load(file_path)
    
    # Return the shape (dimensions) of the array
    return data.shape

# Example usage:
file_path = "C:/Users/dorli/Downloads/prediction_mask_a7778af1-4052-4065-b4ea-f34503c986bc.npy"
dimensions = get_npy_dimensions(file_path)
print("Dimensions of the .npy file:", dimensions)