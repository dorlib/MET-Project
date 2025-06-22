#!/usr/bin/env python3
# mock_model_adapter.py - Lightweight mock model for development and testing

import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)

class MockModelAdapter:
    """
    A lightweight mock model adapter that simulates segmentation results
    for development and testing when the full model is too resource-intensive.
    """
    
    def __init__(self):
        logging.info("Initializing MockModelAdapter (lightweight testing model)")
        
    def load_model(self):
        """Simulates model loading - always returns True"""
        logging.info("Mock model loaded successfully")
        return True
    
    def predict(self, img_path):
        """
        Creates a simple mock segmentation based on the input volume dimensions
        without requiring the full model and its memory footprint.
        
        Args:
            img_path: Path to the input image
            
        Returns:
            Dict with prediction and original image
        """
        try:
            # Load the input volume
            logging.info(f"Loading input volume from {img_path}")
            orig_vol = np.load(img_path)
            
            # Get the volume shape
            volume_shape = orig_vol.shape
            logging.info(f"Input volume shape: {volume_shape}")
            
            # Create a mock segmentation mask
            logging.info("Creating mock segmentation")
            pred_mask = np.zeros_like(orig_vol, dtype=np.int64)
            
            # Add a few "metastases" regions
            depth, height, width = volume_shape
            center_d, center_h, center_w = depth // 2, height // 2, width // 2
            
            # Add a central region as the main "metastasis"
            d_radius, h_radius, w_radius = depth // 10, height // 10, width // 10
            
            # Generate a few random locations for "metastases"
            np.random.seed(42)  # Use a fixed seed for reproducibility
            num_metastases = 5
            
            for i in range(num_metastases):
                # Random location for this metastasis
                d_offset = np.random.randint(-depth // 4, depth // 4)
                h_offset = np.random.randint(-height // 4, height // 4)
                w_offset = np.random.randint(-width // 4, width // 4)
                
                # Random size for this metastasis
                size_factor = np.random.uniform(0.5, 1.5)
                d_size = int(d_radius * size_factor)
                h_size = int(h_radius * size_factor)
                w_size = int(w_radius * size_factor)
                
                # Center of this metastasis
                d_center = center_d + d_offset
                h_center = center_h + h_offset
                w_center = center_w + w_offset
                
                # Create a spherical metastasis
                for d in range(max(0, d_center - d_size), min(depth, d_center + d_size)):
                    for h in range(max(0, h_center - h_size), min(height, h_center + h_size)):
                        for w in range(max(0, w_center - w_size), min(width, w_center + w_size)):
                            # Check if this voxel is within the sphere
                            if ((d - d_center) / d_size) ** 2 + \
                               ((h - h_center) / h_size) ** 2 + \
                               ((w - w_center) / w_size) ** 2 <= 1.0:
                                pred_mask[d, h, w] = 1
            
            # Add some "edema" around the metastases
            edema_mask = np.zeros_like(pred_mask)
            for i in range(1, 4):  # Create a few layers of "edema" around metastases
                # Dilate the metastases
                dilated = np.zeros_like(pred_mask)
                for d in range(1, depth - 1):
                    for h in range(1, height - 1):
                        for w in range(1, width - 1):
                            if pred_mask[d, h, w] == 1 or \
                               pred_mask[max(0, d-1):d+2, max(0, h-1):h+2, max(0, w-1):w+2].sum() > 0:
                                dilated[d, h, w] = 1
                
                # Add new dilation to edema mask
                edema_mask = np.logical_or(edema_mask, np.logical_and(dilated == 1, pred_mask == 0))
            
            # Set edema voxels to class 2
            pred_mask[edema_mask] = 2
            
            logging.info("Mock segmentation created successfully")
            
            return {
                'prediction': pred_mask,
                'original_image': orig_vol
            }
            
        except Exception as e:
            logging.error(f"Error in mock prediction: {str(e)}")
            # Return a simple volume on error
            return {
                'prediction': np.zeros((128, 128, 128), dtype=np.int64),
                'original_image': np.zeros((128, 128, 128), dtype=np.float32)
            }

# Create a global instance
mock_model = MockModelAdapter()
