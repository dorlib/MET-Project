#!/usr/bin/env python3
# api_gateway/preprocessing.py - Utilities for preprocessing MRI scan files

import os
import numpy as np
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)

def preprocess_nifti_to_npy(nifti_file_path, output_npy_path):
    """
    Convert a NIfTI file to NPY format with appropriate preprocessing.
    
    Args:
        nifti_file_path: Path to the input NIfTI file
        output_npy_path: Path where the NPY file should be saved
        
    Returns:
        bool: True if conversion was successful, False otherwise
    """
    try:
        # Load NIfTI file
        logger.info(f"Loading NIfTI file: {nifti_file_path}")
        img_nifti = nib.load(nifti_file_path)
        img_data = img_nifti.get_fdata()
        
        # Apply preprocessing similar to the sample code
        logger.info(f"Preprocessing NIfTI data with shape: {img_data.shape}")
        
        # Apply normalization using MinMaxScaler (0-1 range)
        scaler = MinMaxScaler()
        img_data = scaler.fit_transform(img_data.reshape(-1, 1)).reshape(img_data.shape)
        
        # Save as NPY
        logger.info(f"Saving preprocessed data to NPY: {output_npy_path}")
        np.save(output_npy_path, img_data)
        
        return True
    except Exception as e:
        logger.error(f"Error preprocessing NIfTI file: {str(e)}")
        return False

def preprocess_nifti_t1ce_for_model(nifti_file_path, output_npy_path):
    """
    Preprocess a T1ce NIfTI file to make it compatible with the UNETR model.
    
    This applies preprocessing specifically optimized for the T1ce modality:
    1. Loads the T1ce NIfTI file
    2. Applies MinMaxScaler normalization 
    3. Ensures the data has appropriate dimensions
    4. Saves as NPY format compatible with the model
    
    Args:
        nifti_file_path: Path to the T1ce NIfTI file
        output_npy_path: Path where the NPY file should be saved
        
    Returns:
        bool: True if preprocessing was successful, False otherwise
    """
    try:
        # Load NIfTI file
        logger.info(f"Loading T1ce NIfTI file: {nifti_file_path}")
        img_nifti = nib.load(nifti_file_path)
        img_data = img_nifti.get_fdata()
        
        # Get original shape for logging
        original_shape = img_data.shape
        logger.info(f"Original image shape: {original_shape}")
        
        # Apply normalization
        scaler = MinMaxScaler()
        img_data = scaler.fit_transform(img_data.reshape(-1, 1)).reshape(img_data.shape)
        
        # Crop or pad to ensure dimensions are appropriate
        # We'll center crop/pad to 128x128x128 as the UNETR model expects
        target_shape = (128, 128, 128)
        
        # Create an output array filled with zeros (background)
        processed_data = np.zeros(target_shape, dtype=np.float32)
        
        # Calculate dimensions to copy
        copy_shape = [min(img_data.shape[i], target_shape[i]) for i in range(3)]
        
        # Calculate start indices for centering
        img_start = [(img_data.shape[i] - copy_shape[i]) // 2 for i in range(3)]
        target_start = [(target_shape[i] - copy_shape[i]) // 2 for i in range(3)]
        
        # Copy the relevant portion of the data
        processed_data[
            target_start[0]:target_start[0]+copy_shape[0],
            target_start[1]:target_start[1]+copy_shape[1],
            target_start[2]:target_start[2]+copy_shape[2]
        ] = img_data[
            img_start[0]:img_start[0]+copy_shape[0],
            img_start[1]:img_start[1]+copy_shape[1],
            img_start[2]:img_start[2]+copy_shape[2]
        ]
        
        logger.info(f"Processed image shape: {processed_data.shape}")
        
        # Save as NPY
        logger.info(f"Saving preprocessed data to NPY: {output_npy_path}")
        np.save(output_npy_path, processed_data)
        
        return True
    except Exception as e:
        logger.error(f"Error preprocessing T1ce NIfTI file: {str(e)}")
        return False
