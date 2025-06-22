#!/usr/bin/env python3
# Test script for NIfTI to NPY conversion

import os
import sys
import logging
import numpy as np
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the preprocessing module
from preprocessing import preprocess_nifti_t1ce_for_model, preprocess_nifti_to_npy

def test_nifti_conversion(nifti_path, output_dir):
    """
    Test the NIfTI to NPY conversion functionality
    
    Args:
        nifti_path: Path to a NIfTI file for testing
        output_dir: Directory to save the output NPY file
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output paths
        output_basic = os.path.join(output_dir, "basic_conversion.npy")
        output_t1ce = os.path.join(output_dir, "t1ce_preprocessed.npy")
        
        # Test basic conversion
        logger.info(f"Testing basic NIfTI to NPY conversion with file: {nifti_path}")
        if preprocess_nifti_to_npy(nifti_path, output_basic):
            # Load and verify the output
            basic_data = np.load(output_basic)
            logger.info(f"Basic conversion successful. Output shape: {basic_data.shape}, dtype: {basic_data.dtype}")
        else:
            logger.error("Basic conversion failed")
        
        # Test T1ce-specific preprocessing
        logger.info(f"Testing T1ce preprocessing with file: {nifti_path}")
        if preprocess_nifti_t1ce_for_model(nifti_path, output_t1ce):
            # Load and verify the output
            t1ce_data = np.load(output_t1ce)
            logger.info(f"T1ce preprocessing successful. Output shape: {t1ce_data.shape}, dtype: {t1ce_data.dtype}")
            logger.info(f"Value range: min={t1ce_data.min()}, max={t1ce_data.max()}")
            
            # Check if the output dimensions are correct (expecting 128x128x128)
            if t1ce_data.shape != (128, 128, 128):
                logger.warning(f"T1ce output dimensions {t1ce_data.shape} do not match expected (128, 128, 128)")
        else:
            logger.error("T1ce preprocessing failed")
            
        logger.info("Conversion tests completed. Check the output files for further verification.")
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test NIfTI to NPY conversion")
    parser.add_argument("nifti_path", help="Path to a NIfTI file for testing (.nii or .nii.gz)")
    parser.add_argument("--output-dir", default="./test_output", help="Directory to save the output NPY files")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.nifti_path):
        logger.error(f"NIfTI file not found: {args.nifti_path}")
        sys.exit(1)
        
    if not (args.nifti_path.endswith('.nii') or args.nifti_path.endswith('.nii.gz')):
        logger.error(f"Invalid file format. Expected .nii or .nii.gz, got: {args.nifti_path}")
        sys.exit(1)
    
    success = test_nifti_conversion(args.nifti_path, args.output_dir)
    sys.exit(0 if success else 1)
