#!/usr/bin/env python3
"""
MET Model Inference Script

This script demonstrates how to load the pre-trained UNETR model and use it for inference
on a .npy file containing a brain MRI scan.

Usage:
    python model_inference.py <input_npy_file> [output_npy_file]

If output_npy_file is not provided, it will be saved as <input_npy_file>_prediction.npy
"""

import argparse
import os
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt

# Try to import the correct UNETR implementation
try:
    from self_attention_cv import UNETR
    print("Using self_attention_cv UNETR implementation")
except ImportError:
    print("self_attention_cv not found, falling back to MONAI UNETR")
    from monai.networks.nets import UNETR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def get_model(checkpoint_path, device):
    """
    Initialize and load the UNETR model for MET segmentation
    
    Args:
        checkpoint_path: Path to the model weights file
        device: torch.device for model execution
        
    Returns:
        Loaded and initialized model in eval mode
    """
    logger.info(f"Loading model from {checkpoint_path} on {device}")
    
    # Check if file exists
    if not os.path.exists(checkpoint_path):
        logger.error(f"Model file not found: {checkpoint_path}")
        raise FileNotFoundError(f"Model file not found: {checkpoint_path}")
    
    # Check if file is empty
    if os.path.getsize(checkpoint_path) <= 1024:  # Likely a placeholder
        logger.warning(f"Model file may be a placeholder: {checkpoint_path}")
    
    # Use the exact same UNETR implementation and parameters from the training script
    try:
        from self_attention_cv import UNETR
        # Parameters exactly matching the training script: unetr_t1c_no_permute_tversky.py
        model = UNETR(
            img_shape=(128, 128, 128),
            input_dim=1,
            output_dim=4,  # NUM_CLASSES = 4
            embed_dim=128,
            patch_size=16,
            num_heads=8,
            ext_layers=[3, 6, 9, 12, 15, 18],
            norm='instance',
            dropout=0.2,
            base_filters=16,
            dim_linear_block=1024
        ).to(device)
        logger.info("Using self_attention_cv UNETR implementation")
    except ImportError as e:
        logger.error(f"self_attention_cv not available: {e}")
        raise ImportError("self_attention_cv library is required for this model")

    try:
        state = torch.load(checkpoint_path, map_location=device)
        
        # Check if state dict is valid
        if not isinstance(state, dict) and hasattr(state, 'state_dict'):
            # If we have a full model object instead of just the state dict
            state = state.state_dict()
        elif not isinstance(state, dict):
            # If state is not a dictionary at all, raise error
            raise ValueError("Invalid model state format")
            
        model.load_state_dict(state)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")
        
    model.eval()
    return model

def preprocess_scan(scan):
    """
    Preprocess the scan to match the expected input format of the model
    
    Args:
        scan: Numpy array containing the scan data
        
    Returns:
        Preprocessed scan as a torch tensor
    """
    logger.info(f"Original scan shape: {scan.shape}")
    
    # If scan has multiple channels (e.g., from preprocessing.py), extract T1CE channel (index 1)
    if scan.ndim == 4 and scan.shape[-1] > 1:
        logger.info(f"Multi-channel scan detected with shape {scan.shape}. Using channel 1 (T1CE).")
        scan = scan[..., 1]  # Extract T1CE channel
    
    # Normalize the scan
    scan = (scan - scan.mean()) / (scan.std() + 1e-5)
    
    # Add batch and channel dimensions
    x = torch.from_numpy(scan).unsqueeze(0).unsqueeze(0).float()
    logger.info(f"Preprocessed scan shape: {x.shape}")
    return x

def visualize_slices(original_scan, prediction, output_prefix):
    """
    Create visualization of the original scan and the prediction
    
    Args:
        original_scan: Original input scan
        prediction: Model prediction/segmentation
        output_prefix: Prefix for output image files
    """
    # Extract middle slices for visualization
    if original_scan.ndim == 4 and original_scan.shape[-1] > 1:
        original_scan = original_scan[..., 1]  # Use T1CE channel for visualization
    
    # Get middle slices
    z_mid = original_scan.shape[2] // 2
    y_mid = original_scan.shape[1] // 2
    x_mid = original_scan.shape[0] // 2
    
    # Create figure with 3 rows (axial, coronal, sagittal) and 2 columns (original, segmentation)
    fig, axes = plt.subplots(3, 2, figsize=(12, 18))
    
    # Row 1: Axial view
    axes[0, 0].imshow(original_scan[:, :, z_mid], cmap='gray')
    axes[0, 0].set_title('Original - Axial')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(original_scan[:, :, z_mid], cmap='gray')
    axes[0, 1].imshow(prediction[:, :, z_mid], cmap='jet', alpha=0.5)
    axes[0, 1].set_title('Segmentation - Axial')
    axes[0, 1].axis('off')
    
    # Row 2: Coronal view
    axes[1, 0].imshow(original_scan[:, y_mid, :], cmap='gray')
    axes[1, 0].set_title('Original - Coronal')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(original_scan[:, y_mid, :], cmap='gray')
    axes[1, 1].imshow(prediction[:, y_mid, :], cmap='jet', alpha=0.5)
    axes[1, 1].set_title('Segmentation - Coronal')
    axes[1, 1].axis('off')
    
    # Row 3: Sagittal view
    axes[2, 0].imshow(original_scan[x_mid, :, :], cmap='gray')
    axes[2, 0].set_title('Original - Sagittal')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(original_scan[x_mid, :, :], cmap='gray')
    axes[2, 1].imshow(prediction[x_mid, :, :], cmap='jet', alpha=0.5)
    axes[2, 1].set_title('Segmentation - Sagittal')
    axes[2, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_visualization.png")
    logger.info(f"Saved visualization to {output_prefix}_visualization.png")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='MET Model Inference')
    parser.add_argument('input_file', help='Path to input .npy file')
    parser.add_argument('--output_file', help='Path to output .npy file')
    parser.add_argument('--model_path', default='./Data/saved_models/brats_t1ce.pth',
                        help='Path to the model weights file')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization of the results')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set output file if not provided
    if args.output_file is None:
        base_name = os.path.splitext(args.input_file)[0]
        args.output_file = f"{base_name}_prediction.npy"
    
    try:
        # Load input scan
        logger.info(f"Loading scan from {args.input_file}")
        scan = np.load(args.input_file)
        logger.info(f"Scan loaded with shape {scan.shape}")
        
        # Load model
        model = get_model(args.model_path, device)
        
        # Preprocess scan
        input_tensor = preprocess_scan(scan)
        input_tensor = input_tensor.to(device)
        
        # Run inference
        logger.info("Running inference...")
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            logger.info(f"Prediction shape: {prediction.shape}")
            logger.info(f"Unique values in prediction: {np.unique(prediction)}")
        
        # Save prediction
        np.save(args.output_file, prediction)
        logger.info(f"Prediction saved to {args.output_file}")
        
        # Generate visualization if requested
        if args.visualize:
            output_prefix = os.path.splitext(args.output_file)[0]
            visualize_slices(scan, prediction, output_prefix)
        
        logger.info("Inference completed successfully")
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
