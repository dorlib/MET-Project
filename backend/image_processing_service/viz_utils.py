#!/usr/bin/env python3
# Helper utilities for enhancing visualization quality

import numpy as np
import cv2
from skimage import exposure

def enhance_segmentation_mask(segmentation, 
                             enhance_boundaries=True, 
                             smooth_edges=True):
    """
    Enhance a segmentation mask by sharpening boundaries and smoothing edges
    
    Args:
        segmentation: 3D or 2D segmentation mask
        enhance_boundaries: Whether to enhance boundaries between regions
        smooth_edges: Whether to smooth jagged edges
        
    Returns:
        Enhanced segmentation mask (same shape as input)
    """
    # Handle 3D case by processing each slice
    if len(segmentation.shape) == 3:
        enhanced = np.zeros_like(segmentation)
        for i in range(segmentation.shape[0]):
            enhanced[i] = enhance_segmentation_mask(
                segmentation[i], 
                enhance_boundaries, 
                smooth_edges
            )
        return enhanced
    
    # Process 2D slice
    mask = segmentation.copy()
    
    # 1. Convert to unsigned integer type if it's not already
    if mask.dtype != np.uint8:
        # Make sure we don't exceed class values
        max_val = np.max(mask)
        if max_val > 0:
            mask = (mask * (255 // max_val)).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)
    
    if enhance_boundaries:
        # Create a separate mask for each class
        unique_classes = np.unique(mask)
        for cls in unique_classes:
            if cls == 0:  # Skip background
                continue
                
            # Create binary mask for this class
            binary = (mask == cls).astype(np.uint8) * 255
            
            # Enhance boundaries with morphological operations
            kernel = np.ones((3, 3), np.uint8)
            eroded = cv2.erode(binary, kernel, iterations=1)
            boundary = binary - eroded
            
            # Set the boundary pixels in the original mask
            mask[boundary > 0] = cls
    
    if smooth_edges and mask.shape[0] > 50 and mask.shape[1] > 50:
        # For each class, apply a small blur then threshold back to the original class
        unique_classes = np.unique(mask)
        result = np.zeros_like(mask)
        
        for cls in unique_classes:
            if cls == 0:  # Skip background
                continue
                
            # Create binary mask for this class
            binary = (mask == cls).astype(np.uint8)
            
            # Apply small blur to smooth jagged edges
            smoothed = cv2.GaussianBlur(binary.astype(float), (3, 3), 0.8)
            
            # Convert back to binary with threshold
            smoothed = (smoothed > 0.5).astype(np.uint8)
            
            # Add back to result
            result[smoothed > 0] = cls
            
        # Make sure we preserve the original mask where the smoothed version made changes
        result[mask > 0] = mask[mask > 0]
        mask = result
    
    # Convert back to original mask format
    if segmentation.dtype != np.uint8:
        # If we normalized to 0-255 range, convert back to original range
        if np.max(mask) > np.max(segmentation) and np.max(segmentation) > 0:
            mask = mask.astype(segmentation.dtype) * (np.max(segmentation) / 255)
    
    return mask

def enhance_original_image(image, 
                          contrast_enhancement=True, 
                          denoise=True,
                          detail_enhancement=True,
                          histogram_equalization='adaptive'):
    """
    Enhance the original MRI image to improve visibility and details
    
    Args:
        image: 2D image slice
        contrast_enhancement: Whether to enhance contrast
        denoise: Whether to apply denoising
        detail_enhancement: Whether to enhance fine details
        histogram_equalization: Type of histogram equalization ('none', 'global', 'adaptive')
        
    Returns:
        Enhanced image (same shape as input)
    """
    if image.dtype != np.uint8:
        # Normalize to 0-255 range
        img_min = np.min(image)
        img_max = np.max(image)
        if img_min != img_max:
            img = ((image - img_min) * (255.0 / (img_max - img_min))).astype(np.uint8)
        else:
            img = np.zeros_like(image, dtype=np.uint8)
    else:
        img = image.copy()
    
    # Apply denoising if requested
    if denoise:
        img = cv2.fastNlMeansDenoising(img, None, h=7, templateWindowSize=7, searchWindowSize=21)
    
    # Apply histogram equalization
    if histogram_equalization == 'global':
        img = cv2.equalizeHist(img)
    elif histogram_equalization == 'adaptive':
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
    
    # Apply additional contrast enhancement
    if contrast_enhancement:
        # Enhance contrast using CLAHE again with different parameters
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        img = clahe.apply(img)
        
        # Apply gamma correction for better contrast
        img_float = img.astype(float) / 255.0
        img_float = np.power(img_float, 0.9)  # Gamma < 1 enhances shadows
        img = (img_float * 255).astype(np.uint8)
    
    # Enhance fine details
    if detail_enhancement:
        # Apply unsharp mask for detail enhancement
        blurred = cv2.GaussianBlur(img, (0, 0), 3)
        img = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
        
        # Apply CLAHE one more time to enhance details
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        img = clahe.apply(img)
    
    return img

def create_colorized_overlay(segmentation, original_image, alpha=0.7):
    """
    Create a nicely colorized overlay for visualization
    
    Args:
        segmentation: Segmentation mask (2D)
        original_image: Original image (2D)
        alpha: Transparency level for the overlay
        
    Returns:
        RGB image with colorized overlay
    """
    # Ensure original image is in range 0-255
    if original_image.dtype != np.uint8:
        img_min = np.min(original_image)
        img_max = np.max(original_image)
        if img_min != img_max:
            img_normalized = ((original_image - img_min) * (255.0 / (img_max - img_min))).astype(np.uint8)
        else:
            img_normalized = np.zeros_like(original_image, dtype=np.uint8)
    else:
        img_normalized = original_image.copy()
    
    # Create RGB version of original image
    img_rgb = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2RGB)
    
    # Create colored overlay
    overlay = np.zeros_like(img_rgb, dtype=np.uint8)
    
    # Define colors for each class (BGR format for OpenCV)
    colors = {
        1: (255, 0, 0),    # Blue for tumor core
        2: (0, 255, 0),    # Green for edema
        3: (0, 0, 255)     # Red for metastasis
    }
    
    # Apply colors to each class
    for class_id, color in colors.items():
        mask = (segmentation == class_id)
        if np.any(mask):
            overlay[mask] = color
    
    # Blend original image with overlay
    result = cv2.addWeighted(img_rgb, 1.0, overlay, alpha, 0)
    
    return result
