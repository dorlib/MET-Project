#!/usr/bin/env python3
# High-resolution visualization functions for MET project

import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image, ImageEnhance
import cv2
import logging
import os
from scipy.ndimage import zoom
from viz_utils import enhance_segmentation_mask, enhance_original_image, create_colorized_overlay

def create_high_res_visualization(segmentation, original_image=None, slice_idx=None, tissue_colors=None, tissue_names=None, 
                                 upscale_factor=1.0, contrast_enhancement=True, edge_enhancement=True):
    """
    Create a high-resolution colormap visualization of a segmentation mask
    
    Args:
        segmentation: 3D segmentation mask
        original_image: Optional original 3D image
        slice_idx: Optional slice index, if None middle slice will be used
        tissue_colors: Dictionary mapping class IDs to RGB tuples
        tissue_names: Dictionary mapping class IDs to display names
        upscale_factor: Factor to upscale the image (1.0 = original size)
        contrast_enhancement: Whether to apply contrast enhancement
        edge_enhancement: Whether to apply edge enhancement
        
    Returns:
        PIL Image object containing the visualization
    """
    logging.info(f"Creating high-res visualization with slice_idx={slice_idx}, upscale_factor={upscale_factor}")
    logging.info(f"Segmentation shape: {segmentation.shape}, unique values: {np.unique(segmentation)}")
    if original_image is not None:
        logging.info(f"Original image shape: {original_image.shape}")
        
    if tissue_colors is None:
        tissue_colors = {
            1: (1.0, 0.0, 0.0),    # Red for metastasis (class 1 in simplified model)
            2: (0.0, 1.0, 0.0),    # Green for edema (class 2 in simplified model)
            3: (0.0, 0.0, 1.0)     # Blue for tumor core (not used in simplified model)
        }
        
    if tissue_names is None:
        tissue_names = {
            1: "Metastasis",
            2: "Edema",
            3: "Tumor Core"
        }
    
    # Determine which slice to use and ensure it's valid
    if slice_idx is None:
        slice_idx = segmentation.shape[0] // 2  # Middle slice
        logging.info(f"No slice index provided, using middle slice: {slice_idx}")
    else:
        # Ensure slice_idx is an integer
        try:
            slice_idx = int(slice_idx)
            logging.info(f"Using provided slice index: {slice_idx}")
        except (ValueError, TypeError):
            logging.error(f"Invalid slice_idx: {slice_idx}, must be an integer")
            slice_idx = segmentation.shape[0] // 2
            logging.info(f"Using middle slice instead: {slice_idx}")
        
        # Make sure the slice index is within bounds
        if slice_idx < 0:
            logging.warning(f"Slice index {slice_idx} was negative, using slice 0 instead")
            slice_idx = 0
        elif slice_idx >= segmentation.shape[0]:
            max_idx = segmentation.shape[0] - 1
            logging.warning(f"Slice index {slice_idx} was too large (max={max_idx}), using slice {max_idx} instead")
            slice_idx = max_idx
    
    logging.info(f"Final slice index: {slice_idx} (valid range: 0-{segmentation.shape[0]-1})")
    
    # Extract the requested slice with error handling
    try:
        if len(segmentation.shape) == 3:
            mask_slice = segmentation[slice_idx, :, :]
        else:
            # If the mask has an additional dimension (e.g., one-hot encoding)
            mask_slice = np.argmax(segmentation[slice_idx, :, :], axis=-1) if segmentation.shape[-1] > 1 else segmentation[slice_idx, :, :]
        
        logging.info(f"Extracted mask slice with shape: {mask_slice.shape}, unique values: {np.unique(mask_slice)}")
    
    except Exception as e:
        logging.error(f"Error extracting mask slice: {str(e)}, details: {type(e).__name__}")
        # Fallback to middle slice if there was an error
        middle_slice = segmentation.shape[0] // 2
        mask_slice = segmentation[middle_slice, :, :]
        logging.info(f"Using fallback middle slice {middle_slice} instead")
    
    # Create a colored visualization
    colors = np.zeros((*mask_slice.shape, 4))
    
    # Apply colors for each tissue type with increased vividness for better visibility
    for class_id, color in tissue_colors.items():
        mask = mask_slice == class_id
        if np.any(mask):
            # Use full opacity for a stronger color effect
            colors[mask] = (*color, 1.0)  # RGB + alpha
    
    # If no segmentation found, make entire image transparent
    if not np.any(colors):
        colors[:] = (0, 0, 0, 0)
    
    # If original image provided, use it as background with enhanced contrast
    if original_image is not None:
        if slice_idx < original_image.shape[0]:
            orig_slice = original_image[slice_idx]
            
            # Apply advanced image enhancement with our utility function
            if contrast_enhancement:
                try:
                    # Normalize to 0-1 range for consistency
                    if orig_slice.min() != orig_slice.max():
                        norm_slice = (orig_slice - orig_slice.min()) / (orig_slice.max() - orig_slice.min())
                        img_8bit = (norm_slice * 255).astype(np.uint8)
                    else:
                        img_8bit = np.zeros_like(orig_slice, dtype=np.uint8)
                    
                    # Apply enhanced image processing
                    enhanced_img = enhance_original_image(
                        img_8bit, 
                        contrast_enhancement=True,
                        denoise=True,
                        detail_enhancement=edge_enhancement,
                        histogram_equalization='adaptive'
                    )
                    
                    # Convert back to float in range 0-1
                    orig_slice = enhanced_img / 255.0
                except Exception as e:
                    logging.warning(f"Advanced image enhancement failed: {str(e)}, falling back to basic normalization")
                    # Fall back to basic normalization
                    if orig_slice.min() != orig_slice.max():
                        orig_slice = (orig_slice - orig_slice.min()) / (orig_slice.max() - orig_slice.min())
            else:
                # Basic normalization if no enhancement requested
                if orig_slice.min() != orig_slice.max():
                    orig_slice = (orig_slice - orig_slice.min()) / (orig_slice.max() - orig_slice.min())
        else:
            # If slice index out of range, create blank background
            orig_slice = np.zeros_like(mask_slice, dtype=float)
    else:
        # Create a grayscale background if no original image
        orig_slice = np.zeros_like(mask_slice, dtype=float)
    
    # Enhance segmentation mask for better visualization if quality is high
    try:
        enhanced_mask = enhance_segmentation_mask(
            mask_slice,
            enhance_boundaries=edge_enhancement,
            smooth_edges=True
        )
        mask_slice = enhanced_mask
    except Exception as e:
        logging.warning(f"Segmentation enhancement failed: {str(e)}, using original mask")
    
    # Apply upsampling for smoother, higher resolution output
    if upscale_factor > 1.0:
        try:
            # Upsample the segmentation mask using nearest neighbor interpolation to preserve class labels
            mask_slice_upsampled = zoom(mask_slice, upscale_factor, order=0)
            
            # Upsample the original image using higher order interpolation for smoother results
            orig_slice_upsampled = zoom(orig_slice, upscale_factor, order=3)  # Cubic interpolation
            
            # Replace with upsampled versions
            mask_slice = mask_slice_upsampled
            orig_slice = orig_slice_upsampled
            
            # Recreate colors array at new dimensions
            colors = np.zeros((*mask_slice.shape, 4))
            
            # Reapply colors to the upsampled mask
            for class_id, color in tissue_colors.items():
                mask = mask_slice == class_id
                if np.any(mask):
                    colors[mask] = (*color, 1.0)  # RGB + alpha
            
            if not np.any(colors):
                colors[:] = (0, 0, 0, 0)
        except Exception as e:
            logging.warning(f"Upsampling failed: {str(e)}, using original resolution")
    
    # Create a figure with the visualization at higher resolution
    dpi = 300  # Higher DPI for better resolution (increased from 200)
    
    # Calculate figure size based on image dimensions to maintain aspect ratio
    height, width = mask_slice.shape
    aspect_ratio = width / height
    
    # Set a base size with higher values for better resolution
    fig_height = max(14, height / 60)  # Larger size in inches (increased from 12, height/75)
    fig_width = fig_height * aspect_ratio
    
    plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    
    # Plot the original image as grayscale with improved rendering
    plt.imshow(orig_slice, cmap='gray', interpolation='lanczos')
    
    # Overlay the colormap visualization with increased alpha for better visibility
    plt.imshow(colors, alpha=0.8, interpolation='lanczos')
    
    # Add a legend for tissue types
    legend_elements = []
    for class_id, color in tissue_colors.items():
        if np.any(mask_slice == class_id):  # Only show tissues present in this slice
            class_name = tissue_names.get(class_id, f"Class {class_id}")
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=color, markersize=10, label=class_name))
    
    if legend_elements:
        plt.legend(handles=legend_elements, loc='upper right', framealpha=0.7)
    
    # Improved title with slice information
    plt.title(f"Segmentation Visualization (Slice {slice_idx}/{segmentation.shape[0]-1})", 
              fontsize=14, fontweight='bold')
    plt.axis('off')  # Hide axes
    
    # Reduce padding to maximize image size
    plt.tight_layout(pad=0.5)
    
    # Convert the figure to a high-quality PNG image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, dpi=dpi, 
                metadata={'Software': 'MET-project'}, pil_kwargs={'quality': 95})
    plt.close()
    buf.seek(0)
    
    # Load the image and apply final enhancements
    img = Image.open(buf)
    
    try:
        # Apply PIL-based enhancements for final touches
        if contrast_enhancement:
            # Enhance sharpness slightly
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.2)  # 1.0 is original, >1 is sharper
            
            # Enhance contrast slightly
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.1)  # 1.0 is original, >1 is higher contrast
            
            # Add a tiny bit of brightness
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.05)  # 1.0 is original
        
        # Use high-quality resampling if we need to resize
        if upscale_factor > 1.0 and upscale_factor <= 1.5:
            # For small upscales, we can further enhance with PIL
            current_width, current_height = img.size
            new_width = int(current_width * 1.25)
            new_height = int(current_height * 1.25)
            img = img.resize((new_width, new_height), Image.LANCZOS)
    except Exception as e:
        logging.warning(f"PIL enhancement failed: {str(e)}")
    
    return img

def generate_high_res_multi_slice_view(segmentation, original_image=None, num_slices=5, 
                                      contrast_enhancement=True, edge_enhancement=True,
                                      tissue_colors=None, tissue_names=None):
    """
    Generate multiple high-resolution slice views of the segmentation
    
    Args:
        segmentation: 3D segmentation mask
        original_image: Optional original 3D image
        num_slices: Number of slices to show
        contrast_enhancement: Whether to apply contrast enhancement
        edge_enhancement: Whether to apply edge enhancement
        tissue_colors: Dictionary mapping class IDs to RGB tuples
        tissue_names: Dictionary mapping class IDs to display names
        
    Returns:
        PIL Image object containing the multi-slice visualization
    """
    if tissue_colors is None:
        tissue_colors = {
            1: (1.0, 0.0, 0.0),    # Red for metastasis (class 1 in simplified model)
            2: (0.0, 1.0, 0.0),    # Green for edema (class 2 in simplified model)
            3: (0.0, 0.0, 1.0)     # Blue for tumor core (not used in simplified model)
        }
        
    if tissue_names is None:
        tissue_names = {
            1: "Metastasis",
            2: "Edema",
            3: "Tumor Core"
        }
        
    depth = segmentation.shape[0]
    
    # Calculate slice indices evenly distributed through volume
    indices = np.linspace(0, depth - 1, num_slices).astype(int)
    
    # Create a figure with multiple slices
    fig, axes = plt.subplots(1, num_slices, figsize=(5 * num_slices, 5), dpi=200)
    
    for i, slice_idx in enumerate(indices):
        # Extract the slice
        mask_slice = segmentation[slice_idx]
        
        # Create colored mask with enhanced visibility
        colored_mask = np.zeros((*mask_slice.shape, 4))
        
        # Apply colors for each tissue type
        tissue_present = []
        for class_id, color in tissue_colors.items():
            mask = mask_slice == class_id
            if np.any(mask):
                colored_mask[mask] = (*color, 0.7)  # RGB + alpha
                tissue_present.append(tissue_names.get(class_id, f"Class {class_id}"))
        
        # Get original image slice if available
        if original_image is not None and slice_idx < original_image.shape[0]:
            orig_slice = original_image[slice_idx]
            
            # Apply advanced image enhancement with our utility function
            if contrast_enhancement:
                try:
                    # Normalize to 0-1 range for consistency
                    if orig_slice.min() != orig_slice.max():
                        norm_slice = (orig_slice - orig_slice.min()) / (orig_slice.max() - orig_slice.min())
                        img_8bit = (norm_slice * 255).astype(np.uint8)
                    else:
                        img_8bit = np.zeros_like(orig_slice, dtype=np.uint8)
                    
                    # Apply enhanced image processing
                    enhanced_img = enhance_original_image(
                        img_8bit, 
                        contrast_enhancement=True,
                        denoise=True,
                        detail_enhancement=edge_enhancement,
                        histogram_equalization='adaptive'
                    )
                    
                    # Convert back to float in range 0-1
                    orig_slice = enhanced_img / 255.0
                except Exception as e:
                    logging.warning(f"Advanced image enhancement failed: {str(e)}, falling back to basic normalization")
                    # Fall back to basic normalization
                    if orig_slice.min() != orig_slice.max():
                        orig_slice = (orig_slice - orig_slice.min()) / (orig_slice.max() - orig_slice.min())
            else:
                # Basic normalization if no enhancement requested
                if orig_slice.min() != orig_slice.max():
                    orig_slice = (orig_slice - orig_slice.min()) / (orig_slice.max() - orig_slice.min())
                    
            # Enhance segmentation mask
            try:
                mask_slice = enhance_segmentation_mask(
                    mask_slice,
                    enhance_boundaries=edge_enhancement,
                    smooth_edges=True
                )
            except Exception as e:
                logging.warning(f"Segmentation enhancement failed: {str(e)}")
        else:
            # Create a blank background
            orig_slice = np.zeros_like(mask_slice, dtype=float)
        
        # Plot the original slice with interpolation for smoother appearance
        axes[i].imshow(orig_slice, cmap='gray', interpolation='lanczos')
        
        # Overlay the colored mask with improved interpolation
        axes[i].imshow(colored_mask, interpolation='lanczos')
        
        # Add more detailed slice information
        slice_info = f"Slice {slice_idx}/{depth-1}"
        if tissue_present:
            slice_info += f"\n{', '.join(tissue_present)}"
            
        axes[i].set_title(slice_info, fontsize=10)
        axes[i].axis('off')
    
    # Add a main title with information about the visualization
    plt.suptitle("Multi-Slice Segmentation Visualization", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Convert the figure to a high-quality PNG image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=200, 
                pil_kwargs={'quality': 95})
    plt.close()
    buf.seek(0)
    
    # Load the image and apply final enhancements
    img = Image.open(buf)
    
    try:
        # Apply PIL-based enhancements for final touches
        if contrast_enhancement:
            # Enhance sharpness slightly
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.2)
    except Exception as e:
        logging.warning(f"PIL enhancement failed: {str(e)}")
    
    return img
