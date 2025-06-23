#!/usr/bin/env python3
# High-resolution visualization functions for MET project

import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import cv2
import logging
import os
from scipy.ndimage import zoom
from viz_utils import enhance_segmentation_mask, enhance_original_image, create_colorized_overlay

def create_high_res_visualization(segmentation, original_image=None, slice_idx=None, tissue_colors=None, tissue_names=None, 
                                 upscale_factor=1.0, contrast_enhancement=True, edge_enhancement=True, view_type='axial'):
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
        view_type: Type of view - 'axial' (default), 'coronal', or 'sagittal'
        
    Returns:
        PIL Image object containing the visualization
    """
    logging.info(f"Creating high-res visualization with slice_idx={slice_idx}, upscale_factor={upscale_factor}, view_type={view_type}")
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
    
    # Determine which dimension to slice based on view_type
    slice_dim = 0  # Default to axial view (first dimension)
    slice_description = "Axial"
    
    if view_type == 'axial':
        # Traditional axial view - slice along first dimension
        slice_dim = 0
        slice_description = "Axial"
    elif view_type == 'coronal':
        # Coronal view - slice along second dimension
        slice_dim = 1
        slice_description = "Coronal"
    elif view_type == 'sagittal':
        # Sagittal view - slice along third dimension
        slice_dim = 2
        slice_description = "Sagittal"
    else:
        # Default to axial if unknown view type
        logging.warning(f"Unknown view_type: {view_type}, using axial view")
        slice_dim = 0
        slice_description = "Axial"
    
    # Choose appropriate dimension size based on view_type
    if slice_dim == 0:
        max_slice_idx = segmentation.shape[0] - 1
    elif slice_dim == 1:
        max_slice_idx = segmentation.shape[1] - 1
    else:  # slice_dim == 2
        max_slice_idx = segmentation.shape[2] - 1
    
    # Determine which slice to use and ensure it's valid
    if slice_idx is None:
        if slice_dim == 0:
            slice_idx = segmentation.shape[0] // 2
        elif slice_dim == 1:
            slice_idx = segmentation.shape[1] // 2
        else:  # slice_dim == 2
            slice_idx = segmentation.shape[2] // 2
        logging.info(f"No slice index provided, using middle {slice_description} slice: {slice_idx}")
    else:
        # Ensure slice_idx is an integer
        try:
            slice_idx = int(slice_idx)
            logging.info(f"Using provided slice index: {slice_idx} for {slice_description} view")
        except (ValueError, TypeError):
            logging.error(f"Invalid slice_idx: {slice_idx}, must be an integer")
            if slice_dim == 0:
                slice_idx = segmentation.shape[0] // 2
            elif slice_dim == 1:
                slice_idx = segmentation.shape[1] // 2
            else:  # slice_dim == 2
                slice_idx = segmentation.shape[2] // 2
            logging.info(f"Using middle slice instead: {slice_idx} for {slice_description} view")
        
        # Make sure the slice index is within bounds
        if slice_idx < 0:
            logging.warning(f"Slice index {slice_idx} was negative, using slice 0 instead")
            slice_idx = 0
        elif slice_idx > max_slice_idx:
            logging.warning(f"Slice index {slice_idx} was too large (max={max_slice_idx}), using slice {max_slice_idx} instead")
            slice_idx = max_slice_idx
    
    logging.info(f"Final {slice_description} slice index: {slice_idx} (valid range: 0-{max_slice_idx})")
    
    # Extract the requested slice with error handling
    try:
        if slice_dim == 0:  # Axial
            mask_slice = segmentation[slice_idx, :, :]
        elif slice_dim == 1:  # Coronal
            mask_slice = segmentation[:, slice_idx, :]
        else:  # Sagittal (slice_dim == 2)
            mask_slice = segmentation[:, :, slice_idx]
        
        if len(segmentation.shape) > 3:  # Handle one-hot encoding if present
            mask_slice = np.argmax(mask_slice, axis=-1) if mask_slice.shape[-1] > 1 else mask_slice
            
        logging.info(f"Extracted {slice_description} mask slice with shape: {mask_slice.shape}, unique values: {np.unique(mask_slice)}")
    
    except Exception as e:
        logging.error(f"Error extracting {slice_description} mask slice: {str(e)}, details: {type(e).__name__}")
        # Fallback to middle axial slice if there was an error
        middle_slice = segmentation.shape[0] // 2
        mask_slice = segmentation[middle_slice, :, :]
        logging.info(f"Using fallback middle axial slice {middle_slice} instead")
    
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
        try:
            if slice_dim == 0 and slice_idx < original_image.shape[0]:  # Axial
                orig_slice = original_image[slice_idx, :, :]
            elif slice_dim == 1 and slice_idx < original_image.shape[1]:  # Coronal
                orig_slice = original_image[:, slice_idx, :]
            elif slice_dim == 2 and slice_idx < original_image.shape[2]:  # Sagittal
                orig_slice = original_image[:, :, slice_idx]
            else:
                orig_slice = None
                logging.warning(f"Slice index {slice_idx} out of bounds for {slice_description} view in original image")
        except Exception as e:
            logging.error(f"Error extracting original image slice: {str(e)}")
            orig_slice = None
            
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
    plt.title(f"{slice_description} Segmentation Visualization (Slice {slice_idx}/{max_slice_idx})", 
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

def create_side_by_side_visualization(segmentation, original_image=None, slice_idx=None, tissue_colors=None, tissue_names=None,
                               upscale_factor=1.0, contrast_enhancement=True, edge_enhancement=True, view_type='axial'):
    """
    Create a side-by-side visualization with original image and segmentation mask separately
    for 2D slice view (not overlaid)
    
    Args:
        segmentation: 3D segmentation mask
        original_image: Optional original 3D image
        slice_idx: Optional slice index, if None middle slice will be used
        tissue_colors: Dictionary mapping class IDs to RGB tuples
        tissue_names: Dictionary mapping class IDs to display names
        upscale_factor: Factor to upscale the image (1.0 = original size)
        contrast_enhancement: Whether to apply contrast enhancement
        edge_enhancement: Whether to apply edge enhancement
        view_type: Type of view - 'axial' (default), 'coronal', or 'sagittal'
        
    Returns:
        PIL Image object containing the visualization with original image and segmentation side-by-side
    """
    logging.info(f"Creating side-by-side visualization, view_type={view_type}, slice_idx={slice_idx}")
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
    
    # Determine which dimension to slice based on view_type
    if view_type == 'axial':
        # Traditional axial view - slice along first dimension
        slice_dim = 0
        slice_description = "Axial"
    elif view_type == 'coronal':
        # Coronal view - slice along second dimension
        slice_dim = 1
        slice_description = "Coronal"
    elif view_type == 'sagittal':
        # Sagittal view - slice along third dimension
        slice_dim = 2
        slice_description = "Sagittal"
    else:
        # Default to axial if unknown view type
        logging.warning(f"Unknown view_type: {view_type}, using axial view")
        slice_dim = 0
        slice_description = "Axial"
    
    # Extract the requested slice with error handling
    try:
        if slice_dim == 0:  # Axial
            mask_slice = segmentation[slice_idx, :, :]
            if original_image is not None:
                orig_slice = original_image[slice_idx, :, :] if slice_idx < original_image.shape[0] else None
        elif slice_dim == 1:  # Coronal
            mask_slice = segmentation[:, slice_idx, :]
            if original_image is not None:
                orig_slice = original_image[:, slice_idx, :] if slice_idx < original_image.shape[1] else None
        else:  # Sagittal
            mask_slice = segmentation[:, :, slice_idx]
            if original_image is not None:
                orig_slice = original_image[:, :, slice_idx] if slice_idx < original_image.shape[2] else None
        
        logging.info(f"Extracted {slice_description} mask slice with shape: {mask_slice.shape}, unique values: {np.unique(mask_slice)}")
    
    except Exception as e:
        logging.error(f"Error extracting {slice_description} mask slice: {str(e)}, details: {type(e).__name__}")
        # Fallback to middle axial slice if there was an error
        mask_slice = segmentation[segmentation.shape[0] // 2, :, :]
        orig_slice = original_image[segmentation.shape[0] // 2, :, :] if original_image is not None else None
        logging.info(f"Using fallback middle axial slice instead")
    
    # Create a colored visualization for the segmentation mask
    colors = np.zeros((*mask_slice.shape, 4))
    
    # Track metastasis count in this slice
    metastasis_count = 0
    
    # Apply colors for each tissue type with increased vividness for better visibility
    for class_id, color in tissue_colors.items():
        mask = mask_slice == class_id
        if np.any(mask):
            # Use full opacity for a stronger color effect
            colors[mask] = (*color, 1.0)  # RGB + alpha
            
            # Count metastasis areas (assumed to be class_id 1)
            if class_id == 1:
                # Use connected components analysis to count distinct regions
                labeled_mask, num_features = cv2.connectedComponents(mask.astype(np.uint8))
                metastasis_count = num_features - 1  # Subtract 1 for background
    
    # If no segmentation found, make entire image transparent
    if not np.any(colors):
        colors[:] = (0, 0, 0, 0)
    
    # Process the original image if available
    if original_image is not None and orig_slice is not None:
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
        # Create a grayscale background if no original image
        orig_slice = np.zeros_like(mask_slice, dtype=float)
    
    # Enhance segmentation mask for better visualization
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
    
    # Create a figure with the side-by-side visualization at higher resolution
    dpi = 300  # Higher DPI for better resolution
    
    # Calculate figure size based on image dimensions to maintain aspect ratio
    height, width = mask_slice.shape
    aspect_ratio = width / height
    
    # Set a base size with higher values for better resolution
    # For side-by-side, we need a wider figure
    fig_height = max(7, height / 80)  # Adjust for side-by-side layout
    fig_width = fig_height * aspect_ratio * 2.2  # Double width plus a bit for spacing
    
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=dpi)
    
    # Plot the original image as grayscale with improved rendering on the left
    axes[0].imshow(orig_slice, cmap='gray', interpolation='lanczos')
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis('off')  # Hide axes
    
    # Plot the segmentation mask only (not overlaid) on the right
    axes[1].imshow(colors, interpolation='lanczos')
    
    # Add a title with metastasis count
    axes[1].set_title(f"Segmentation Mask (Metastases: {metastasis_count})", fontsize=12)
    axes[1].axis('off')  # Hide axes
    
    # Add a legend for tissue types
    legend_elements = []
    for class_id, color in tissue_colors.items():
        if np.any(mask_slice == class_id):  # Only show tissues present in this slice
            class_name = tissue_names.get(class_id, f"Class {class_id}")
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=color, markersize=10, label=class_name))
    
    if legend_elements:
        axes[1].legend(handles=legend_elements, loc='upper right', framealpha=0.7, fontsize=10)
    
    # Improved title with slice information
    plt.suptitle(f"{slice_description} View (Slice {slice_idx})", 
              fontsize=14, fontweight='bold')
    
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
    except Exception as e:
        logging.warning(f"PIL enhancement failed: {str(e)}")
    
    return img

def create_side_by_side_three_plane_visualization(segmentation, original_image=None, axial_slice_idx=None, 
                                                coronal_slice_idx=None, sagittal_slice_idx=None,
                                                tissue_colors=None, tissue_names=None,
                                                contrast_enhancement=True, edge_enhancement=True):
    """
    Create a visualization with all three anatomical planes (axial, coronal, sagittal) side-by-side
    with original images and separate segmentation masks (not overlaid)
    
    Args:
        segmentation: 3D segmentation mask (depth, height, width)
        original_image: Optional original 3D image
        axial_slice_idx: Optional axial slice index, if None middle slice will be used
        coronal_slice_idx: Optional coronal slice index, if None middle slice will be used
        sagittal_slice_idx: Optional sagittal slice index, if None middle slice will be used
        tissue_colors: Dictionary mapping class IDs to RGB tuples
        tissue_names: Dictionary mapping class IDs to display names
        contrast_enhancement: Whether to apply contrast enhancement
        edge_enhancement: Whether to apply edge enhancement
        
    Returns:
        PIL Image object containing the three-plane visualization
    """
    logging.info(f"Creating three-plane side-by-side visualization")
    
    # Set default indices for slices if not provided
    if axial_slice_idx is None:
        axial_slice_idx = segmentation.shape[0] // 2
    if coronal_slice_idx is None:
        coronal_slice_idx = segmentation.shape[1] // 2
    if sagittal_slice_idx is None:
        sagittal_slice_idx = segmentation.shape[2] // 2
    
    # Create individual plane visualizations
    try:
        axial_img = create_side_by_side_visualization(
            segmentation, original_image, axial_slice_idx, 
            tissue_colors, tissue_names, upscale_factor=1.0,
            contrast_enhancement=contrast_enhancement, 
            edge_enhancement=edge_enhancement,
            view_type='axial'
        )
        
        coronal_img = create_side_by_side_visualization(
            segmentation, original_image, coronal_slice_idx, 
            tissue_colors, tissue_names, upscale_factor=1.0,
            contrast_enhancement=contrast_enhancement, 
            edge_enhancement=edge_enhancement,
            view_type='coronal'
        )
        
        sagittal_img = create_side_by_side_visualization(
            segmentation, original_image, sagittal_slice_idx, 
            tissue_colors, tissue_names, upscale_factor=1.0,
            contrast_enhancement=contrast_enhancement, 
            edge_enhancement=edge_enhancement,
            view_type='sagittal'
        )
    except Exception as e:
        logging.error(f"Error creating individual plane visualizations: {str(e)}")
        # Return a simple error message image
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, f"Error creating visualization: {str(e)}", 
                horizontalalignment='center', verticalalignment='center')
        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return Image.open(buf)
    
    # Get dimensions of images
    widths, heights = zip(*(i.size for i in [axial_img, coronal_img, sagittal_img]))
    
    # Calculate total metrics from the segmentation mask
    total_metastases = 0
    metastasis_volume_mm3 = 0.0
    
    # Use connected components analysis to count distinct regions in 3D
    # Assuming class 1 is metastasis
    metastasis_mask = (segmentation == 1).astype(np.uint8)
    if np.any(metastasis_mask):
        try:
            # For 3D connected components
            from scipy.ndimage import label
            labeled_mask, num_features = label(metastasis_mask)
            total_metastases = num_features
            
            # Calculate volume (assuming isotropic voxels of 1mm^3 - adjust as needed)
            # In a real application, you'd use actual voxel dimensions from the DICOM/NIFTI metadata
            voxel_volume = 1.0  # mm^3
            metastasis_volume_mm3 = np.sum(metastasis_mask) * voxel_volume
        except Exception as e:
            logging.error(f"Error calculating metastasis metrics: {str(e)}")
            total_metastases = "Error calculating"
            metastasis_volume_mm3 = 0.0
    
    # Create a new composite image with all three views and metrics
    # Vertical stacking for better mobile viewing
    total_width = max(widths)
    total_height = sum(heights) + 100  # Extra space for the metrics panel
    
    composite = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    
    # Paste the individual images
    y_offset = 0
    for img in [axial_img, coronal_img, sagittal_img]:
        # Center each image if width is less than total_width
        x_offset = (total_width - img.width) // 2
        composite.paste(img, (x_offset, y_offset))
        y_offset += img.height
    
    # Add metrics panel at the bottom
    metrics_height = 100
    metrics_img = Image.new('RGB', (total_width, metrics_height), (240, 240, 240))
    draw = ImageDraw.Draw(metrics_img)
    
    # Add text for metrics (we'd need to import ImageFont for this)
    try:
        from PIL import ImageFont
        font_large = ImageFont.truetype("Arial", 16)
        font_small = ImageFont.truetype("Arial", 14)
    except Exception:
        # Fallback if specific fonts not available
        font_large = None
        font_small = None
    
    # Draw metrics text
    title = "Segmentation Metrics"
    metastases_text = f"Total Metastases: {total_metastases}"
    volume_text = f"Metastasis Volume: {metastasis_volume_mm3:.2f} mm³"
    
    # Position text
    text_y = 10
    text_x = 20
    line_spacing = 25
    
    draw.text((text_x, text_y), title, fill=(0, 0, 0), font=font_large)
    draw.text((text_x, text_y + line_spacing), metastases_text, fill=(0, 0, 0), font=font_small)
    draw.text((text_x, text_y + 2 * line_spacing), volume_text, fill=(0, 0, 0), font=font_small)
    
    # Add the metrics panel to the bottom
    composite.paste(metrics_img, (0, y_offset))
    
    return composite
