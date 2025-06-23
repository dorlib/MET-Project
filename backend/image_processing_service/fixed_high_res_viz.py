import os
import io
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
import os
from scipy.ndimage import zoom
from viz_utils import enhance_segmentation_mask, enhance_original_image, create_colorized_overlay

# Force matplotlib to use a non-GUI backend to avoid display issues
matplotlib.use('Agg')

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
    # Ensure 3D arrays
    if len(segmentation.shape) < 3:
        segmentation = segmentation.reshape(1, segmentation.shape[0], segmentation.shape[1])
    if original_image is not None and len(original_image.shape) < 3:
        original_image = original_image.reshape(1, original_image.shape[0], original_image.shape[1])

    # Set default tissue colors if none provided
    if tissue_colors is None:
        # Use a better color palette for medical imaging
        tissue_colors = {
            0: (0.0, 0.0, 0.0),  # Background - black
            1: (0.8, 0.2, 0.2),  # Class 1 - red
            2: (0.2, 0.8, 0.2),  # Class 2 - green
            3: (0.2, 0.2, 0.8),  # Class 3 - blue
            4: (0.8, 0.8, 0.2),  # Class 4 - yellow
            5: (0.8, 0.2, 0.8),  # Class 5 - magenta
            6: (0.2, 0.8, 0.8),  # Class 6 - cyan
            7: (0.7, 0.4, 0.0),  # Class 7 - brown
            8: (0.6, 0.6, 0.6),  # Class 8 - gray
            9: (0.9, 0.6, 0.2)   # Class 9 - orange
        }

    # Set default tissue names if none provided
    if tissue_names is None:
        tissue_names = {
            0: "Background",
            1: "Class 1",
            2: "Class 2", 
            3: "Class 3",
            4: "Class 4",
            5: "Class 5",
            6: "Class 6",
            7: "Class 7",
            8: "Class 8",
            9: "Class 9"
        }

    # Get segmentation data in the right orientation based on view_type
    if view_type == 'axial':
        # Use the original data (axial slices)
        # If slice_idx is None, get the middle slice
        if slice_idx is None:
            slice_idx = segmentation.shape[0] // 2
            
        try:
            # Handle out of range indices with a safer approach
            if 0 <= slice_idx < segmentation.shape[0]:
                mask_slice = segmentation[slice_idx].copy()
            else:
                logging.warning(f"Slice index {slice_idx} out of range (0-{segmentation.shape[0]-1}), using middle slice")
                mask_slice = segmentation[segmentation.shape[0] // 2].copy()
        except Exception as e:
            logging.error(f"Error extracting segmentation slice: {str(e)}")
            # Fallback to a zeros array if extraction fails
            mask_slice = np.zeros((256, 256), dtype=int)  # Default size fallback
    
    elif view_type == 'coronal':
        # Get coronal view (slice along the y-axis)
        # If slice_idx is None, get the middle slice
        if slice_idx is None:
            slice_idx = segmentation.shape[1] // 2
            
        try:
            # Handle out of range indices with a safer approach
            if 0 <= slice_idx < segmentation.shape[1]:
                mask_slice = segmentation[:, slice_idx, :].copy()
            else:
                logging.warning(f"Coronal slice index {slice_idx} out of range (0-{segmentation.shape[1]-1}), using middle slice")
                mask_slice = segmentation[:, segmentation.shape[1] // 2, :].copy()
        except Exception as e:
            logging.error(f"Error extracting coronal segmentation slice: {str(e)}")
            # Fallback to a zeros array if extraction fails
            mask_slice = np.zeros((segmentation.shape[0], segmentation.shape[2]), dtype=int)
    
    elif view_type == 'sagittal':
        # Get sagittal view (slice along the x-axis)
        # If slice_idx is None, get the middle slice
        if slice_idx is None:
            slice_idx = segmentation.shape[2] // 2
            
        try:
            # Handle out of range indices with a safer approach
            if 0 <= slice_idx < segmentation.shape[2]:
                mask_slice = segmentation[:, :, slice_idx].copy()
            else:
                logging.warning(f"Sagittal slice index {slice_idx} out of range (0-{segmentation.shape[2]-1}), using middle slice")
                mask_slice = segmentation[:, :, segmentation.shape[2] // 2].copy()
        except Exception as e:
            logging.error(f"Error extracting sagittal segmentation slice: {str(e)}")
            # Fallback to a zeros array if extraction fails
            mask_slice = np.zeros((segmentation.shape[0], segmentation.shape[1]), dtype=int)
    
    else:
        logging.warning(f"Invalid view_type '{view_type}', using 'axial'")
        # Default to axial view if view_type is not recognized
        if slice_idx is None:
            slice_idx = segmentation.shape[0] // 2
            
        try:
            if 0 <= slice_idx < segmentation.shape[0]:
                mask_slice = segmentation[slice_idx].copy()
            else:
                mask_slice = segmentation[segmentation.shape[0] // 2].copy()
        except:
            mask_slice = np.zeros((256, 256), dtype=int)

    # Create RGBA array for the colors (4 channels)
    colors = np.zeros((*mask_slice.shape, 4))
    
    # Apply colors to the mask
    for class_id, color in tissue_colors.items():
        mask = mask_slice == class_id
        if np.any(mask):
            colors[mask] = (*color, 1.0)  # RGB + alpha
    
    # If no segmentation found, set all transparent
    if not np.any(colors):
        colors[:] = (0, 0, 0, 0)
    
    # Process the original image if available
    if original_image is not None:
        # Get the original image slice in the right orientation based on view_type
        if view_type == 'axial':
            try:
                if slice_idx < original_image.shape[0]:
                    orig_slice = original_image[slice_idx].copy()
                else:
                    orig_slice = None
            except Exception as e:
                logging.error(f"Error extracting original image slice: {str(e)}")
                orig_slice = None
                
        elif view_type == 'coronal':
            try:
                if slice_idx < original_image.shape[1]:
                    orig_slice = original_image[:, slice_idx, :].copy()
                else:
                    orig_slice = None
            except Exception as e:
                logging.error(f"Error extracting coronal original image slice: {str(e)}")
                orig_slice = None
                
        elif view_type == 'sagittal':
            try:
                if slice_idx < original_image.shape[2]:
                    orig_slice = original_image[:, :, slice_idx].copy()
                else:
                    orig_slice = None
            except Exception as e:
                logging.error(f"Error extracting sagittal original image slice: {str(e)}")
                orig_slice = None
        else:
            # Default to axial if unknown view type
            try:
                if slice_idx < original_image.shape[0]:
                    orig_slice = original_image[slice_idx].copy()
                else:
                    orig_slice = None
            except Exception as e:
                logging.error(f"Error extracting original image slice: {str(e)}")
                orig_slice = None
        
        # Apply advanced image enhancement with our utility function
        if orig_slice is not None:
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
    
    # Create the figure and plot
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    
    # Show the original image as the background
    ax.imshow(orig_slice, cmap='gray', alpha=1.0)
    
    # Overlay the segmentation with alpha
    ax.imshow(colors)
    
    # Remove all axes, ticks, and labels for a clean image
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Add title with slice information
    view_info = f"{view_type.capitalize()} Slice {slice_idx}" if slice_idx is not None else f"Middle {view_type.capitalize()} Slice"
    ax.set_title(f"Segmentation Overlay - {view_info}", fontsize=16, pad=10)
    
    # Add legend with class information
    legend_handles = []
    legend_labels = []
    
    # Find which classes are present in this slice
    present_classes = np.unique(mask_slice)
    
    # Only show legend entries for classes that are visible in the slice
    for class_id in present_classes:
        if class_id in tissue_colors and class_id != 0:  # Skip background
            color_patch = plt.Rectangle((0, 0), 1, 1, fc=tissue_colors[class_id])
            legend_handles.append(color_patch)
            name = tissue_names.get(class_id, f"Class {class_id}")
            legend_labels.append(name)
    
    # Only add legend if there are classes to show
    if legend_handles:
        ax.legend(legend_handles, legend_labels, loc='lower right', fontsize=12)
    
    # Save the figure to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
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
            img = enhancer.enhance(1.1)  # 1.0 is original, >1 is more contrast
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
    
    # Get segmentation data in the right orientation based on view_type
    if view_type == 'axial':
        # Use the original data (axial slices)
        # If slice_idx is None, get the middle slice
        if slice_idx is None:
            slice_idx = segmentation.shape[0] // 2
            
        try:
            # Handle out of range indices with a safer approach
            if 0 <= slice_idx < segmentation.shape[0]:
                mask_slice = segmentation[slice_idx].copy()
            else:
                logging.warning(f"Slice index {slice_idx} out of range (0-{segmentation.shape[0]-1}), using middle slice")
                mask_slice = segmentation[segmentation.shape[0] // 2].copy()
        except Exception as e:
            logging.error(f"Error extracting segmentation slice: {str(e)}")
            # Fallback to a zeros array if extraction fails
            mask_slice = np.zeros((256, 256), dtype=int)  # Default size fallback
    
    elif view_type == 'coronal':
        # Get coronal view (slice along the y-axis)
        # If slice_idx is None, get the middle slice
        if slice_idx is None:
            slice_idx = segmentation.shape[1] // 2
            
        try:
            # Handle out of range indices with a safer approach
            if 0 <= slice_idx < segmentation.shape[1]:
                mask_slice = segmentation[:, slice_idx, :].copy()
            else:
                logging.warning(f"Coronal slice index {slice_idx} out of range (0-{segmentation.shape[1]-1}), using middle slice")
                mask_slice = segmentation[:, segmentation.shape[1] // 2, :].copy()
        except Exception as e:
            logging.error(f"Error extracting coronal segmentation slice: {str(e)}")
            # Fallback to a zeros array if extraction fails
            mask_slice = np.zeros((segmentation.shape[0], segmentation.shape[2]), dtype=int)
    
    elif view_type == 'sagittal':
        # Get sagittal view (slice along the x-axis)
        # If slice_idx is None, get the middle slice
        if slice_idx is None:
            slice_idx = segmentation.shape[2] // 2
            
        try:
            # Handle out of range indices with a safer approach
            if 0 <= slice_idx < segmentation.shape[2]:
                mask_slice = segmentation[:, :, slice_idx].copy()
            else:
                logging.warning(f"Sagittal slice index {slice_idx} out of range (0-{segmentation.shape[2]-1}), using middle slice")
                mask_slice = segmentation[:, :, segmentation.shape[2] // 2].copy()
        except Exception as e:
            logging.error(f"Error extracting sagittal segmentation slice: {str(e)}")
            # Fallback to a zeros array if extraction fails
            mask_slice = np.zeros((segmentation.shape[0], segmentation.shape[1]), dtype=int)
    
    else:
        logging.warning(f"Invalid view_type '{view_type}', using 'axial'")
        # Default to axial view if view_type is not recognized
        if slice_idx is None:
            slice_idx = segmentation.shape[0] // 2
            
        try:
            if 0 <= slice_idx < segmentation.shape[0]:
                mask_slice = segmentation[slice_idx].copy()
            else:
                mask_slice = segmentation[segmentation.shape[0] // 2].copy()
        except:
            mask_slice = np.zeros((256, 256), dtype=int)
    
    # Create RGBA array for the colors (4 channels)
    colors = np.zeros((*mask_slice.shape, 4))
    
    # Apply colors to the mask
    for class_id, color in tissue_colors.items():
        mask = mask_slice == class_id
        if np.any(mask):
            colors[mask] = (*color, 1.0)  # RGB + alpha
    
    # If no segmentation found, set all transparent
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
    
    # Create a figure with the visualization at higher resolution
    dpi = 200
    
    # Calculate figure size based on image dimensions to maintain aspect ratio
    height, width = mask_slice.shape
    aspect_ratio = width / height
    
    # Double the width to fit both images side by side
    fig_height = max(8, height / 75)
    fig_width = fig_height * aspect_ratio * 2.1  # *2 for side-by-side, extra 0.1 for spacing
    
    # Create the figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=dpi)
    
    # Show the original image on the left
    if original_image is not None and orig_slice is not None:
        ax1.imshow(orig_slice, cmap='gray')
    else:
        ax1.imshow(np.zeros_like(mask_slice), cmap='gray')
        
    ax1.set_title(f"Original Image - {view_type.capitalize()} Slice {slice_idx}")
    ax1.set_axis_off()
    
    # Show the segmentation on the right
    if np.any(colors):
        ax2.imshow(colors)
        
        # Add legend to second axis
        legend_handles = []
        legend_labels = []
        
        # Find which classes are present in this slice
        present_classes = np.unique(mask_slice)
        
        # Only show legend entries for classes that are visible in the slice
        for class_id in present_classes:
            if class_id in tissue_colors and class_id != 0:  # Skip background
                color_patch = plt.Rectangle((0, 0), 1, 1, fc=tissue_colors[class_id])
                legend_handles.append(color_patch)
                name = tissue_names.get(class_id, f"Class {class_id}") if tissue_names else f"Class {class_id}"
                legend_labels.append(name)
        
        # Only add legend if there are classes to show
        if legend_handles:
            ax2.legend(legend_handles, legend_labels, loc='lower right', fontsize=10)
    else:
        ax2.imshow(np.zeros_like(mask_slice), cmap='gray')
    
    ax2.set_title(f"Segmentation - {view_type.capitalize()} Slice {slice_idx}")
    ax2.set_axis_off()
    
    plt.tight_layout(pad=1.0)
    
    # Save the figure to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
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
            img = enhancer.enhance(1.1)  # 1.0 is original, >1 is more contrast
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
        segmentation: 3D segmentation mask
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
    # Set default slice indices to the middle of each dimension
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
        return None
    
    # Get the dimensions of the individual images
    axial_width, axial_height = axial_img.size
    coronal_width, coronal_height = coronal_img.size
    sagittal_width, sagittal_height = sagittal_img.size
    
    # Create a composite image with all three views stacked vertically
    composite_width = max(axial_width, coronal_width, sagittal_width)
    composite_height = axial_height + coronal_height + sagittal_height
    
    composite = Image.new('RGB', (composite_width, composite_height), color='white')
    
    # Paste the individual views
    y_offset = 0
    for img in [axial_img, coronal_img, sagittal_img]:
        width, height = img.size
        x_offset = (composite_width - width) // 2  # Center horizontally
        composite.paste(img, (x_offset, y_offset))
        y_offset += height
    
    return composite

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
        PIL Image object containing multiple slices
    """
    if segmentation.shape[0] < num_slices:
        num_slices = segmentation.shape[0]
    
    # Calculate slice indices to show a good distribution
    slice_indices = np.linspace(0, segmentation.shape[0]-1, num_slices, dtype=int)
    
    # Generate visualizations for each slice
    slice_imgs = []
    for slice_idx in slice_indices:
        img = create_high_res_visualization(
            segmentation, original_image, slice_idx,
            tissue_colors=tissue_colors, tissue_names=tissue_names,
            contrast_enhancement=contrast_enhancement,
            edge_enhancement=edge_enhancement
        )
        slice_imgs.append(img)
    
    # Get the dimensions of the images
    img_width, img_height = slice_imgs[0].size
    
    # Create a composite image with all slices in a row
    composite_width = img_width * num_slices
    composite_height = img_height
    
    composite = Image.new('RGB', (composite_width, composite_height), color='white')
    
    # Paste the slices side by side
    for i, img in enumerate(slice_imgs):
        composite.paste(img, (i * img_width, 0))
    
    return composite
