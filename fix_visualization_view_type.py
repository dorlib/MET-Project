#!/usr/bin/env python3
"""
Script to add view_type support to the high-resolution visualization functions
"""
import os
import shutil
import re

def add_view_type_to_visualization_function():
    """
    Modify the create_high_res_visualization function to support view_type parameter
    for axial, coronal, and sagittal views.
    """
    # Path to the visualization file
    viz_file = os.path.join(os.getcwd(), 'fixed_high_res_viz.py')
    
    # Make sure the file exists
    if not os.path.exists(viz_file):
        print(f"File not found: {viz_file}")
        return False
    
    # Create a backup (just in case)
    backup_file = viz_file + '.bak'
    shutil.copy2(viz_file, backup_file)
    print(f"Created backup: {backup_file}")
    
    # Read the file contents
    with open(viz_file, 'r') as f:
        content = f.read()
    
    # Update the function signature to include view_type parameter
    updated_content = re.sub(
        r'def create_high_res_visualization\(segmentation, original_image=None, slice_idx=None, tissue_colors=None, tissue_names=None,\s+upscale_factor=1.0, contrast_enhancement=True, edge_enhancement=True\):',
        'def create_high_res_visualization(segmentation, original_image=None, slice_idx=None, tissue_colors=None, tissue_names=None,\n                                 upscale_factor=1.0, contrast_enhancement=True, edge_enhancement=True, view_type=\'axial\'):',
        content
    )
    
    # Update the docstring to include view_type parameter
    updated_content = re.sub(
        r'(\s+Args:\s+.*?\s+edge_enhancement: Whether to apply edge enhancement\s+)(\s+Returns:)',
        r'\1    view_type: Type of view - \'axial\' (default), \'coronal\', or \'sagittal\'\n\n\2',
        updated_content, 
        flags=re.DOTALL
    )
    
    # Update the logging statement to include view_type
    updated_content = re.sub(
        r'(\s+logging.info\(f"Creating high-res visualization with slice_idx={slice_idx}, upscale_factor={upscale_factor}"\))',
        r'\1\n    logging.info(f"View type: {view_type}")',
        updated_content
    )
    
    # Add view type handling before slice extraction
    slice_handling_code = """
    # Determine which dimension to slice based on view_type
    slice_dim = 0
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
"""
    
    # Insert slice dimension handling after the tissue_names block
    updated_content = re.sub(
        r'(\s+if tissue_names is None:.*?\s+}\s+\n)',
        r'\1' + slice_handling_code,
        updated_content,
        flags=re.DOTALL
    )
    
    # Update slice index determination to handle different dimensions
    slice_idx_code = """
    # Handle slice index based on the selected dimension
    if slice_idx is None:
        # Use middle slice of the appropriate dimension
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
            # Default to middle slice of the appropriate dimension
            if slice_dim == 0:
                slice_idx = segmentation.shape[0] // 2
            elif slice_dim == 1:
                slice_idx = segmentation.shape[1] // 2
            else:  # slice_dim == 2
                slice_idx = segmentation.shape[2] // 2
            logging.info(f"Using middle slice instead: {slice_idx} for {slice_description} view")
        
        # Make sure the slice index is within bounds of selected dimension
        max_idx = 0
        if slice_dim == 0:
            max_idx = segmentation.shape[0] - 1
        elif slice_dim == 1:
            max_idx = segmentation.shape[1] - 1
        else:  # slice_dim == 2
            max_idx = segmentation.shape[2] - 1
            
        if slice_idx < 0:
            logging.warning(f"Slice index {slice_idx} was negative, using slice 0 instead for {slice_description} view")
            slice_idx = 0
        elif slice_idx > max_idx:
            logging.warning(f"Slice index {slice_idx} was too large (max={max_idx}), using slice {max_idx} instead for {slice_description} view")
            slice_idx = max_idx
    
    logging.info(f"Final {slice_description} slice index: {slice_idx}")
"""
    
    # Replace the existing slice_idx handling with the new code that handles different dimensions
    updated_content = re.sub(
        r'(\s+# Determine which slice to use and ensure it\'s valid\s+if slice_idx is None:.*?logging.info\(f"Final slice index: \{slice_idx\}.*?\))',
        slice_idx_code,
        updated_content,
        flags=re.DOTALL
    )
    
    # Update the mask slice extraction to handle different dimensions
    slice_extraction_code = """
    # Extract the requested slice with error handling
    try:
        if slice_dim == 0:  # Axial
            mask_slice = segmentation[slice_idx, :, :]
            if original_image is not None and slice_idx < original_image.shape[0]:
                orig_slice = original_image[slice_idx, :, :]
            else:
                orig_slice = None
        elif slice_dim == 1:  # Coronal
            mask_slice = segmentation[:, slice_idx, :]
            if original_image is not None and slice_idx < original_image.shape[1]:
                orig_slice = original_image[:, slice_idx, :]
            else:
                orig_slice = None
        else:  # Sagittal
            mask_slice = segmentation[:, :, slice_idx]
            if original_image is not None and slice_idx < original_image.shape[2]:
                orig_slice = original_image[:, :, slice_idx]
            else:
                orig_slice = None
        
        if len(segmentation.shape) > 3:  # Handle one-hot encoding if present
            mask_slice = np.argmax(mask_slice, axis=-1) if mask_slice.shape[-1] > 1 else mask_slice
            
        logging.info(f"Extracted {slice_description} mask slice with shape: {mask_slice.shape}, unique values: {np.unique(mask_slice)}")
    
    except Exception as e:
        logging.error(f"Error extracting {slice_description} mask slice: {str(e)}, details: {type(e).__name__}")
        # Fallback to middle axial slice if there was an error
        mask_slice = segmentation[segmentation.shape[0] // 2, :, :]
        if original_image is not None:
            orig_slice = original_image[segmentation.shape[0] // 2, :, :]
        else:
            orig_slice = None
        logging.info(f"Using fallback middle axial slice instead")
    
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
    if original_image is not None and orig_slice is not None:
"""
    
    # Replace the existing slice extraction with the new code that handles different dimensions
    updated_content = re.sub(
        r'(\s+# Extract the requested slice with error handling\s+try:.*?# If original image provided, use it as background with enhanced contrast\s+if original_image is not None:)',
        slice_extraction_code,
        updated_content,
        flags=re.DOTALL
    )
    
    # Update the title to include the slice description
    updated_content = re.sub(
        r'(\s+# Improved title with slice information\s+plt\.title\(f"Segmentation Visualization \(Slice \{slice_idx\}/\{segmentation\.shape\[0\]-1\}\)", \s+fontsize=14, fontweight=\'bold\'\))',
        r'    # Improved title with slice information\n    plt.title(f"{slice_description} Segmentation (Slice {slice_idx})", \n              fontsize=14, fontweight=\'bold\')',
        updated_content,
        flags=re.DOTALL
    )
    
    # Write the updated content back to the file
    with open(viz_file, 'w') as f:
        f.write(updated_content)
    
    print(f"Updated {viz_file} with view_type support")
    
    # Also update the backend image processing service copy
    backend_viz_file = os.path.join(os.getcwd(), 'backend', 'image_processing_service', 'fixed_high_res_viz.py')
    if os.path.exists(backend_viz_file):
        shutil.copy2(viz_file, backend_viz_file)
        print(f"Updated backend copy: {backend_viz_file}")
    
    return True

if __name__ == "__main__":
    print("Adding view_type support to high-resolution visualization function...")
    if add_view_type_to_visualization_function():
        print("Done! View type support has been added successfully.")
    else:
        print("Failed to update visualization function.")
