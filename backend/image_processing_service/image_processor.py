#!/usr/bin/env python3
# image_processing_service/image_processor.py - Image processing service for metastasis analysis

import os
import numpy as np
from flask import Flask, jsonify, request, send_file, Response
import logging
from scipy import ndimage
import json
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import io
from PIL import Image
from skimage import measure
import pandas as pd
import nibabel as nib
from scipy.ndimage import zoom
import cv2
import matplotlib
# Import our high-resolution visualization module
from high_res_viz import create_high_res_visualization, generate_high_res_multi_slice_view
matplotlib.use('Agg')  # Use non-interactive backend for matplotlib

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

RESULTS_FOLDER = '/app/results'
METASTASIS_CLASS = 3  # Assuming class 3 is for metastasis in the segmentation masks
EDEMA_CLASS = 2       # Assuming class 2 is for edema
TUMOR_CORE_CLASS = 1  # Assuming class 1 is for tumor core
VOXEL_VOLUME_MM3 = 1.0  # Default voxel volume in mm³ (can be adjusted based on scan parameters)
TISSUE_COLORS = {
    METASTASIS_CLASS: (1.0, 0.0, 0.0),       # Red for metastasis
    EDEMA_CLASS: (0.0, 1.0, 0.0),            # Green for edema
    TUMOR_CORE_CLASS: (0.0, 0.0, 1.0)        # Blue for tumor core
}
TISSUE_NAMES = {
    METASTASIS_CLASS: "Metastasis",
    EDEMA_CLASS: "Edema",
    TUMOR_CORE_CLASS: "Tumor Core"
}

# Enhanced utility functions for visualization and analysis
def load_volume_data(file_path):
    """
    Load volume data from .npy or .nii/.nii.gz files
    
    Args:
        file_path: Path to the volume file
        
    Returns:
        numpy.ndarray: The loaded volume data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    if file_path.endswith('.npy'):
        return np.load(file_path)
    elif file_path.endswith('.nii') or file_path.endswith('.nii.gz'):
        nii_img = nib.load(file_path)
        return np.asarray(nii_img.dataobj)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def create_colormap_visualization(segmentation, original_image=None, slice_idx=None):
    """
    Create a high-resolution colormap visualization of a segmentation mask
    
    Args:
        segmentation: 3D segmentation mask
        original_image: Optional original 3D image
        slice_idx: Optional slice index, if None middle slice will be used
        
    Returns:
        PIL Image object containing the visualization
    """
    # Determine which slice to use
    if slice_idx is None:
        slice_idx = segmentation.shape[0] // 2  # Middle slice
    
    # Extract the requested slice
    if len(segmentation.shape) == 3:
        mask_slice = segmentation[slice_idx, :, :]
    else:
        # If the mask has an additional dimension (e.g., one-hot encoding)
        mask_slice = np.argmax(segmentation[slice_idx, :, :], axis=-1) if segmentation.shape[-1] > 1 else segmentation[slice_idx, :, :]
    
    # Create a colored visualization
    colors = np.zeros((*mask_slice.shape, 4))
    
    # Apply colors for each tissue type
    for class_id, color in TISSUE_COLORS.items():
        mask = mask_slice == class_id
        if np.any(mask):
            colors[mask] = (*color, 1.0)  # RGB + alpha
    
    # If no segmentation found, make entire image transparent
    if not np.any(colors):
        colors[:] = (0, 0, 0, 0)
    
    # If original image provided, use it as background
    if original_image is not None:
        if slice_idx < original_image.shape[0]:
            orig_slice = original_image[slice_idx]
            # Normalize the original image for display
            if orig_slice.min() != orig_slice.max():
                orig_slice = (orig_slice - orig_slice.min()) / (orig_slice.max() - orig_slice.min())
                
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_slice = clahe.apply((orig_slice * 255).astype(np.uint8))
            orig_slice = enhanced_slice / 255.0
        else:
            # If slice index out of range, create blank background
            orig_slice = np.zeros_like(mask_slice, dtype=float)
    else:
        # Create a grayscale background if no original image
        orig_slice = np.zeros_like(mask_slice, dtype=float)
    
    # Create a figure with the visualization at higher resolution
    dpi = 150  # Higher DPI for better resolution
    
    # Calculate figure size based on image dimensions to maintain aspect ratio
    height, width = mask_slice.shape
    aspect_ratio = width / height
    
    # Set a minimum size to ensure good quality
    fig_height = max(12, height / 100)  # Reasonable size in inches
    fig_width = fig_height * aspect_ratio
    
    plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    
    # Plot the original image as grayscale
    plt.imshow(orig_slice, cmap='gray', interpolation='nearest')
    
    # Overlay the colormap visualization
    plt.imshow(colors, alpha=0.7, interpolation='nearest')
    
    # Improved title with slice information
    plt.title(f"Segmentation Visualization (Slice {slice_idx}/{segmentation.shape[0]-1})", fontsize=14)
    plt.axis('off')  # Hide axes
    
    # Reduce padding to maximize image size
    plt.tight_layout(pad=0.5)
    
    # Convert the figure to a high-quality PNG image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, dpi=dpi)
    plt.close()
    buf.seek(0)
    
    return Image.open(buf)

def generate_3d_projection(segmentation, class_id=None):
    """
    Generate a 3D projection visualization of the segmentation
    
    Args:
        segmentation: 3D segmentation mask
        class_id: Optional specific class to highlight, if None show all classes
        
    Returns:
        PIL Image object containing the 3D projection
    """
    if class_id is not None:
        # Extract specific class
        binary_mask = (segmentation == class_id).astype(np.uint8)
    else:
        # Use all non-zero values
        binary_mask = (segmentation > 0).astype(np.uint8)
    
    # Create three projections (axial, sagittal, coronal)
    axial = np.max(binary_mask, axis=0)
    sagittal = np.max(binary_mask, axis=1)
    coronal = np.max(binary_mask, axis=2)
    
    # Create a figure with the three projections
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Choose color based on class_id if provided
    color = TISSUE_COLORS.get(class_id, (1, 0, 0)) if class_id is not None else (1, 1, 1)
    
    # Create a colormap from the color
    cmap = ListedColormap(['black', color])
    
    # Plot the three projections
    axes[0].imshow(axial, cmap=cmap)
    axes[0].set_title('Axial Projection')
    axes[0].axis('off')
    
    axes[1].imshow(sagittal, cmap=cmap)
    axes[1].set_title('Sagittal Projection')
    axes[1].axis('off')
    
    axes[2].imshow(coronal, cmap=cmap)
    axes[2].set_title('Coronal Projection')
    axes[2].axis('off')
    
    # Add title based on class if available
    if class_id is not None:
        class_name = TISSUE_NAMES.get(class_id, f"Class {class_id}")
        plt.suptitle(f"3D Projections - {class_name}")
    else:
        plt.suptitle("3D Projections - All Classes")
    
    plt.tight_layout()
    
    # Convert the figure to a PNG image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    return Image.open(buf)

def generate_multi_slice_view(segmentation, original_image=None, num_slices=5):
    """
    Generate multiple slice views of the segmentation
    
    Args:
        segmentation: 3D segmentation mask
        original_image: Optional original 3D image
        num_slices: Number of slices to show
        
    Returns:
        PIL Image object containing the multi-slice visualization
    """
    depth = segmentation.shape[0]
    
    # Calculate slice indices
    indices = np.linspace(0, depth - 1, num_slices).astype(int)
    
    # Create a figure with multiple slices
    fig, axes = plt.subplots(1, num_slices, figsize=(4 * num_slices, 4))
    
    for i, slice_idx in enumerate(indices):
        # Extract the slice
        mask_slice = segmentation[slice_idx]
        
        # Create colored mask
        colored_mask = np.zeros((*mask_slice.shape, 4))
        
        # Apply colors for each tissue type
        for class_id, color in TISSUE_COLORS.items():
            mask = mask_slice == class_id
            if np.any(mask):
                colored_mask[mask] = (*color, 0.7)  # RGB + alpha
        
        # Get original image slice if available
        if original_image is not None and slice_idx < original_image.shape[0]:
            orig_slice = original_image[slice_idx]
            # Normalize for display
            if orig_slice.min() != orig_slice.max():
                orig_slice = (orig_slice - orig_slice.min()) / (orig_slice.max() - orig_slice.min())
        else:
            # Create a blank background
            orig_slice = np.zeros_like(mask_slice, dtype=float)
        
        # Plot the original slice
        axes[i].imshow(orig_slice, cmap='gray')
        
        # Overlay the colored mask
        axes[i].imshow(colored_mask)
        
        axes[i].set_title(f"Slice {slice_idx}")
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Convert the figure to a PNG image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    return Image.open(buf)

def analyze_tissue_type(segmentation, class_id, voxel_volume_mm3=1.0):
    """
    Analyze a specific tissue type in the segmentation mask
    
    Args:
        segmentation: 3D segmentation mask
        class_id: Class ID to analyze
        voxel_volume_mm3: Volume of each voxel in mm³
        
    Returns:
        dict: Analysis results including count, volumes, and properties
    """
    # Extract the specified class
    binary_mask = (segmentation == class_id).astype(np.int32)
    
    # Skip analysis if no regions found
    if not np.any(binary_mask):
        return {
            "count": 0,
            "regions": [],
            "total_volume": 0,
            "message": f"No {TISSUE_NAMES.get(class_id, f'Class {class_id}')} regions found"
        }
    
    # Identify individual connected components
    labeled_mask, num_features = ndimage.label(binary_mask)
    
    # Calculate properties of each region
    regions = []
    total_volume = 0
    
    # Get region properties
    props = measure.regionprops(labeled_mask)
    
    for i, region in enumerate(props):
        # Calculate volume
        volume_mm3 = region.area * voxel_volume_mm3
        total_volume += volume_mm3
        
        # Get centroid (x, y, z)
        centroid = region.centroid
        
        # Calculate sphericity (how close to a perfect sphere)
        # 0 = line, 1 = perfect sphere
        if region.area > 0 and region.equivalent_diameter > 0:
            surface_area = measure.perimeter(region.image)
            sphericity = (6 * np.sqrt(np.pi) * np.power(region.area, 1.5)) / (surface_area * np.sqrt(surface_area))
            if sphericity > 1:  # Correct for numerical errors
                sphericity = 1.0
        else:
            sphericity = 0
        
        regions.append({
            "id": i + 1,
            "volume_mm3": float(volume_mm3),
            "centroid": [float(c) for c in centroid],
            "equivalent_diameter_mm": float(region.equivalent_diameter),
            "sphericity": float(sphericity)
        })
    
    # Sort by volume (largest first)
    regions.sort(key=lambda r: r["volume_mm3"], reverse=True)
    
    return {
        "count": num_features,
        "regions": regions,
        "total_volume": float(total_volume),
        "average_volume": float(total_volume / num_features) if num_features > 0 else 0
    }

def analyze_connected_components(segmentation, class_id, voxel_volume_mm3=1.0):
    """
    Analyze individual connected components for a specific class
    
    Args:
        segmentation: 3D segmentation mask
        class_id: Class ID to analyze
        voxel_volume_mm3: Volume of each voxel in mm³
        
    Returns:
        tuple: (labeled_mask, analysis_results)
    """
    # Extract binary mask for this class
    binary = (segmentation == class_id).astype(np.int32)
    
    # Skip if no regions found
    if not np.any(binary):
        return None, {
            "count": 0,
            "regions": [],
            "total_volume": 0,
            "message": f"No {TISSUE_NAMES.get(class_id, f'Class {class_id}')} regions found"
        }
    
    # Split into connected components
    labeled_mask, num_instances = ndimage.label(binary)
    
    # Analyze each instance
    regions = []
    total_volume = 0
    
    for inst_idx in range(1, num_instances + 1):
        # Extract this instance
        instance_mask = (labeled_mask == inst_idx)
        
        # Calculate volume
        voxel_count = np.sum(instance_mask)
        volume_mm3 = voxel_count * voxel_volume_mm3
        total_volume += volume_mm3
        
        # Find centroid
        coords = np.where(instance_mask)
        if len(coords[0]) > 0:
            centroid = [float(np.mean(coords[i])) for i in range(3)]
        else:
            centroid = [0, 0, 0]
        
        # Store data for this region
        regions.append({
            "id": inst_idx,
            "volume_mm3": float(volume_mm3),
            "voxel_count": int(voxel_count),
            "centroid": centroid
        })
    
    # Sort by volume (largest first)
    regions.sort(key=lambda r: r["volume_mm3"], reverse=True)
    
    return labeled_mask, {
        "count": num_instances,
        "regions": regions,
        "total_volume": float(total_volume),
        "average_volume": float(total_volume / num_instances) if num_instances > 0 else 0
    }

def generate_lesion_visualization(segmentation, original_image=None, slice_idx=None):
    """
    Generate a visualization of individual lesions with different colors
    
    Args:
        segmentation: 3D segmentation mask
        original_image: Optional original 3D image
        slice_idx: Optional slice index, if None middle slice will be used
        
    Returns:
        PIL Image object containing the visualization
    """
    # Determine which slice to use
    if slice_idx is None:
        slice_idx = segmentation.shape[0] // 2  # Middle slice
    
    # Extract the requested slice
    if len(segmentation.shape) == 3:
        mask_slice = segmentation[slice_idx, :, :]
    else:
        mask_slice = np.argmax(segmentation[slice_idx, :, :], axis=-1) if segmentation.shape[-1] > 1 else segmentation[slice_idx, :, :]
    
    # Create a figure
    plt.figure(figsize=(10, 8))
    
    # If original image provided, use it as background
    if original_image is not None:
        if slice_idx < original_image.shape[0]:
            orig_slice = original_image[slice_idx]
            # Normalize the original image for display
            if orig_slice.min() != orig_slice.max():
                orig_slice = (orig_slice - orig_slice.min()) / (orig_slice.max() - orig_slice.min())
            plt.imshow(orig_slice, cmap='gray')
        else:
            plt.imshow(np.zeros_like(mask_slice), cmap='gray')
    else:
        plt.imshow(np.zeros_like(mask_slice), cmap='gray')
    
    # Process each tissue type with a different color scheme
    for class_id in TISSUE_COLORS.keys():
        # Skip if not present in this slice
        if not np.any(mask_slice == class_id):
            continue
            
        # Extract binary mask for this class in this slice
        binary_slice = (mask_slice == class_id).astype(np.int32)
        
        # Find connected components (lesions) in 2D
        labeled_slice, num_lesions = ndimage.label(binary_slice)
        
        # Create a colormap with unique colors for each lesion
        # Background is transparent (alpha=0)
        cmap = np.zeros((num_lesions + 1, 4))
        
        # Base color for this tissue type
        base_color = np.array(TISSUE_COLORS.get(class_id, (1.0, 1.0, 0.0)))
        
        # Generate slightly different colors for each instance
        for i in range(1, num_lesions + 1):
            # Vary the color slightly to distinguish instances
            color_variation = 0.2 * (np.random.random(3) - 0.5)
            color = np.clip(base_color + color_variation, 0, 1)
            cmap[i, :3] = color
            cmap[i, 3] = 0.7  # Alpha
            
        # Create a custom colormap
        lesion_cmap = ListedColormap(cmap)
        
        # Show the labeled lesions
        plt.imshow(labeled_slice, cmap=lesion_cmap)
    
    class_names = ", ".join([TISSUE_NAMES.get(c, f"Class {c}") for c in TISSUE_COLORS.keys() if np.any(mask_slice == c)])
    plt.title(f"Lesion Visualization - {class_names} (Slice {slice_idx})")
    plt.axis('off')
    
    # Convert the figure to a PNG image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    return Image.open(buf)

def generate_multi_class_analysis(segmentation, original_image=None, voxel_volume_mm3=1.0):
    """
    Generate a comprehensive multi-class analysis of the segmentation mask
    
    Args:
        segmentation: 3D segmentation mask
        original_image: Optional original 3D image
        voxel_volume_mm3: Volume of each voxel in mm³
        
    Returns:
        dict: Comprehensive analysis results
    """
    # Identify all classes in the segmentation
    classes = np.unique(segmentation)
    classes = classes[classes > 0]  # Skip background
    
    analysis = {
        "total_classes_found": len(classes),
        "classes": {},
        "overall_summary": {
            "total_regions": 0,
            "total_volume": 0
        }
    }
    
    # Analyze each class
    for class_id in classes:
        class_name = TISSUE_NAMES.get(int(class_id), f"Class {class_id}")
        _, class_analysis = analyze_connected_components(segmentation, class_id, voxel_volume_mm3)
        
        analysis["classes"][class_name] = class_analysis
        analysis["overall_summary"]["total_regions"] += class_analysis["count"]
        analysis["overall_summary"]["total_volume"] += class_analysis["total_volume"]
    
    return analysis

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "image-processing-service"})

@app.route('/analyze/<job_id>', methods=['GET'])
def analyze_segmentation(job_id):
    """
    Analyze a segmentation mask to count metastases and calculate their volumes
    """
    if not job_id:
        return jsonify({
            "error": "Missing job ID"
        }), 400
        
    # Validate job_id format to prevent path traversal
    if '/' in job_id or '\\' in job_id or '..' in job_id:
        return jsonify({
            "error": "Invalid job ID format",
            "job_id": job_id
        }), 400
    
    # Find the prediction file
    pred_path = os.path.join(RESULTS_FOLDER, f"{job_id}_prediction.npy")
    
    if not os.path.exists(pred_path):
        logging.warning(f"Segmentation not found for job ID: {job_id}")
        return jsonify({
            "error": "Segmentation not found",
            "job_id": job_id
        }), 404
    
    try:
        # Load the segmentation mask
        try:
            segmentation = np.load(pred_path)
        except Exception as e:
            logging.error(f"Failed to load segmentation file for job {job_id}: {str(e)}")
            return jsonify({
                "error": "Invalid segmentation data format",
                "job_id": job_id
            }), 400
        
        # Extract metastasis areas (assuming class 3 represents metastasis)
        met_mask = (segmentation == METASTASIS_CLASS).astype(np.int32)
        
        # Identify individual metastases using connected component analysis
        labeled_mask, num_features = ndimage.label(met_mask)
        
        # If no metastases found
        if num_features == 0:
            return jsonify({
                "job_id": job_id,
                "metastasis_count": 0,
                "metastasis_volumes": [],
                "total_volume": 0.0,
                "message": "No metastases detected"
            })
        
        # Calculate volumes
        volumes = []
        for label in range(1, num_features + 1):
            # Count voxels for this metastasis
            voxel_count = np.sum(labeled_mask == label)
            # Calculate volume in mm³
            volume_mm3 = voxel_count * VOXEL_VOLUME_MM3
            volumes.append(float(volume_mm3))
        
        # Sort volumes from largest to smallest
        volumes.sort(reverse=True)
        
        # Save analysis results
        analysis_result = {
            "job_id": job_id,
            "metastasis_count": num_features,
            "metastasis_volumes": volumes,
            "total_volume": sum(volumes),
            "average_volume": sum(volumes) / len(volumes) if volumes else 0
        }
        
        # Save results to file for future reference
        with open(os.path.join(RESULTS_FOLDER, f"{job_id}_analysis.json"), 'w') as f:
            json.dump(analysis_result, f)
        
        logging.info(f"Analysis completed for job {job_id}: {num_features} metastases found")
        return jsonify(analysis_result)
    
    except Exception as e:
        logging.error(f"Error analyzing segmentation for job {job_id}: {str(e)}")
        return jsonify({
            "error": f"Analysis failed: {str(e)}",
            "job_id": job_id
        }), 500

@app.route('/advanced-analysis/<job_id>', methods=['GET'])
def advanced_analysis(job_id):
    """
    Perform advanced multi-class analysis for all tissue types
    """
    if not job_id:
        return jsonify({"error": "Missing job ID"}), 400
        
    # Validate job_id format
    if '/' in job_id or '\\' in job_id or '..' in job_id:
        return jsonify({"error": "Invalid job ID format"}), 400
    
    # Find prediction and original files
    pred_path = os.path.join(RESULTS_FOLDER, f"{job_id}_prediction.npy")
    orig_path = os.path.join(RESULTS_FOLDER, f"{job_id}_original.npy")
    
    if not os.path.exists(pred_path):
        return jsonify({"error": "Segmentation not found"}), 404
    
    try:
        # Load the segmentation mask
        segmentation = load_volume_data(pred_path)
        
        # Load original image if available
        original_image = None
        if os.path.exists(orig_path):
            try:
                original_image = load_volume_data(orig_path)
            except Exception as e:
                logging.warning(f"Could not load original image: {str(e)}")
        
        # Perform comprehensive analysis
        analysis_result = generate_multi_class_analysis(
            segmentation, 
            original_image, 
            VOXEL_VOLUME_MM3
        )
        
        # Add job ID to the result
        analysis_result["job_id"] = job_id
        
        # Save results to file
        with open(os.path.join(RESULTS_FOLDER, f"{job_id}_advanced_analysis.json"), 'w') as f:
            json.dump(analysis_result, f)
        
        logging.info(f"Advanced analysis completed for job {job_id}")
        return jsonify(analysis_result)
        
    except Exception as e:
        logging.error(f"Error performing advanced analysis: {str(e)}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route('/visualization/<job_id>', methods=['GET'])
def get_visualization(job_id):
    """
    Generate and return visualization images for segmentation results
    
    Query parameters:
    - type: Type of visualization (slice, projection, multi-slice)
    - class_id: Optional class ID to highlight
    - slice_idx: Optional slice index for slice visualizations
    - quality: Quality level (standard, high)
    - upscale: Upscaling factor for high-res visualizations
    - enhance_contrast: Whether to enhance contrast
    - enhance_edges: Whether to enhance edges
    """
    if not job_id:
        return jsonify({"error": "Missing job ID"}), 400
        
    # Validate job_id format
    if '/' in job_id or '\\' in job_id or '..' in job_id:
        return jsonify({"error": "Invalid job ID format"}), 400
    
    # Get visualization parameters
    viz_type = request.args.get('type', 'slice')
    
    try:
        class_id = int(request.args.get('class_id')) if request.args.get('class_id') else None
    except ValueError:
        return jsonify({"error": "Invalid class_id parameter"}), 400
        
    try:
        slice_idx = int(request.args.get('slice_idx')) if request.args.get('slice_idx') else None
        logging.info(f"Requested slice_idx={slice_idx} for job {job_id}")
    except ValueError:
        logging.error(f"Invalid slice_idx parameter: {request.args.get('slice_idx')}")
        return jsonify({"error": "Invalid slice_idx parameter"}), 400
        
    # Find prediction and original files
    pred_path = os.path.join(RESULTS_FOLDER, f"{job_id}_prediction.npy")
    orig_path = os.path.join(RESULTS_FOLDER, f"{job_id}_original.npy")
    
    logging.info(f"Looking for segmentation at {pred_path} for visualization type={viz_type}")
    
    if not os.path.exists(pred_path):
        logging.error(f"Segmentation not found at {pred_path}")
        return jsonify({"error": "Segmentation not found"}), 404
    
    try:
        # Load the segmentation mask
        logging.info(f"Loading segmentation from {pred_path} for visualization")
        segmentation = load_volume_data(pred_path)
        logging.info(f"Segmentation loaded successfully. Shape: {segmentation.shape}, Unique values: {np.unique(segmentation)}")
        
        # Load original image if available
        original_image = None
        if os.path.exists(orig_path):
            try:
                logging.info(f"Loading original image from {orig_path}")
                original_image = load_volume_data(orig_path)
                logging.info(f"Original image loaded successfully. Shape: {original_image.shape}")
            except Exception as e:
                logging.warning(f"Could not load original image: {str(e)}")
        else:
            logging.warning(f"Original image file not found: {orig_path}")
        
        # Generate visualization based on type
        logging.info(f"Generating visualization of type: {viz_type}, slice_idx: {slice_idx}")
        
        # Make sure we have a valid slice index (either provided or middle slice)
        if slice_idx is None:
            # If no slice index provided, use the middle slice
            slice_idx = segmentation.shape[0] // 2
            logging.info(f"No slice index provided, using middle slice: {slice_idx}")
        else:
            # Validate slice_idx is within bounds and handle out-of-bounds values
            if slice_idx < 0:
                logging.warning(f"Slice index {slice_idx} was negative, setting to 0")
                slice_idx = 0
            elif slice_idx >= segmentation.shape[0]:
                logging.warning(f"Slice index {slice_idx} was too large (max={segmentation.shape[0]-1}), setting to max")
                slice_idx = segmentation.shape[0] - 1
            else:
                logging.info(f"Using provided slice index: {slice_idx}")
        
        # Generate the appropriate visualization based on type
        if viz_type == 'slice':
            # Additional bounds check for safety
            if slice_idx < 0 or slice_idx >= segmentation.shape[0]:
                logging.error(f"Slice index {slice_idx} is out of range for volume with {segmentation.shape[0]} slices")
                return jsonify({"error": f"Slice index out of range (0-{segmentation.shape[0]-1})"}), 400
                
            # Use high resolution visualization for slice type
            try:
                # Get quality parameter, default to high resolution
                quality = request.args.get('quality', 'high')
                logging.info(f"Using visualization quality: {quality}")
                
                if quality == 'standard':
                    # Use the original lower resolution visualization for backward compatibility
                    logging.info("Using standard visualization method")
                    image = create_colormap_visualization(segmentation, original_image, slice_idx)
                else:
                    # Parse additional enhancement options from request parameters
                    try:
                        upscale_factor = float(request.args.get('upscale', '1.0'))
                        # Limit upscale factor to reasonable values
                        upscale_factor = min(max(upscale_factor, 1.0), 2.0)
                    except (ValueError, TypeError):
                        upscale_factor = 1.0
                        
                    # Parse boolean options
                    contrast_enhancement = request.args.get('enhance_contrast', 'true').lower() == 'true'
                    edge_enhancement = request.args.get('enhance_edges', 'true').lower() == 'true'
                    
                    logging.info(f"Using high-res visualization with upscale={upscale_factor}, contrast={contrast_enhancement}, edges={edge_enhancement}")
                    
                    # Use the new high resolution visualization with enhanced parameters
                    image = create_high_res_visualization(
                        segmentation, 
                        original_image, 
                        slice_idx,
                        tissue_colors=TISSUE_COLORS,
                        tissue_names=TISSUE_NAMES,
                        upscale_factor=upscale_factor,
                        contrast_enhancement=contrast_enhancement,
                        edge_enhancement=edge_enhancement
                    )
                    logging.info("High-res visualization completed successfully")
            except Exception as e:
                logging.error(f"High-res visualization failed, falling back to standard: {str(e)}")
                # Fall back to the original visualization on error
                image = create_colormap_visualization(segmentation, original_image, slice_idx)
        
        elif viz_type == 'projection':
            image = generate_3d_projection(segmentation, class_id)
            
        elif viz_type == 'multi-slice':
            num_slices = int(request.args.get('num_slices', 5))
            
            try:
                # Get quality parameter, default to high resolution
                quality = request.args.get('quality', 'high')
                
                if quality == 'standard':
                    # Use the original lower resolution visualization for backward compatibility
                    image = generate_multi_slice_view(segmentation, original_image, num_slices)
                else:
                    # Parse additional enhancement options
                    contrast_enhancement = request.args.get('enhance_contrast', 'true').lower() == 'true'
                    edge_enhancement = request.args.get('enhance_edges', 'true').lower() == 'true'
                    
                    # Use high-resolution multi-slice view
                    image = generate_high_res_multi_slice_view(
                        segmentation, 
                        original_image, 
                        num_slices,
                        contrast_enhancement=contrast_enhancement,
                        edge_enhancement=edge_enhancement,
                        tissue_colors=TISSUE_COLORS,
                        tissue_names=TISSUE_NAMES
                    )
            except Exception as e:
                logging.error(f"High-res multi-slice visualization failed, falling back to standard: {str(e)}")
                image = generate_multi_slice_view(segmentation, original_image, num_slices)
            
        elif viz_type == 'lesions':
            image = generate_lesion_visualization(segmentation, original_image, slice_idx)
            
        else:
            return jsonify({"error": f"Unknown visualization type: {viz_type}"}), 400
        
        # Convert to PNG and send
        img_io = io.BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)
        
        return send_file(
            img_io, 
            mimetype='image/png',
            as_attachment=False,
            download_name=f"{job_id}_{viz_type}.png"
        )
        
    except Exception as e:
        logging.error(f"Error generating visualization: {str(e)}")
        return jsonify({"error": f"Visualization failed: {str(e)}"}), 500

@app.route('/lesion-analysis/<job_id>', methods=['GET'])
def analyze_lesions(job_id):
    """
    Analyze individual lesions for each tissue class
    """
    if not job_id:
        return jsonify({"error": "Missing job ID"}), 400
        
    # Validate job_id format
    if '/' in job_id or '\\' in job_id or '..' in job_id:
        return jsonify({"error": "Invalid job ID format"}), 400
    
    # Check if specific class is requested
    try:
        class_id = int(request.args.get('class_id')) if request.args.get('class_id') else None
    except ValueError:
        return jsonify({"error": "Invalid class_id parameter"}), 400
    
    # Find prediction file
    pred_path = os.path.join(RESULTS_FOLDER, f"{job_id}_prediction.npy")
    
    if not os.path.exists(pred_path):
        return jsonify({"error": "Segmentation not found"}), 404
    
    try:
        # Load the segmentation mask
        segmentation = load_volume_data(pred_path)
        
        results = {}
        
        # If specific class requested, analyze only that class
        if class_id is not None:
            _, analysis = analyze_connected_components(segmentation, class_id, VOXEL_VOLUME_MM3)
            class_name = TISSUE_NAMES.get(class_id, f"Class {class_id}")
            results[class_name] = analysis
        else:
            # Otherwise analyze all classes
            classes = np.unique(segmentation)
            classes = classes[classes > 0]  # Skip background
            
            for cls in classes:
                _, analysis = analyze_connected_components(segmentation, cls, VOXEL_VOLUME_MM3)
                class_name = TISSUE_NAMES.get(int(cls), f"Class {cls}")
                results[class_name] = analysis
        
        # Add job ID to the result
        results["job_id"] = job_id
        
        # Save results to file
        with open(os.path.join(RESULTS_FOLDER, f"{job_id}_lesion_analysis.json"), 'w') as f:
            json.dump(results, f)
        
        return jsonify(results)
        
    except Exception as e:
        logging.error(f"Error analyzing lesions: {str(e)}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route('/slice-summary/<job_id>', methods=['GET'])
def get_slice_summary(job_id):
    """
    Generate a summary of class distribution across slices
    """
    if not job_id:
        return jsonify({"error": "Missing job ID"}), 400
        
    # Validate job_id format
    if '/' in job_id or '\\' in job_id or '..' in job_id:
        return jsonify({"error": "Invalid job ID format"}), 400
    
    # Find prediction file
    pred_path = os.path.join(RESULTS_FOLDER, f"{job_id}_prediction.npy")
    
    if not os.path.exists(pred_path):
        return jsonify({"error": "Segmentation not found"}), 404
    
    try:
        # Load the segmentation mask
        segmentation = load_volume_data(pred_path)
        
        # Analyze distribution across slices
        depth = segmentation.shape[0]
        slice_data = []
        
        for z in range(depth):
            slice_mask = segmentation[z]
            classes, counts = np.unique(slice_mask, return_counts=True)
            
            # Create dictionary of class counts for this slice
            class_counts = {}
            for cls, count in zip(classes, counts):
                if cls > 0:  # Skip background
                    class_name = TISSUE_NAMES.get(int(cls), f"Class {cls}")
                    class_counts[class_name] = int(count)
            
            # Only include slices with some segmentation
            if class_counts:
                slice_data.append({
                    "slice_idx": z,
                    "classes": class_counts,
                    "total_segmented_voxels": sum(class_counts.values())
                })
        
        # Find slices with the most segmentation for each class
        class_to_best_slice = {}
        all_classes = set()
        
        for slice_info in slice_data:
            for class_name, count in slice_info["classes"].items():
                all_classes.add(class_name)
                if class_name not in class_to_best_slice or count > class_to_best_slice[class_name]["count"]:
                    class_to_best_slice[class_name] = {
                        "slice_idx": slice_info["slice_idx"],
                        "count": count
                    }
        
        # Create summary statistics
        summary = {
            "job_id": job_id,
            "total_slices": depth,
            "slices_with_segmentation": len(slice_data),
            "best_slices_per_class": class_to_best_slice,
            "slice_data": slice_data
        }
        
        # Visualize distribution as graph (optional)
        if request.args.get('with_graph', 'false').lower() == 'true':
            plt.figure(figsize=(12, 6))
            
            # Collect data for plotting
            slice_indices = [data["slice_idx"] for data in slice_data]
            total_counts = [data["total_segmented_voxels"] for data in slice_data]
            
            plt.bar(slice_indices, total_counts)
            plt.xlabel('Slice Index')
            plt.ylabel('Segmentation Voxel Count')
            plt.title('Segmentation Distribution Across Slices')
            plt.grid(True, alpha=0.3)
            
            # Convert to PNG
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close()
            buf.seek(0)
            
            # Convert to base64 for embedding in JSON
            import base64
            image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            summary["distribution_graph"] = image_base64
        
        # Save summary to file
        with open(os.path.join(RESULTS_FOLDER, f"{job_id}_slice_summary.json"), 'w') as f:
            json.dump(summary, f)
        
        return jsonify(summary)
        
    except Exception as e:
        logging.error(f"Error generating slice summary: {str(e)}")
        return jsonify({"error": f"Summary generation failed: {str(e)}"}), 500

@app.route('/metadata', methods=['POST'])
def set_metadata():
    """
    Set metadata for analysis calculation (like voxel size)
    """
    try:
        data = request.json
        if not data:
            return jsonify({
                "error": "No data provided"
            }), 400
            
        global VOXEL_VOLUME_MM3, METASTASIS_CLASS
        
        if 'voxel_volume_mm3' in data:
            try:
                voxel_volume = float(data['voxel_volume_mm3'])
                if voxel_volume <= 0:
                    return jsonify({
                        "error": "Voxel volume must be positive"
                    }), 400
                VOXEL_VOLUME_MM3 = voxel_volume
            except ValueError:
                return jsonify({
                    "error": "Invalid voxel volume value"
                }), 400
            
        if 'metastasis_class' in data:
            try:
                metastasis_class = int(data['metastasis_class'])
                if metastasis_class < 0:
                    return jsonify({
                        "error": "Metastasis class must be non-negative"
                    }), 400
                METASTASIS_CLASS = metastasis_class
            except ValueError:
                return jsonify({
                    "error": "Invalid metastasis class value"
                }), 400
        
        # Update tissue class mappings
        if 'tissue_names' in data and isinstance(data['tissue_names'], dict):
            for key, value in data['tissue_names'].items():
                try:
                    TISSUE_NAMES[int(key)] = str(value)
                except (ValueError, TypeError):
                    pass
                    
        if 'tissue_colors' in data and isinstance(data['tissue_colors'], dict):
            for key, value in data['tissue_colors'].items():
                try:
                    if isinstance(value, list) and len(value) >= 3:
                        TISSUE_COLORS[int(key)] = tuple(value[:3])
                except (ValueError, TypeError):
                    pass
            
        return jsonify({
            "message": "Metadata updated successfully",
            "voxel_volume_mm3": VOXEL_VOLUME_MM3,
            "metastasis_class": METASTASIS_CLASS,
            "tissue_names": TISSUE_NAMES,
            "tissue_colors": {k: list(v) for k, v in TISSUE_COLORS.items()}
        })
    
    except Exception as e:
        logging.error(f"Error updating metadata: {str(e)}")
        return jsonify({
            "error": f"Failed to update metadata: {str(e)}"
        }), 500

@app.route('/volume-info/<job_id>', methods=['GET'])
def get_volume_info(job_id):
    """
    Get information about the volume dimensions for a specific job
    """
    if not job_id:
        return jsonify({"error": "Missing job ID"}), 400
        
    # Validate job_id format to prevent path traversal
    if '/' in job_id or '\\' in job_id or '..' in job_id:
        return jsonify({"error": "Invalid job ID format"}), 400
    
    # Find the prediction file
    pred_path = os.path.join(RESULTS_FOLDER, f"{job_id}_prediction.npy")
    orig_path = os.path.join(RESULTS_FOLDER, f"{job_id}_original.npy")
    
    logging.info(f"Fetching volume info for job {job_id}")
    logging.info(f"Looking for prediction file at {pred_path}")
    
    try:
        # Try to load the prediction file
        if os.path.exists(pred_path):
            logging.info(f"Prediction file found for job {job_id}")
            data = np.load(pred_path)
            dimensions = data.shape
            logging.info(f"Volume dimensions for job {job_id}: {dimensions}")
            
            # Get voxel spacing if available
            voxel_spacing = [1.0, 1.0, 1.0]  # Default if not available
            
            # Get original image dimensions if available
            orig_dimensions = None
            if os.path.exists(orig_path):
                try:
                    logging.info(f"Original image found for job {job_id}")
                    orig_data = np.load(orig_path)
                    orig_dimensions = orig_data.shape
                    logging.info(f"Original image dimensions: {orig_dimensions}")
                except Exception as e:
                    logging.warning(f"Failed to load original image: {str(e)}")
            else:
                logging.info(f"No original image found for job {job_id}")
            
            # Return volume information
            result = {
                "job_id": job_id,
                "dimensions": dimensions,
                "max_slice_index": dimensions[0] - 1 if len(dimensions) > 0 else 0,
                "voxel_spacing": voxel_spacing,
                "voxel_volume_mm3": VOXEL_VOLUME_MM3,
                "original_dimensions": orig_dimensions
            }
            logging.info(f"Returning volume info: {result}")
            return jsonify(result)
        else:
            logging.error(f"Prediction file not found for job {job_id} at {pred_path}")
            return jsonify({"error": "Prediction file not found"}), 404
    except Exception as e:
        logging.error(f"Error getting volume info for job {job_id}: {str(e)}")
        return jsonify({"error": f"Failed to get volume info: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
