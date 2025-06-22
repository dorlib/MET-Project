#!/usr/bin/env python3
# simple_model_service.py - Simplified version of model service for testing

import os
import numpy as np
import json
import logging
import time
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
import uuid
import threading
import gc
from collections import defaultdict

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Configuration
UPLOAD_FOLDER = '/app/uploads'
RESULTS_FOLDER = '/app/results'
MODEL_PATH = os.environ.get('MODEL_PATH', '/app/models/brats_t1ce.pth')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Job status tracker with automatic cleanup (48 hour TTL)
job_status = defaultdict(lambda: {"status": "not_found"})
job_timestamps = {}
JOB_TTL = 48 * 60 * 60  # 48 hours in seconds

def generate_mock_segmentation(input_volume):
    """Generate a mock segmentation for testing"""
    # Get the volume shape
    logging.info(f"Generating mock segmentation for volume of shape {input_volume.shape}")
    
    # Create a simple mock segmentation
    segmentation = np.zeros_like(input_volume, dtype=np.int64)
    
    # Add some "metastasis" regions in the center
    depth, height, width = input_volume.shape
    center_d, center_h, center_w = depth // 2, height // 2, width // 2
    
    # Create a central metastasis
    d_radius, h_radius, w_radius = depth // 10, height // 10, width // 10
    for d in range(center_d - d_radius, center_d + d_radius):
        for h in range(center_h - h_radius, center_h + h_radius):
            for w in range(center_w - w_radius, center_w + w_radius):
                if 0 <= d < depth and 0 <= h < height and 0 <= w < width:
                    # Simple sphere equation
                    if ((d - center_d) / d_radius) ** 2 + \
                       ((h - center_h) / h_radius) ** 2 + \
                       ((w - center_w) / w_radius) ** 2 <= 1.0:
                        segmentation[d, h, w] = 1
    
    # Add some random smaller metastases
    for i in range(3):
        # Random location, biased towards center
        d_offset = int(np.random.normal(0, depth // 8))
        h_offset = int(np.random.normal(0, height // 8))
        w_offset = int(np.random.normal(0, width // 8))
        
        # Center of this metastasis
        d_center = center_d + d_offset
        h_center = center_h + h_offset
        w_center = center_w + w_offset
        
        # Size of this metastasis - smaller than main one
        size_factor = np.random.uniform(0.2, 0.6)
        d_size = int(d_radius * size_factor)
        h_size = int(h_radius * size_factor)
        w_size = int(w_radius * size_factor)
        
        for d in range(d_center - d_size, d_center + d_size):
            for h in range(h_center - h_size, h_center + h_size):
                for w in range(w_center - w_size, w_center + w_size):
                    if 0 <= d < depth and 0 <= h < height and 0 <= w < width:
                        # Simple sphere equation
                        if ((d - d_center) / d_size) ** 2 + \
                           ((h - h_center) / h_size) ** 2 + \
                           ((w - w_center) / w_size) ** 2 <= 1.0:
                            segmentation[d, h, w] = 1
    
    # Add some "edema" around metastases (class 2)
    mask_with_margin = np.zeros_like(segmentation)
    
    # Dilate the metastases to create margin
    for d in range(1, depth - 1):
        for h in range(1, height - 1):
            for w in range(1, width - 1):
                if segmentation[d, h, w] == 1:
                    # Add margin around each metastasis voxel
                    for dd in [-1, 0, 1]:
                        for hh in [-1, 0, 1]:
                            for ww in [-1, 0, 1]:
                                if 0 <= d+dd < depth and 0 <= h+hh < height and 0 <= w+ww < width:
                                    mask_with_margin[d+dd, h+hh, w+ww] = 1
    
    # Set voxels in the margin (but not in original mask) to edema (class 2)
    edema_mask = (mask_with_margin == 1) & (segmentation == 0)
    segmentation[edema_mask] = 2
    
    return segmentation

def generate_visualization(orig_vol, pred_mask, job_id):
    """Generate visualizations of the segmentation"""
    try:
        # Lower resolution for large volumes
        if max(orig_vol.shape) > 256:
            dpi = 80
        else:
            dpi = 100
        
        # Prepare mid-slices
        D, H, W = orig_vol.shape
        mids = {'axial': D//2, 'coronal': H//2, 'sagittal': W//2}
        
        # Create figure with 3 rows (axial, coronal, sagittal) and 2 columns (original, segmentation)
        fig, axs = plt.subplots(3, 2, figsize=(10, 12), dpi=dpi)
        plt.subplots_adjust(hspace=0.3)
        
        # Plot the slices
        for r, axis in enumerate(['axial', 'coronal', 'sagittal']):
            if axis == 'axial':
                img_slice = orig_vol[mids[axis], :, :]
                mask_slice = pred_mask[mids[axis], :, :]
            elif axis == 'coronal':
                img_slice = orig_vol[:, mids[axis], :]
                mask_slice = pred_mask[:, mids[axis], :]
            else:  # sagittal
                img_slice = orig_vol[:, :, mids[axis]]
                mask_slice = pred_mask[:, :, mids[axis]]
                
            # Plot original slice
            axs[r, 0].imshow(img_slice, interpolation='nearest', cmap='gray')
            axs[r, 0].set_title(f"Original {axis}")
            axs[r, 0].axis('off')
            
            # Plot segmentation slice
            axs[r, 1].imshow(mask_slice, interpolation='nearest', cmap='viridis')
            axs[r, 1].set_title(f"Segmentation {axis}")
            axs[r, 1].axis('off')
        
        # Save the figure
        plt.tight_layout()
        vis_path = os.path.join(RESULTS_FOLDER, f"{job_id}_visualization.png")
        fig.savefig(vis_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        # Clean up
        gc.collect()
        
        return vis_path
        
    except Exception as e:
        logging.error(f"Error generating visualization: {str(e)}")
        # Create a simple fallback visualization
        try:
            fig, ax = plt.subplots(figsize=(6, 4), dpi=80)
            ax.text(0.5, 0.5, f"Error generating visualization: {str(e)}", 
                   ha='center', va='center', fontsize=10)
            ax.axis('off')
            
            vis_path = os.path.join(RESULTS_FOLDER, f"{job_id}_visualization.png")
            fig.savefig(vis_path, dpi=80)
            plt.close(fig)
            return vis_path
        except Exception as e2:
            logging.error(f"Error creating fallback visualization: {str(e2)}")
            return None

# Clean up old jobs
def cleanup_old_jobs():
    current_time = time.time()
    expired_jobs = []
    
    for job_id, timestamp in job_timestamps.items():
        if current_time - timestamp > JOB_TTL:
            expired_jobs.append(job_id)
    
    for job_id in expired_jobs:
        try:
            # Remove job files
            for suffix in ['_prediction.npy', '_original.npy', '_visualization.png']:
                file_path = os.path.join(RESULTS_FOLDER, f"{job_id}{suffix}")
                if os.path.exists(file_path):
                    os.remove(file_path)
                    
            # Remove from tracking dictionaries
            del job_status[job_id]
            del job_timestamps[job_id]
            logging.info(f"Cleaned up expired job {job_id}")
        except Exception as e:
            logging.error(f"Error cleaning up job {job_id}: {str(e)}")

# Process job in background
def process_job(file_path, job_id):
    # Record job start time
    job_timestamps[job_id] = time.time()
    
    try:
        job_status[job_id] = {"status": "processing"}
        
        # Check if file exists
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            job_status[job_id] = {"status": "failed", "error": "File not found"}
            return
            
        # Get file size and log it
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        logging.info(f"Processing job {job_id}, file size: {file_size:.2f} MB")
        
        # Load the input volume
        try:
            orig_vol = np.load(file_path)
            
            # Handle 4D data if needed
            if orig_vol.ndim == 4 and orig_vol.shape[3] > 1:
                orig_vol = orig_vol[..., 0]  # Use first channel
                
            if orig_vol.ndim != 3:
                logging.warning(f"Input volume has unexpected shape {orig_vol.shape}, reshaping")
                if orig_vol.ndim == 2:
                    # Add a dimension for 2D inputs
                    orig_vol = orig_vol[np.newaxis, :, :]
                elif orig_vol.ndim > 3:
                    # Take first 3 dims for higher-dimensional inputs
                    orig_vol = orig_vol[:, :, :, 0]
        except Exception as e:
            logging.error(f"Error loading input file: {str(e)}")
            job_status[job_id] = {"status": "failed", "error": f"Error loading input file: {str(e)}"}
            return
        
        # Generate mock segmentation
        pred_mask = generate_mock_segmentation(orig_vol)
        
        # Save prediction to file
        pred_path = os.path.join(RESULTS_FOLDER, f"{job_id}_prediction.npy")
        np.save(pred_path, pred_mask)
        
        # Save original volume for better visualization
        orig_path = os.path.join(RESULTS_FOLDER, f"{job_id}_original.npy")
        np.save(orig_path, orig_vol)
        
        # Generate visualization
        vis_path = generate_visualization(orig_vol, pred_mask, job_id)
        
        # Update job status
        job_status[job_id] = {
            "status": "completed",
            "prediction_path": pred_path,
            "original_path": orig_path,
            "visualization_path": vis_path
        }
        
        logging.info(f"Job {job_id} completed successfully")
        
        # Clean up memory
        del orig_vol, pred_mask
        gc.collect()
        
        # Run cleanup of old jobs
        threading.Thread(target=cleanup_old_jobs, daemon=True).start()
        
    except Exception as e:
        logging.error(f"Error processing job {job_id}: {str(e)}")
        job_status[job_id] = {"status": "failed", "error": str(e)}
        
        # Ensure memory is cleaned up
        gc.collect()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "service": "simple-model-service", 
        "version": "1.0.0",
        "active_jobs": len(job_status)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict endpoint - starts a background job"""
    data = request.json
    file_path = data.get('file_path')
    job_id = data.get('job_id', str(uuid.uuid4()))
    
    # Check if we're already processing this job
    existing_status = job_status.get(job_id, {}).get('status')
    if existing_status == 'processing':
        return jsonify({
            "message": "Job is already being processed",
            "job_id": job_id,
            "status": "processing"
        })
    elif existing_status == 'completed':
        return jsonify({
            "message": "Job is already completed",
            "job_id": job_id,
            "status": "completed"
        })
    
    # Validate file
    if not file_path:
        return jsonify({"error": "File path not provided"}), 400
    
    if not os.path.exists(file_path):
        return jsonify({"error": f"File not found at path: {file_path}"}), 400
    
    # Start processing in a background thread
    job_thread = threading.Thread(
        target=process_job, 
        args=(file_path, job_id),
        daemon=True
    )
    job_thread.start()
    
    return jsonify({
        "message": "Processing started",
        "job_id": job_id,
        "status": "processing"
    })

@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """Status endpoint - returns job status"""
    status_data = job_status[job_id].copy()
    
    # Add timestamp info if available
    if job_id in job_timestamps:
        elapsed_time = time.time() - job_timestamps[job_id]
        status_data['elapsed_seconds'] = round(elapsed_time, 1)
        status_data['created_at'] = time.strftime(
            '%Y-%m-%d %H:%M:%S', 
            time.localtime(job_timestamps[job_id])
        )
    
    return jsonify(status_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
