#!/usr/bin/env python3
# model_service/model_service.py - UNETR model inference service for MRI brain metastasis segmentation

import os
import numpy as np
import torch
import json
import logging
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
import uuid
import threading
import time
import gc
from collections import defaultdict

# Configure memory usage before importing other modules
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Limit torch memory usage
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.7)  # Use only 70% of available GPU memory
    
# Determine which model adapter to use
USE_MOCK_MODEL = os.environ.get('MOCK_MODEL', '').lower() in ('true', '1', 'yes')

if USE_MOCK_MODEL:
    logging.info("Using MockModelAdapter for lightweight testing")
    from mock_model_adapter import MockModelAdapter as ModelAdapter
else:
    logging.info("Using UnetrModelAdapter for production inference")
    from unetr_adapter import UnetrModelAdapter as ModelAdapter

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Configuration
BASE_DIR = os.environ.get('BASE_DIR', '/app/data')
MODEL_PATH = os.environ.get('MODEL_PATH', '/app/models/brats_t1ce.pth')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4
UPLOAD_FOLDER = '/app/uploads'
RESULTS_FOLDER = '/app/results'

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Job status tracker with automatic cleanup (48 hour TTL)
job_status = defaultdict(lambda: {"status": "not_found"})
job_timestamps = {}
JOB_TTL = 48 * 60 * 60  # 48 hours in seconds

# Initialize the model adapter
if USE_MOCK_MODEL:
    model_adapter = ModelAdapter()
else:
    model_adapter = ModelAdapter(
        model_path=MODEL_PATH,
        device=DEVICE,
        num_classes=NUM_CLASSES
    )

# Generate visualization of segmentation
def generate_visualization(orig_vol, pred_mask, job_id):
    # Use reduced DPI to save memory
    dpi = 100
    
    try:
        # Prepare mid-slices
        D, H, W = orig_vol.shape
        mids = {'axial': D//2, 'coronal': H//2, 'sagittal': W//2}
        
        # Create a memory-efficient figure by only loading slices we need
        fig, axs = plt.subplots(3, 2, figsize=(10, 12), dpi=dpi)
        plt.subplots_adjust(hspace=0.3)  # Add more space between rows
        
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
            
            # Clear variables to free memory
            del img_slice, mask_slice
        
        # Free more memory
        plt.tight_layout()
        vis_path = os.path.join(RESULTS_FOLDER, f"{job_id}_visualization.png")
        
        # Save with compression to reduce file size
        fig.savefig(vis_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1, optimize=True)
        plt.close(fig)
        
        # Force cleanup
        del fig, axs
        gc.collect()
        
        return vis_path
    except Exception as e:
        logging.error(f"Error generating visualization: {str(e)}")
        # Create a simple fallback image with text
        try:
            fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
            ax.text(0.5, 0.5, f"Error generating visualization:\n{str(e)}", 
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
            
            vis_path = os.path.join(RESULTS_FOLDER, f"{job_id}_visualization.png")
            fig.savefig(vis_path)
            plt.close(fig)
            
            return vis_path
        except:
            logging.error("Failed to create fallback visualization")
            return None

# Clean up old jobs periodically
def cleanup_old_jobs():
    current_time = time.time()
    expired_jobs = []
    
    for job_id, timestamp in job_timestamps.items():
        if current_time - timestamp > JOB_TTL:
            expired_jobs.append(job_id)
    
    for job_id in expired_jobs:
        try:
            # Remove job files if they exist
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
        
        # Check if file exists before processing
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            job_status[job_id] = {"status": "failed", "error": "File not found"}
            return
            
        # Get file size and log it
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        logging.info(f"Processing job {job_id}, file size: {file_size:.2f} MB")
        
        # Run garbage collection before prediction
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Use model adapter for prediction
        results = model_adapter.predict(file_path)
        pred_mask = results['prediction']
        orig_vol = results['original_image']
        
        # Free memory explicitly
        del results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Save prediction to file
        pred_path = os.path.join(RESULTS_FOLDER, f"{job_id}_prediction.npy")
        np.save(pred_path, pred_mask)
        
        # Save original volume for better visualization
        orig_path = os.path.join(RESULTS_FOLDER, f"{job_id}_original.npy")
        np.save(orig_path, orig_vol)
        
        # Generate visualization - use lower resolution for large volumes
        vis_path = generate_visualization(orig_vol, pred_mask, job_id)
        
        # Clean up memory after saving files
        del pred_mask, orig_vol
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Update job status
        job_status[job_id] = {
            "status": "completed",
            "prediction_path": pred_path,
            "original_path": orig_path,
            "visualization_path": vis_path
        }
        
        logging.info(f"Job {job_id} completed successfully")
        
        # Run cleanup of old jobs
        threading.Thread(target=cleanup_old_jobs).start()
        
    except Exception as e:
        logging.error(f"Error processing job {job_id}: {str(e)}")
        job_status[job_id] = {"status": "failed", "error": str(e)}
        
        # Ensure memory is cleaned up even after errors
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@app.route('/health', methods=['GET'])
def health_check():
    health_info = {
        "status": "healthy", 
        "service": "model-service", 
        "device": str(DEVICE),
        "model_type": "mock" if USE_MOCK_MODEL else "unetr",
        "active_jobs": len(job_status),
        "version": "1.0.1"
    }
    
    # Add memory info if available
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)    # MB
            health_info["gpu_memory"] = {
                "allocated_mb": round(allocated, 1),
                "reserved_mb": round(reserved, 1)
            }
        
        # Add CPU memory info
        import psutil
        memory = psutil.virtual_memory()
        health_info["cpu_memory"] = {
            "total_gb": round(memory.total / (1024**3), 1),
            "available_gb": round(memory.available / (1024**3), 1),
            "percent_used": memory.percent
        }
    except Exception as e:
        health_info["memory_info_error"] = str(e)
        
    return jsonify(health_info)

@app.route('/predict', methods=['POST'])
def predict():
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
    
    # Check file size
    try:
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        if file_size > 500:  # Limit file size to 500MB
            return jsonify({
                "error": f"File too large ({file_size:.1f} MB). Maximum allowed size is 500MB."
            }), 413  # Payload Too Large
    except Exception as e:
        logging.error(f"Error checking file size: {str(e)}")
    
    # Start processing in a background thread
    job_thread = threading.Thread(
        target=process_job, 
        args=(file_path, job_id),
        daemon=True  # Make thread a daemon so it doesn't block server shutdown
    )
    job_thread.start()
    
    # Return response immediately
    return jsonify({
        "message": "Processing started",
        "job_id": job_id,
        "status": "processing",
        "model_type": "mock" if USE_MOCK_MODEL else "unetr"
    })

@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    status_data = job_status[job_id].copy()
    
    # Add timestamp info if available
    if job_id in job_timestamps:
        elapsed_time = time.time() - job_timestamps[job_id]
        status_data['elapsed_seconds'] = round(elapsed_time, 1)
        status_data['created_at'] = time.strftime(
            '%Y-%m-%d %H:%M:%S', 
            time.localtime(job_timestamps[job_id])
        )
    
    # Check file sizes if completed
    if status_data.get('status') == 'completed':
        try:
            result_files = {}
            for key in ['prediction_path', 'original_path', 'visualization_path']:
                if key in status_data and os.path.exists(status_data[key]):
                    size_mb = os.path.getsize(status_data[key]) / (1024 * 1024)
                    result_files[os.path.basename(status_data[key])] = f"{size_mb:.2f} MB"
            status_data['result_files'] = result_files
        except Exception as e:
            status_data['file_info_error'] = str(e)
    
    return jsonify(status_data)

if __name__ == '__main__':
    # Load the model when the service starts
    model_adapter.load_model()
    app.run(host='0.0.0.0', port=5001)
