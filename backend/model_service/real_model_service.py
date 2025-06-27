#!/usr/bin/env python3
# real_model_service.py - Real model service using UNETR model

import os
import numpy as np
import json
import logging
import time
import pickle
import signal
import traceback
import psutil
from flask import Flask, request, jsonify, send_file, send_file
import matplotlib.pyplot as plt
import uuid
import threading
import gc
from collections import defaultdict
import torch
from unetr_adapter import UnetrModelAdapter

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Configuration
UPLOAD_FOLDER = '/app/uploads'
RESULTS_FOLDER = '/app/results'
MODEL_PATH = os.environ.get('MODEL_PATH', '/app/models/brats_t1ce.pth')
STATUS_FILE_PATH = os.path.join(RESULTS_FOLDER, 'job_status.pkl')
JOB_TIMEOUT = int(os.environ.get('JOB_TIMEOUT', 600))  # Default 10 minutes

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Job status tracker with automatic cleanup (48 hour TTL)
job_status = defaultdict(lambda: {"status": "not_found"})
job_timestamps = {}
JOB_TTL = 48 * 60 * 60  # 48 hours in seconds

# Load job status from disk if available
try:
    if os.path.exists(STATUS_FILE_PATH):
        with open(STATUS_FILE_PATH, 'rb') as f:
            saved_data = pickle.load(f)
            job_status.update(saved_data.get('job_status', {}))
            job_timestamps.update(saved_data.get('job_timestamps', {}))
            logging.info(f"Loaded {len(job_status)} job statuses from disk")
except Exception as e:
    logging.error(f"Error loading job status from disk: {e}")

# Save job status to disk periodically and on exit
def save_job_status():
    try:
        with open(STATUS_FILE_PATH, 'wb') as f:
            pickle.dump({
                'job_status': dict(job_status),
                'job_timestamps': job_timestamps
            }, f)
        logging.info(f"Saved {len(job_status)} job statuses to disk")
    except Exception as e:
        logging.error(f"Error saving job status to disk: {e}")

# Register signal handlers to save status on shutdown
def signal_handler(sig, frame):
    logging.info(f"Received signal {sig}, saving job status and exiting")
    save_job_status()
    exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Initialize model adapter
logging.info(f"🔧 Initializing model adapter with path={MODEL_PATH}")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info(f"💻 Using device: {device} (CUDA available: {torch.cuda.is_available()})")

# If no CUDA is available, warn about using CPU which might be slow
if not torch.cuda.is_available():
    logging.warning("⚠️ CUDA not available - using CPU for model inference may be very slow. Timeout mechanism enabled.")
else:
    logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
model_adapter = UnetrModelAdapter(
    model_path=MODEL_PATH, 
    device=device,
    num_classes=4
)
logging.info(f"✅ Model adapter initialized (model loaded: {model_adapter.model is not None})")

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

# Clean up old jobs and save state
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
    
    # Save updated job status to disk
    save_job_status()

# Watch thread to monitor job timeouts
def job_watchdog():
    while True:
        try:
            current_time = time.time()
            for job_id, job_data in list(job_status.items()):
                if job_data.get('status') == 'processing' and job_id in job_timestamps:
                    elapsed = current_time - job_timestamps[job_id]
                    if elapsed > JOB_TIMEOUT:
                        logging.warning(f"Job {job_id} timed out after {elapsed:.1f}s")
                        job_status[job_id] = {
                            "status": "failed", 
                            "error": f"Processing timed out after {elapsed:.1f} seconds"
                        }
                        save_job_status()
        except Exception as e:
            logging.error(f"Error in job watchdog: {e}")
        
        time.sleep(30)  # Check every 30 seconds

# Start the watchdog thread
watchdog_thread = threading.Thread(target=job_watchdog, daemon=True)
watchdog_thread.start()

# Process job in background with better error handling and memory management
def process_job(file_path, job_id):
    # Record job start time
    job_timestamps[job_id] = time.time()
    success = False
    
    logging.info(f"🚀 PROCESS_JOB STARTED: job_id={job_id}, file_path={file_path}")
    
    try:
        job_status[job_id] = {"status": "processing"}
        save_job_status()  # Save status early to survive crashes
        
        # Check if file exists
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            job_status[job_id] = {"status": "failed", "error": "File not found"}
            save_job_status()
            return
            
        # Get file size and log it
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        logging.info(f"Processing job {job_id}, file size: {file_size:.2f} MB")
        
        # Check file format and readability
        try:
            import numpy as np
            logging.info(f"🔍 Attempting to load file: {file_path}")
            temp_data = np.load(file_path, mmap_mode='r')  # Read-only memory mapping
            logging.info(f"📁 File successfully loaded: shape={temp_data.shape}, dtype={temp_data.dtype}")
            del temp_data  # Release the memory mapping
        except Exception as e:
            logging.error(f"⚠️ Error loading input file: {str(e)}")
            job_status[job_id] = {"status": "failed", "error": f"Failed to load input file: {str(e)}"}
            save_job_status()
            return
            
        # Free memory before processing
        logging.info(f"🧹 Clearing memory before model prediction for job {job_id}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info(f"📊 CUDA memory status: {torch.cuda.memory_allocated()/1e6:.2f}MB allocated, {torch.cuda.memory_reserved()/1e6:.2f}MB reserved")
        
        # Log the start of model prediction
        logging.info(f"⏳ STARTING MODEL PREDICTION for job {job_id}, input shape will be loaded from {file_path}")
        
        # Check model adapter status
        if model_adapter.model is None:
            logging.warning(f"⚠️ Model not loaded yet for job {job_id}, attempting to load...")
            try:
                model_adapter.load_model()
                if model_adapter.model is not None:
                    logging.info(f"✅ Model successfully loaded for job {job_id}")
                else:
                    logging.error(f"❌ Failed to load model for job {job_id}")
                    job_status[job_id] = {"status": "failed", "error": "Failed to load model"}
                    save_job_status()
                    return
            except Exception as model_load_error:
                logging.error(f"❌ Error loading model for job {job_id}: {str(model_load_error)}")
                job_status[job_id] = {"status": "failed", "error": f"Model loading error: {str(model_load_error)}"}
                save_job_status()
                return
        
        # Use model adapter to process volume - handle memory errors
        try:
            logging.info(f"🧠 Calling model_adapter.predict() for job {job_id}")
            predict_start_time = time.time()
            results = model_adapter.predict(file_path)
            predict_elapsed = time.time() - predict_start_time
            logging.info(f"⏱️ Prediction completed in {predict_elapsed:.2f} seconds for job {job_id}")
            
            pred_mask = results['prediction']
            orig_vol = results['original_image']
            
            # Log successful prediction completion with shape and metadata information
            logging.info(f"✅ MODEL PREDICTION SUCCESSFUL for job {job_id}: Shape={pred_mask.shape}, "
                        f"Classes={len(np.unique(pred_mask))}, Max Value={np.max(pred_mask)}")
            
            # Quick plot for development - show the middle slices right after prediction
            try:
                logging.info("Generating quick development plot...")
                D, H, W = orig_vol.shape
                mid_z = D // 2
                mid_y = H // 2
                mid_x = W // 2
                
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.imshow(orig_vol[mid_z, :, :], cmap='gray')
                plt.contour(pred_mask[mid_z, :, :] > 0, colors='r')
                plt.title(f'Axial (Z={mid_z})')
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(orig_vol[:, mid_y, :], cmap='gray')
                plt.contour(pred_mask[:, mid_y, :] > 0, colors='g')
                plt.title(f'Coronal (Y={mid_y})')
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(orig_vol[:, :, mid_x], cmap='gray')
                plt.contour(pred_mask[:, :, mid_x] > 0, colors='b')
                plt.title(f'Sagittal (X={mid_x})')
                plt.axis('off')
                
                plt.tight_layout()
                
                # Save in the usual results folder
                dev_plot_path = os.path.join(RESULTS_FOLDER, f"{job_id}_dev_plot.png")
                plt.savefig(dev_plot_path)
                
                # Also save in the dedicated dev_plots folder for easy access
                dev_plots_dir = '/app/dev_plots'
                if os.path.exists(dev_plots_dir):
                    os.makedirs(dev_plots_dir, exist_ok=True)
                    dev_plot_path_external = os.path.join(dev_plots_dir, f"{job_id}_dev_plot.png")
                    plt.savefig(dev_plot_path_external)
                    logging.info(f"Dev plot also saved to {dev_plot_path_external}")
                
                plt.close()
                logging.info(f"Dev plot saved to {dev_plot_path}")
                
                # Print counts of segmentation classes for quick analysis
                classes, counts = np.unique(pred_mask, return_counts=True)
                class_percentages = counts / np.prod(pred_mask.shape) * 100
                for cl, count, pct in zip(classes, counts, class_percentages):
                    logging.info(f"Class {cl}: {count} voxels ({pct:.2f}%)")
            except Exception as e:
                logging.error(f"Dev plot generation failed: {e}")
            
            # Save prediction to file immediately to free memory if needed
            pred_path = os.path.join(RESULTS_FOLDER, f"{job_id}_prediction.npy")
            np.save(pred_path, pred_mask)
            
            # Save original volume for better visualization
            orig_path = os.path.join(RESULTS_FOLDER, f"{job_id}_original.npy")
            np.save(orig_path, orig_vol)
            
            # Update status as soon as main results are saved
            job_status[job_id] = {
                "status": "saving",
                "prediction_path": pred_path,
                "original_path": orig_path
            }
            save_job_status()
            
            # Generate visualization
            vis_path = generate_visualization(orig_vol, pred_mask, job_id)
            
            # Final status update
            job_status[job_id] = {
                "status": "completed",
                "prediction_path": pred_path,
                "original_path": orig_path,
                "visualization_path": vis_path
            }
            
            logging.info(f"🎉 JOB {job_id} COMPLETED SUCCESSFULLY - Prediction, visualization and file saving complete")
            logging.info(f"📊 Results saved to: prediction={pred_path}, visualization={vis_path}")
            success = True
            
        except torch.cuda.OutOfMemoryError as cuda_err:
            logging.error(f"❌ CUDA out of memory for job {job_id}: {str(cuda_err)}")
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    logging.error(f"GPU {i} memory: {torch.cuda.memory_allocated(i)/1e6:.2f}MB allocated, {torch.cuda.memory_reserved(i)/1e6:.2f}MB reserved")
            job_status[job_id] = {"status": "failed", "error": "GPU memory exceeded"}
        except MemoryError as mem_err:
            logging.error(f"❌ Out of system memory for job {job_id}: {str(mem_err)}")
            import psutil
            mem = psutil.virtual_memory()
            logging.error(f"System memory: {mem.percent}% used, {mem.available/1e6:.2f}MB available")
            job_status[job_id] = {"status": "failed", "error": "System memory exceeded"}
        except Exception as pred_err:
            logging.error(f"❌ Prediction failed with unexpected error for job {job_id}: {str(pred_err)}")
            logging.error(f"Error details: {traceback.format_exc()}")
            job_status[job_id] = {"status": "failed", "error": f"Prediction error: {str(pred_err)}"}
        
        # Clean up memory
        if 'orig_vol' in locals(): del orig_vol
        if 'pred_mask' in locals(): del pred_mask
        if 'results' in locals(): del results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Save status
        save_job_status()
        
        # Run cleanup of old jobs
        threading.Thread(target=cleanup_old_jobs, daemon=True).start()
        
    except Exception as e:
        # Get detailed error info
        error_str = f"{str(e)}\n{traceback.format_exc()}"
        logging.error(f"Error processing job {job_id}: {error_str}")
        job_status[job_id] = {"status": "failed", "error": str(e)}
        save_job_status()
        
        # Ensure memory is cleaned up
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    # Check for active model
    model_loaded = False
    try:
        model_loaded = model_adapter.model is not None
        if not model_loaded:
            # Try to reload model
            model_adapter.load_model()
            model_loaded = model_adapter.model is not None
    except:
        model_loaded = False
    
    # Check disk space in results folder
    disk_space = {}
    try:
        if os.path.exists(RESULTS_FOLDER):
            stat = os.statvfs(RESULTS_FOLDER)
            free_space_mb = (stat.f_bavail * stat.f_frsize) / (1024 * 1024)
            disk_space = {
                "free_mb": round(free_space_mb, 1),
                "status": "ok" if free_space_mb > 500 else "low"
            }
    except:
        disk_space = {"status": "unknown"}
    
    return jsonify({
        "status": "healthy", 
        "service": "real-model-service", 
        "version": "1.0.1",
        "active_jobs": len(job_status),
        "model_loaded": model_loaded,
        "disk_space": disk_space,
        "recovery_enabled": True
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict endpoint - starts a background job"""
    logging.info(f"📝 Received prediction request: {request.json}")
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
    try:
        # Warn if using CPU for a large file
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if not torch.cuda.is_available() and file_size_mb > 10:
            logging.warning(f"⚠️ Processing large file ({file_size_mb:.1f} MB) on CPU - may be slow and timeout might occur")
            
        job_thread = threading.Thread(
            target=process_job, 
            args=(file_path, job_id),
            daemon=True
        )
        job_thread.start()
        logging.info(f"🧵 Background thread started for job {job_id}")
        
        # Update status immediately so it can be queried
        job_status[job_id] = {"status": "processing"}
        save_job_status()
        
        response_msg = "Processing started"
        if not torch.cuda.is_available():
            response_msg += " (Note: Running on CPU may take longer or use mock prediction if timeout occurs)"
            
        return jsonify({
            "message": response_msg,
            "job_id": job_id,
            "status": "processing",
            "device": "cpu" if not torch.cuda.is_available() else "gpu"
        })
    except Exception as thread_err:
        logging.error(f"❌ Failed to start processing thread for job {job_id}: {str(thread_err)}")
        return jsonify({
            "error": f"Failed to start processing: {str(thread_err)}",
            "job_id": job_id,
            "status": "failed"
        }), 500

@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """Status endpoint - returns job status"""
    status_data = job_status[job_id].copy()
    
    # Check if job files exist even if status is not found
    if status_data.get('status') == "not_found":
        # Check if prediction file exists
        pred_path = os.path.join(RESULTS_FOLDER, f"{job_id}_prediction.npy")
        if os.path.exists(pred_path):
            # Job files exist but status was lost - recover it
            orig_path = os.path.join(RESULTS_FOLDER, f"{job_id}_original.npy")
            vis_path = os.path.join(RESULTS_FOLDER, f"{job_id}_visualization.png")
            
            if os.path.exists(pred_path):
                # Recover job status
                recovered_status = {
                    "status": "completed",
                    "prediction_path": pred_path,
                    "recovered": True
                }
                
                if os.path.exists(orig_path):
                    recovered_status["original_path"] = orig_path
                if os.path.exists(vis_path):
                    recovered_status["visualization_path"] = vis_path
                
                # Update job status
                job_status[job_id] = recovered_status
                job_timestamps[job_id] = os.path.getmtime(pred_path)
                save_job_status()
                
                logging.info(f"Recovered job status for {job_id}")
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

@app.route('/results/<job_id>', methods=['GET'])
def get_results(job_id):
    """Results endpoint - returns processed results for visualization"""
    # Check if job exists and is completed
    status_data = job_status[job_id]
    
    if status_data.get('status') != 'completed':
        return jsonify({
            "error": f"Job {job_id} is not completed. Current status: {status_data.get('status', 'unknown')}"
        }), 404
    
    try:
        # Get paths from job status
        pred_path = status_data.get('prediction_path')
        orig_path = status_data.get('original_path')
        
        # Ensure files exist
        if not pred_path or not os.path.exists(pred_path):
            return jsonify({"error": f"Prediction file not found for job {job_id}"}), 404
        if not orig_path or not os.path.exists(orig_path):
            return jsonify({"error": f"Original volume file not found for job {job_id}"}), 404
        
        # Load the data
        pred_data = np.load(pred_path)
        orig_data = np.load(orig_path)
        
        # Get basic stats
        pred_stats = {
            "shape": pred_data.shape,
            "dtype": str(pred_data.dtype),
            "min": float(pred_data.min()),
            "max": float(pred_data.max()),
            "unique_values": [int(x) for x in np.unique(pred_data)]
        }
        
        # Calculate some basic metrics
        metrics = {
            "total_voxels": int(np.prod(pred_data.shape)),
            "tumor_voxels": int(np.sum(pred_data > 0)),
            "tumor_percentage": float(np.sum(pred_data > 0) / np.prod(pred_data.shape) * 100)
        }
        
        # Return the results
        return jsonify({
            "job_id": job_id,
            "status": "completed",
            "prediction_stats": pred_stats,
            "metrics": metrics,
            "visualization_url": f"/visualization/{job_id}" if 'visualization_path' in status_data else None
        })
        
    except Exception as e:
        error_str = f"{str(e)}\n{traceback.format_exc()}"
        logging.error(f"Error getting results for job {job_id}: {error_str}")
        return jsonify({
            "error": f"Error retrieving results: {str(e)}"
        }), 500

@app.route('/visualization/<job_id>', methods=['GET'])
def get_visualization(job_id):
    """Visualization endpoint - returns the PNG visualization"""
    # Check if job exists and is completed
    status_data = job_status[job_id]
    
    if status_data.get('status') != 'completed':
        return jsonify({
            "error": f"Job {job_id} is not completed. Current status: {status_data.get('status', 'unknown')}"
        }), 404
    
    try:
        # Get visualization path
        vis_path = status_data.get('visualization_path')
        
        # Ensure file exists
        if not vis_path or not os.path.exists(vis_path):
            return jsonify({"error": f"Visualization not found for job {job_id}"}), 404
        
        # Return the image file
        return send_file(vis_path, mimetype='image/png')
        
    except Exception as e:
        error_str = f"{str(e)}\n{traceback.format_exc()}"
        logging.error(f"Error getting visualization for job {job_id}: {error_str}")
        return jsonify({
            "error": f"Error retrieving visualization: {str(e)}"
        }), 500

@app.route('/prediction-file/<job_id>', methods=['GET'])
def get_prediction_file(job_id):
    """Return the raw prediction file for a job, for use by other services"""
    # Check if job exists and is completed
    status_data = job_status[job_id]
    
    if status_data.get('status') != 'completed':
        return jsonify({
            "error": f"Job {job_id} is not completed. Current status: {status_data.get('status', 'unknown')}"
        }), 404
    
    try:
        # Get prediction path
        pred_path = status_data.get('prediction_path')
        
        # Ensure file exists
        if not pred_path or not os.path.exists(pred_path):
            return jsonify({"error": f"Prediction file not found for job {job_id}"}), 404
        
        # Return the raw file
        return send_file(pred_path)
        
    except Exception as e:
        error_str = f"{str(e)}\n{traceback.format_exc()}"
        logging.error(f"Error getting prediction file for job {job_id}: {error_str}")
        return jsonify({
            "error": f"Error retrieving prediction file: {str(e)}"
        }), 500

# Save job status periodically
@app.before_request
def before_request():
    # Periodically save job status (every ~20 requests)
    if np.random.random() < 0.05:  
        save_job_status()

# Initialize the model when the app starts
@app.before_first_request
def initialize_model():
    logging.info("Loading model...")
    try:
        model_adapter.load_model()
        logging.info("Model loading complete")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        logging.error(traceback.format_exc())
        logging.warning("Will continue and try to load model again when needed")

if __name__ == '__main__':
    # Load the model upfront - but continue even if it fails
    try:
        initialize_model()
    except:
        logging.error("Failed to initialize model at startup")
        logging.error(traceback.format_exc())
    
    # Run the app
    app.run(host='0.0.0.0', port=5001, threaded=True)
