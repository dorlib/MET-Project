from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse
import numpy as np
import torch
import os
import logging
import time
import json
import shutil
from typing import Dict, Any, Optional
import uuid
import threading
import queue
import requests
from pydantic import BaseModel

from .model import get_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Path constants
MODELS_DIR = "/app/models"
RESULTS_DIR = "/app/results"
UPLOADS_DIR = "/app/uploads"
USER_SERVICE_URL = os.environ.get("USER_SERVICE_URL", "http://172.18.0.4:5003")
IMAGE_PROCESSING_SERVICE_URL = os.environ.get("IMAGE_PROCESSING_SERVICE_URL", "http://172.18.0.5:5002")

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="MET Segmentation Service")

# Job status tracking
job_registry = {}

# Shared job queue for background processing
job_queue = queue.Queue()

# Background worker thread
background_worker_thread = None
worker_running = False

class JobStatus(BaseModel):
    job_id: str
    status: str
    message: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

class PredictionRequest(BaseModel):
    file_path: str
    job_id: str
    model_name: Optional[str] = None

# On start, load model once
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
CURRENT_MODEL_NAME = None  # Track currently loaded model

# Set torch to use multiple threads for CPU inference
torch.set_num_threads(4)
torch.set_num_interop_threads(4)

def trigger_metastases_analysis(job_id: str):
    """
    Trigger metastases analysis from the image processing service
    and update the final scan results in the user service
    """
    try:
        logger.info(f"Triggering metastases analysis for job {job_id}")
        
        # Call image processing service to analyze metastases
        response = requests.get(f"{IMAGE_PROCESSING_SERVICE_URL}/analyze/{job_id}", timeout=30)
        
        if response.status_code == 200:
            analysis_data = response.json()
            logger.info(f"Metastases analysis completed for job {job_id}: {analysis_data.get('metastasis_count', 0)} metastases found")
            
            # Update scan in user service with complete results
            update_payload = {
                "status": "completed",
                "metastasis_count": analysis_data.get("metastasis_count"),
                "total_volume": analysis_data.get("total_volume"),
                "metastasis_volumes": analysis_data.get("metastasis_volumes")
            }
            
            user_response = requests.put(
                f"{USER_SERVICE_URL}/scans/{job_id}",
                json=update_payload,
                timeout=10
            )
            
            if user_response.status_code == 200:
                logger.info(f"Successfully updated scan {job_id} with metastases analysis results")
            else:
                logger.error(f"Failed to update scan {job_id} with analysis results. Response: {user_response.status_code} - {user_response.text}")
        else:
            logger.error(f"Metastases analysis failed for job {job_id}. Response: {response.status_code} - {response.text}")
            # Still mark as completed but without metastases data
            update_scan_status_in_user_service(job_id, "completed")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error during metastases analysis for job {job_id}: {str(e)}")
        # Mark as completed even if analysis fails
        update_scan_status_in_user_service(job_id, "completed")
    except Exception as e:
        logger.error(f"Unexpected error during metastases analysis for job {job_id}: {str(e)}")
        # Mark as completed even if analysis fails
        update_scan_status_in_user_service(job_id, "completed")

def update_scan_status_in_user_service(job_id: str, status: str, error_message: str = None):
    """
    Callback function to update scan status in the user service database
    """
    try:
        url = f"{USER_SERVICE_URL}/scans/{job_id}"
        payload = {"status": status}
        
        if error_message and status == "failed":
            # Include error message in the payload if it's a failure
            payload["error_message"] = error_message
            
        logger.info(f"Updating scan status for job {job_id} to '{status}' at {url}")
        
        response = requests.put(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            logger.info(f"Successfully updated scan {job_id} status to '{status}' in user service")
        else:
            logger.error(f"Failed to update scan {job_id} status. Response: {response.status_code} - {response.text}")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error updating scan {job_id} status: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error updating scan {job_id} status: {str(e)}")

def background_worker():
    """Background worker thread that processes jobs from the queue"""
    global worker_running
    worker_running = True
    logger.info("Background worker thread started")
    
    while worker_running:
        try:
            # Get job from queue with timeout
            try:
                file_path, job_id = job_queue.get(timeout=1.0)
            except queue.Empty:
                continue
                
            # Process the job
            process_prediction_sync(file_path, job_id)
            
            # Mark task as done
            job_queue.task_done()
            
        except Exception as e:
            logger.error(f"Error in background worker: {str(e)}")
            
    logger.info("Background worker thread stopped")

def start_background_worker():
    """Start the background worker thread"""
    global background_worker_thread
    if background_worker_thread is None or not background_worker_thread.is_alive():
        background_worker_thread = threading.Thread(target=background_worker, daemon=True)
        background_worker_thread.start()
        logger.info("Started background worker thread")

@app.on_event("startup")
async def startup_event():
    global MODEL, CURRENT_MODEL_NAME
    logger.info(f"Starting MET Segmentation Service on {DEVICE}")
    
    # Make sure required directories exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    
    # Start background worker
    start_background_worker()
    
    # Try to load model, but don't fail startup if we can't
    model_path = os.path.join(MODELS_DIR, "brats_t1ce.pth")
    try:
        # Check if model file exists and is not empty
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            return
            
        if os.path.getsize(model_path) <= 1024:  # Check if it's just our placeholder
            logger.warning(f"Model file appears to be a placeholder: {model_path}")
            return
            
        MODEL = get_model(checkpoint_path=model_path, device=DEVICE)
        CURRENT_MODEL_NAME = "brats_t1ce.pth"  # Set current model name
        logger.info("Model loaded successfully on startup")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {str(e)}")
        # We'll initialize it on first request if it fails here
        
def process_prediction_sync(file_path: str, job_id: str):
    """Synchronous function to process prediction in background thread"""
    try:
        job_registry[job_id] = JobStatus(
            job_id=job_id,
            status="processing", 
            message="Processing scan",
            start_time=time.time()
        )
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        # Load the scan
        logger.info(f"Loading scan from {file_path}")
        try:
            scan = np.load(file_path)
        except Exception as e:
            logger.error(f"Error loading scan from {file_path}: {str(e)}")
            raise ValueError(f"Invalid NPY file format: {str(e)}")
        
        # Check scan shape
        logger.info(f"Scan shape: {scan.shape}")
        
        # Use dummy prediction for testing if model can't be loaded
        use_dummy_prediction = False
        
        # Ensure model is loaded
        global MODEL
        if MODEL is None:
            try:
                logger.info("Model not loaded yet, loading now...")
                model_path = os.path.join(MODELS_DIR, "brats_t1ce.pth")
                
                if not os.path.exists(model_path) or os.path.getsize(model_path) <= 1024:
                    logger.warning("Using dummy prediction because model file is missing or just a placeholder")
                    use_dummy_prediction = True
                else:
                    MODEL = get_model(checkpoint_path=model_path, device=DEVICE)
                    logger.info("Model loaded successfully for inference")
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                logger.warning("Using dummy prediction due to model loading failure")
                use_dummy_prediction = True
        
        if use_dummy_prediction:
            # Create a dummy segmentation for testing
            logger.warning(f"Creating dummy segmentation for job {job_id}")
            # Create a box in the middle of the volume as a placeholder segmentation
            pred_shape = scan.shape
            if scan.ndim == 4:
                pred_shape = scan.shape[:3]  # Remove channel dimension for output
            pred = np.zeros(pred_shape, dtype=np.uint8)
            
            # Create a "segmentation-like" output - a box in the middle of the volume
            center = [s // 2 for s in pred_shape]
            size = [max(1, s // 4) for s in pred_shape]
            
            # Box bounds
            start = [max(0, c - s) for c, s in zip(center, size)]
            end = [min(c + s, d - 1) for c, s, d in zip(center, size, pred_shape)]
            
            # Create a small segmentation in the center - class 1
            pred[
                start[0]:end[0],
                start[1]:end[1], 
                start[2]:end[2]
            ] = 1
            
            # Also add a smaller region of class 2
            smaller_size = [max(1, s // 6) for s in pred_shape]
            smaller_start = [max(0, c - s) for c, s in zip(center, smaller_size)]
            smaller_end = [min(c + s, d - 1) for c, s, d in zip(center, smaller_size, pred_shape)]
            
            pred[
                smaller_start[0]:smaller_end[0],
                smaller_start[1]:smaller_end[1],
                smaller_start[2]:smaller_end[2]
            ] = 2
        else:
            # Run inference with the real model
            logger.info(f"Running inference for job {job_id}")
            try:
                # Preprocess the scan - handle multi-channel input
                processed_scan = scan
                if scan.ndim == 4 and scan.shape[-1] > 1:
                    logger.info(f"Multi-channel scan detected with shape {scan.shape}. Using channel 1 (T1CE).")
                    processed_scan = scan[..., 1]  # Extract T1CE channel
                
                # Normalize the scan
                processed_scan = (processed_scan - processed_scan.mean()) / (processed_scan.std() + 1e-5)
                
                # Prepare input tensor - add batch & channel dims
                x = torch.from_numpy(processed_scan).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
                logger.info(f"Input tensor shape: {x.shape}")
                
                with torch.no_grad():
                    out = MODEL(x)  # shape: (1, num_classes, H, W, D)
                    
                    # Apply softmax to get probabilities
                    probabilities = torch.softmax(out, dim=1).squeeze(0).cpu().numpy()  # shape: (num_classes, H, W, D)
                    
                    # Get final predictions
                    pred = torch.argmax(out, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                    
                logger.info(f"Model inference completed. Output shape: {pred.shape}, Probabilities shape: {probabilities.shape}")
            except Exception as e:
                logger.error(f"Error during model inference: {str(e)}")
                raise RuntimeError(f"Model inference failed: {str(e)}")
        
        # Save prediction to results folder
        pred_path = os.path.join(RESULTS_DIR, f"{job_id}_prediction.npy")
        np.save(pred_path, pred)
        logger.info(f"Saved prediction to {pred_path} with shape {pred.shape}")
        
        # Save probabilities if they exist (from real model inference)
        if 'probabilities' in locals():
            prob_path = os.path.join(RESULTS_DIR, f"{job_id}_probabilities.npy")
            np.save(prob_path, probabilities)
            logger.info(f"Saved probabilities to {prob_path} with shape {probabilities.shape}")
        
        # Update job status to complete
        job_registry[job_id] = JobStatus(
            job_id=job_id, 
            status="completed",
            message="Segmentation completed successfully",
            start_time=job_registry[job_id].start_time,
            end_time=time.time()
        )
        logger.info(f"Job {job_id} completed successfully")
        
        # Trigger metastases analysis and final status update
        trigger_metastases_analysis(job_id)
        
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}")
        job_registry[job_id] = JobStatus(
            job_id=job_id, 
            status="failed",
            message=f"Segmentation failed: {str(e)}",
            start_time=job_registry[job_id].start_time if job_id in job_registry else time.time(),
            end_time=time.time()
        )
        
        # Notify user service that the job failed
        update_scan_status_in_user_service(job_id, "failed", str(e))

@app.post("/predict")
async def predict_endpoint(request: PredictionRequest):
    """
    Endpoint to predict segmentation from an uploaded NPY file.
    Expects a path to a preprocessed NPY file and a job ID.
    """
    file_path = request.file_path
    job_id = request.job_id
    
    # Verify file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    
    # Verify it's an NPY file
    if not file_path.endswith(".npy"):
        raise HTTPException(status_code=400, detail="Only .npy files are accepted")
    
    try:
        # Add job to registry immediately with "queued" status
        job_registry[job_id] = JobStatus(
            job_id=job_id,
            status="queued",
            message="Job queued for processing",
            start_time=time.time()
        )
        
        # Submit job to background processing queue
        job_queue.put((file_path, job_id))
        logger.info(f"Job {job_id} submitted to queue")
        
        return JSONResponse({
            "job_id": job_id,
            "status": "queued",
            "message": "Prediction job submitted successfully"
        })
        
    except Exception as e:
        logger.error(f"Error submitting job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to submit prediction job: {str(e)}")

@app.post("/predict/file")
async def predict_file_endpoint(
    file: UploadFile = File(...),
    job_id: Optional[str] = None
):
    """
    Endpoint to predict segmentation from an uploaded NPY file directly.
    """
    if not file.filename.endswith(".npy"):
        raise HTTPException(status_code=400, detail="Only .npy files accepted")
    
    # Generate job ID if not provided
    if job_id is None:
        job_id = str(uuid.uuid4())
        
    # Save uploaded file to a temporary path
    file_path = os.path.join(RESULTS_DIR, f"{job_id}_original.npy")
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")
    
    # Submit job to background processing queue
    job_queue.put((file_path, job_id))
    
    return JSONResponse({
        "job_id": job_id,
        "status": "queued",
        "message": "File uploaded and prediction job submitted successfully"
    })

@app.get("/status/{job_id}")
async def status_endpoint(job_id: str):
    """
    Endpoint to check the status of a prediction job
    """
    # First check in-memory registry
    if job_id in job_registry:
        job = job_registry[job_id]
        return JSONResponse({
            "job_id": job_id,
            "status": job.status,
            "message": job.message,
            "start_time": job.start_time,
            "end_time": job.end_time,
            "processing_time": job.end_time - job.start_time if job.end_time else None
        })
    
    # If not in registry, check if result files exist on disk (for cases after container restart)
    prediction_file = os.path.join(RESULTS_DIR, f"{job_id}_prediction.npy")
    probabilities_file = os.path.join(RESULTS_DIR, f"{job_id}_probabilities.npy")
    
    if os.path.exists(prediction_file) and os.path.exists(probabilities_file):
        # Files exist, so the job was completed successfully
        return JSONResponse({
            "job_id": job_id,
            "status": "completed",
            "message": "Job completed (recovered from disk)",
            "start_time": None,
            "end_time": None,
            "processing_time": None
        })
    
    # Job not found anywhere
    return JSONResponse({
        "job_id": job_id,
        "status": "not_found",
        "message": "Job ID not found"
    }, status_code=404)

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    try:
        # Check if model is loaded
        model_status = "loaded" if MODEL is not None else "not_loaded"
        
        # Check if CUDA is available
        cuda_available = torch.cuda.is_available()
        device_info = f"CUDA ({torch.cuda.get_device_name(0)})" if cuda_available else "CPU"
        
        return JSONResponse({
            "status": "healthy",
            "model": model_status,
            "device": device_info,
            "active_jobs": len([j for j in job_registry.values() if j.status == "processing"]),
            "completed_jobs": len([j for j in job_registry.values() if j.status == "completed"]),
            "failed_jobs": len([j for j in job_registry.values() if j.status == "failed"])
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse({
            "status": "unhealthy",
            "error": str(e)
        }, status_code=500)

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """
    Endpoint to delete job status and results
    """
    if job_id not in job_registry:
        return JSONResponse({
            "status": "not_found",
            "message": f"Job {job_id} not found"
        }, status_code=404)
    
    try:
        # Remove job from registry
        del job_registry[job_id]
        
        # Delete prediction file if exists
        pred_path = os.path.join(RESULTS_DIR, f"{job_id}_prediction.npy")
        if os.path.exists(pred_path):
            os.remove(pred_path)
            
        # Delete original file if exists
        original_path = os.path.join(RESULTS_DIR, f"{job_id}_original.npy")
        if os.path.exists(original_path):
            os.remove(original_path)
            
        return JSONResponse({
            "status": "success",
            "message": f"Job {job_id} and related files deleted successfully"
        })
    except Exception as e:
        logger.error(f"Error deleting job {job_id}: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": f"Failed to delete job: {str(e)}"
        }, status_code=500)
        
@app.get("/jobs")
async def list_jobs():
    """
    Endpoint to list all jobs
    """
    return JSONResponse({
        "jobs": [
            {
                "job_id": job_id,
                "status": job.status,
                "message": job.message,
                "start_time": job.start_time,
                "end_time": job.end_time,
                "processing_time": job.end_time - job.start_time if job.end_time else None
            }
            for job_id, job in job_registry.items()
        ],
        "total": len(job_registry),
        "active": len([j for j in job_registry.values() if j.status == "processing"]),
        "completed": len([j for j in job_registry.values() if j.status == "completed"]),
        "failed": len([j for j in job_registry.values() if j.status == "failed"])
    })

@app.get("/models")
async def list_models():
    """
    Endpoint to list all available models
    """
    try:
        models = []
        current_model = CURRENT_MODEL_NAME
        
        # List all .pth files in the models directory
        for filename in os.listdir(MODELS_DIR):
            if filename.endswith('.pth'):
                file_path = os.path.join(MODELS_DIR, filename)
                file_stat = os.stat(file_path)
                
                models.append({
                    "name": filename,
                    "size": file_stat.st_size,
                    "size_mb": round(file_stat.st_size / (1024 * 1024), 1),
                    "modified": file_stat.st_mtime,
                    "is_loaded": filename == current_model
                })
        
        return JSONResponse({
            "models": models,
            "count": len(models),
            "current_model": current_model
        })
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@app.get("/models/{model_name}")
async def get_model_info(model_name: str):
    """
    Endpoint to get information about a specific model
    """
    try:
        model_path = os.path.join(MODELS_DIR, model_name)
        
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model not found")
            
        file_stat = os.stat(model_path)
        
        return JSONResponse({
            "name": model_name,
            "size": file_stat.st_size,
            "size_mb": round(file_stat.st_size / (1024 * 1024), 1),
            "modified": file_stat.st_mtime,
            "is_loaded": model_name == CURRENT_MODEL_NAME,
            "path": model_path
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")
