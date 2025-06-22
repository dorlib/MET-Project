#!/bin/bash
# Script to modify model service to use a memory-efficient approach

echo "Applying memory-efficient model service patch..."

# Check if model_service.py exists
MODEL_SERVICE_FILE="/Users/dorliber/Library/CloudStorage/OneDrive-IntelCorporation/Desktop/private/MET-project/backend/model_service/model_service.py"

if [ ! -f "$MODEL_SERVICE_FILE" ]; then
    echo "Error: Cannot find model_service.py file"
    exit 1
fi

# Create a backup of the original file
cp "$MODEL_SERVICE_FILE" "${MODEL_SERVICE_FILE}.backup"
echo "Created backup at ${MODEL_SERVICE_FILE}.backup"

# Update the model_service.py file to use our mock model
cat > "$MODEL_SERVICE_FILE" << 'EOF'
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
from collections import defaultdict

# Conditionally import the appropriate model adapter
USE_MOCK_MODEL = os.environ.get('MOCK_MODEL', 'False').lower() in ('true', '1', 't')

if USE_MOCK_MODEL:
    from mock_model_adapter import mock_model as model_adapter
    logging.info("Using mock model adapter for lightweight processing")
else:
    from unetr_adapter import UnetrModelAdapter
    logging.info("Using full UNETR model adapter")

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

# Job status tracker
job_status = defaultdict(lambda: {"status": "not_found"})

# Initialize the model adapter if using full model
if not USE_MOCK_MODEL:
    model_adapter = UnetrModelAdapter(
        model_path=MODEL_PATH,
        device=DEVICE,
        num_classes=NUM_CLASSES
    )

# Generate visualization of segmentation
def generate_visualization(orig_vol, pred_mask, job_id):
    # Prepare mid-slices
    D, H, W = orig_vol.shape
    mids = {'axial': D//2, 'coronal': H//2, 'sagittal': W//2}
    
    slices_vol = {
        'axial': orig_vol[mids['axial'], :, :],
        'coronal': orig_vol[:, mids['coronal'], :],
        'sagittal': orig_vol[:, :, mids['sagittal']]
    }
    
    slices_pred = {
        'axial': pred_mask[mids['axial'], :, :],
        'coronal': pred_mask[:, mids['coronal'], :],
        'sagittal': pred_mask[:, :, mids['sagittal']]
    }
    
    # Plot and save figure
    fig, axs = plt.subplots(3, 2, figsize=(10, 12), dpi=100)  # Lower DPI to reduce memory usage
    for r, axis in enumerate(['axial', 'coronal', 'sagittal']):
        axs[r, 0].imshow(slices_vol[axis], interpolation='nearest', cmap='gray')
        axs[r, 0].set_title(f"Original {axis}")
        axs[r, 0].axis('off')
        
        axs[r, 1].imshow(slices_pred[axis], interpolation='nearest', cmap='viridis')
        axs[r, 1].set_title(f"Segmentation {axis}")
        axs[r, 1].axis('off')
    
    plt.tight_layout()
    vis_path = os.path.join(RESULTS_FOLDER, f"{job_id}_visualization.png")
    fig.savefig(vis_path, dpi=100)  # Lower DPI to reduce memory usage
    plt.close(fig)
    
    return vis_path

# Process job in background with additional memory management
def process_job(file_path, job_id):
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
        
        # Use model adapter for prediction
        results = model_adapter.predict(file_path)
        pred_mask = results['prediction']
        orig_vol = results['original_image']
        
        # Free memory explicitly
        torch.cuda.empty_cache()
        
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
            "visualization_path": vis_path,
            "segmentation_path": f"/visualization/{job_id}",
            "volume_dimensions": orig_vol.shape
        }
        
        logging.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        logging.error(f"Error processing job {job_id}: {str(e)}")
        job_status[job_id] = {"status": "failed", "error": str(e)}
        
        # Try to clean up memory
        import gc
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "service": "model-service", 
        "device": str(DEVICE),
        "mock_model": USE_MOCK_MODEL
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    file_path = data.get('file_path')
    job_id = data.get('job_id', str(uuid.uuid4()))
    
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 400
    
    # Start processing in a background thread
    threading.Thread(target=process_job, args=(file_path, job_id)).start()
    
    return jsonify({
        "message": "Processing started",
        "job_id": job_id,
        "status": "processing",
        "mock_model": USE_MOCK_MODEL
    })

@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    return jsonify(job_status[job_id])

if __name__ == '__main__':
    # Load the model when the service starts
    if not USE_MOCK_MODEL:
        model_adapter.load_model()
    app.run(host='0.0.0.0', port=5001)
EOF

echo "Updated model_service.py with memory-efficient implementation"

# Apply a similar patch to the gunicorn command in the model-verify section to use timeout
DOCKER_COMPOSE_FILE="/Users/dorliber/Library/CloudStorage/OneDrive-IntelCorporation/Desktop/private/MET-project/docker-compose.yml"

if [ -f "$DOCKER_COMPOSE_FILE" ]; then
    # Create a backup
    cp "$DOCKER_COMPOSE_FILE" "${DOCKER_COMPOSE_FILE}.backup"
    echo "Created backup of docker-compose.yml"
    
    # Update the gunicorn command to include additional parameters
    sed -i.bak 's/gunicorn --bind 0.0.0.0:5001 --timeout 300 model_service:app/gunicorn --bind 0.0.0.0:5001 --timeout 600 --workers 1 --threads 1 --max-requests 5 --max-requests-jitter 2 model_service:app/g' "$DOCKER_COMPOSE_FILE"
    
    echo "Updated gunicorn command in docker-compose.yml"
fi

echo "Memory-efficient model service patch applied successfully"
echo "Restart your Docker services to apply these changes:"
echo "   docker-compose down"
echo "   docker-compose up -d"
