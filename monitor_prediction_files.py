#!/usr/bin/env python3
"""
Monitor for creation and changes to prediction.npy and original.npy files.
This helps confirm if the model service is saving both files correctly.
"""

import time
import os
import sys
import signal
import numpy as np
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

# Global flag for clean exit
running = True

def signal_handler(sig, frame):
    """Handle Ctrl+C to exit cleanly"""
    global running
    logging.info("Stopping file monitor...")
    running = False

# Set up signal handler
signal.signal(signal.SIGINT, signal_handler)

def get_file_info(file_path):
    """Get detailed info about a file, especially .npy files"""
    size = os.path.getsize(file_path)
    modified_time = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
    file_info = {
        "size": size,
        "modified": modified_time
    }
    
    # For numpy files, extract additional information
    if file_path.endswith('.npy'):
        try:
            data = np.load(file_path)
            file_info.update({
                "shape": data.shape,
                "dtype": str(data.dtype),
                "min": float(data.min()),
                "max": float(data.max()),
                "mean": float(data.mean()),
                "unique_values": np.unique(data).tolist() if data.size < 1000000 else len(np.unique(data))
            })
        except Exception as e:
            file_info["error"] = str(e)
            
    return file_info

def find_paired_files(directory):
    """Find prediction.npy and its corresponding original.npy file pairs"""
    prediction_files = {}
    original_files = {}
    
    for filename in os.listdir(directory):
        if '_prediction.npy' in filename:
            job_id = filename.split('_prediction.npy')[0]
            prediction_files[job_id] = os.path.join(directory, filename)
        elif '_original.npy' in filename:
            job_id = filename.split('_original.npy')[0]
            original_files[job_id] = os.path.join(directory, filename)
    
    # Find job IDs with both prediction and original files
    paired_job_ids = set(prediction_files.keys()) & set(original_files.keys())
    unpaired_prediction_job_ids = set(prediction_files.keys()) - paired_job_ids
    unpaired_original_job_ids = set(original_files.keys()) - paired_job_ids
    
    return {
        "paired": {job_id: (prediction_files[job_id], original_files[job_id]) for job_id in paired_job_ids},
        "unpaired_predictions": {job_id: prediction_files[job_id] for job_id in unpaired_prediction_job_ids},
        "unpaired_originals": {job_id: original_files[job_id] for job_id in unpaired_original_job_ids}
    }

def monitor_prediction_files(directory_path, interval=2):
    """Monitor directory for prediction and original files"""
    logging.info(f"Starting monitoring of {directory_path} for prediction and original files")
    
    last_state = {}
    
    global running
    while running:
        try:
            current_files = find_paired_files(directory_path)
            
            # Check for new paired files
            for job_id, (pred_path, orig_path) in current_files["paired"].items():
                if job_id not in last_state.get("paired", {}):
                    logging.info(f"✅ New paired files detected for job ID: {job_id}")
                    pred_info = get_file_info(pred_path)
                    orig_info = get_file_info(orig_path)
                    
                    logging.info(f"  Prediction file: {os.path.basename(pred_path)}")
                    logging.info(f"    Size: {pred_info['size']} bytes")
                    if "shape" in pred_info:
                        logging.info(f"    Shape: {pred_info['shape']}, Type: {pred_info['dtype']}")
                        logging.info(f"    Values: min={pred_info['min']:.4f}, max={pred_info['max']:.4f}, mean={pred_info['mean']:.4f}")
                        logging.info(f"    Unique values: {pred_info['unique_values']}")
                    
                    logging.info(f"  Original file: {os.path.basename(orig_path)}")
                    logging.info(f"    Size: {orig_info['size']} bytes")
                    if "shape" in orig_info:
                        logging.info(f"    Shape: {orig_info['shape']}, Type: {orig_info['dtype']}")
                        logging.info(f"    Values: min={orig_info['min']:.4f}, max={orig_info['max']:.4f}, mean={orig_info['mean']:.4f}")
            
            # Check for new unpaired prediction files
            for job_id, pred_path in current_files["unpaired_predictions"].items():
                if job_id not in last_state.get("unpaired_predictions", {}):
                    logging.warning(f"⚠️ New unpaired prediction file for job ID: {job_id}")
                    pred_info = get_file_info(pred_path)
                    logging.info(f"  Prediction file: {os.path.basename(pred_path)}")
                    logging.info(f"    Size: {pred_info['size']} bytes")
                    if "shape" in pred_info:
                        logging.info(f"    Shape: {pred_info['shape']}, Type: {pred_info['dtype']}")
                        logging.info(f"    Values: min={pred_info['min']:.4f}, max={pred_info['max']:.4f}, mean={pred_info['mean']:.4f}")
                        logging.info(f"    Unique values: {pred_info['unique_values']}")
                    logging.info(f"  Original file: Missing!")
            
            # Check for new unpaired original files
            for job_id, orig_path in current_files["unpaired_originals"].items():
                if job_id not in last_state.get("unpaired_originals", {}):
                    logging.warning(f"⚠️ New unpaired original file for job ID: {job_id}")
                    orig_info = get_file_info(orig_path)
                    logging.info(f"  Prediction file: Missing!")
                    logging.info(f"  Original file: {os.path.basename(orig_path)}")
                    logging.info(f"    Size: {orig_info['size']} bytes")
                    if "shape" in orig_info:
                        logging.info(f"    Shape: {orig_info['shape']}, Type: {orig_info['dtype']}")
                        logging.info(f"    Values: min={orig_info['min']:.4f}, max={orig_info['max']:.4f}, mean={orig_info['mean']:.4f}")
            
            # Store current state for next comparison
            last_state = current_files
            
            # Print summary periodically
            if current_files["paired"]:
                num_paired = len(current_files["paired"])
                num_unpaired_pred = len(current_files["unpaired_predictions"])
                num_unpaired_orig = len(current_files["unpaired_originals"])
                logging.info(f"Summary: {num_paired} paired files, {num_unpaired_pred} unpaired predictions, {num_unpaired_orig} unpaired originals")
            
            # Sleep before next check
            time.sleep(interval)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logging.error(f"Error in monitoring: {e}")
            time.sleep(interval)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        directory_path = sys.argv[1]
    else:
        directory_path = "/app/results"  # Default path in Docker
    
    monitor_prediction_files(directory_path)
