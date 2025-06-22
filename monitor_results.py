#!/usr/bin/env python3
import time
import os
import sys
import datetime
import argparse
from colorama import init, Fore, Style

# Initialize colorama
try:
    init()
    COLOR_SUPPORT = True
except:
    COLOR_SUPPORT = False

def colored(text, color=None, style=None):
    """Add color to text if colorama is available"""
    if not COLOR_SUPPORT:
        return text
    
    color_code = getattr(Fore, color.upper(), '') if color else ''
    style_code = getattr(Style, style.upper(), '') if style else ''
    return f"{color_code}{style_code}{text}{Style.RESET_ALL}"

def get_time_str():
    """Get formatted current time"""
    return datetime.datetime.now().strftime("%H:%M:%S")

def monitor_directory(path, check_interval=2, watch_for_changes=False):
    """
    Monitor a directory for prediction and original NPY files.
    
    Args:
        path: Directory path to monitor
        check_interval: Time in seconds between checks
        watch_for_changes: If True, continue monitoring for file changes
    """
    print(colored(f"[{get_time_str()}] Monitoring results directory: {path}", "cyan", "bright"))
    
    # Track job directories and their status
    job_status = {}
    known_dirs = set()
    
    try:
        while True:
            # Get all job directories (assuming each subdirectory is a job ID)
            current_dirs = set()
            try:
                for item in os.listdir(path):
                    job_dir = os.path.join(path, item)
                    if os.path.isdir(job_dir):
                        current_dirs.add(item)
            except Exception as e:
                print(colored(f"[{get_time_str()}] Error reading directory {path}: {e}", "red"))
                time.sleep(check_interval)
                continue
                
            # Check for new job directories
            new_dirs = current_dirs - known_dirs
            if new_dirs:
                print(colored(f"\n[{get_time_str()}] New job directories detected: {new_dirs}", "green"))
                
            # Process all job directories
            for job_id in current_dirs:
                job_dir = os.path.join(path, job_id)
                
                # Get the status of prediction and original files
                job_id_name = job_id  # Job ID might be the full path in some cases
                
                # Check both file naming conventions: 
                # - {job_id}_prediction.npy (file in results dir)
                # - prediction.npy (file in job subdir)
                prediction_path = os.path.join(path, f"{job_id}_prediction.npy")
                original_path = os.path.join(path, f"{job_id}_original.npy")
                
                if not os.path.exists(prediction_path):
                    prediction_path = os.path.join(job_dir, 'prediction.npy')
                    
                if not os.path.exists(original_path):
                    original_path = os.path.join(job_dir, 'original.npy')
                
                has_prediction = os.path.exists(prediction_path)
                has_original = os.path.exists(original_path)
                
                # Skip if already checked and files haven't changed
                if job_id in job_status:
                    old_status = job_status[job_id]
                    if (old_status['has_prediction'] == has_prediction and 
                        old_status['has_original'] == has_original and
                        old_status['prediction_size'] == (os.path.getsize(prediction_path) if has_prediction else 0) and
                        old_status['original_size'] == (os.path.getsize(original_path) if has_original else 0)):
                        continue  # No changes to this job
                
                # Update job status
                prediction_size = os.path.getsize(prediction_path) if has_prediction else 0
                original_size = os.path.getsize(original_path) if has_original else 0
                
                job_status[job_id] = {
                    'has_prediction': has_prediction,
                    'has_original': has_original,
                    'prediction_size': prediction_size,
                    'original_size': original_size,
                }
                
                # Print status for new or updated jobs
                if job_id in new_dirs or job_id not in known_dirs:
                    print(colored(f"[{get_time_str()}] Job {job_id} status:", "yellow"))
                    
                    # Prediction file status
                    if has_prediction:
                        print(colored(f"  ✓ Prediction file: {prediction_size/1024:.1f} KB", "green"))
                        try:
                            import numpy as np
                            data = np.load(prediction_path)
                            print(f"    Shape: {data.shape}, dtype: {data.dtype}")
                            print(f"    Values: min={data.min()}, max={data.max()}")
                            unique_vals = np.unique(data)
                            print(f"    Unique values: {unique_vals[:10]}" + 
                                 (f" ... ({len(unique_vals)} total)" if len(unique_vals) > 10 else ""))
                        except Exception as e:
                            print(colored(f"    Error reading numpy data: {e}", "red"))
                    else:
                        print(colored(f"  ✗ Prediction file: Missing", "red"))
                    
                    # Original file status
                    if has_original:
                        print(colored(f"  ✓ Original file: {original_size/1024:.1f} KB", "green"))
                        try:
                            import numpy as np
                            data = np.load(original_path)
                            print(f"    Shape: {data.shape}, dtype: {data.dtype}")
                            print(f"    Values: min={data.min()}, max={data.max()}")
                        except Exception as e:
                            print(colored(f"    Error reading numpy data: {e}", "red"))
                    else:
                        print(colored(f"  ✗ Original file: Missing", "red"))
            
            # Update known directories
            known_dirs = current_dirs
            
            # Exit if not watching for changes
            if not watch_for_changes:
                break
                
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print(colored(f"\n[{get_time_str()}] Monitoring stopped by user", "yellow"))
    except Exception as e:
        print(colored(f"[{get_time_str()}] Error: {e}", "red"))
    
    return job_status

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monitor prediction and original NPY files')
    parser.add_argument('-d', '--directory', default='/app/results', 
                        help='Path to results directory (default: /app/results)')
    parser.add_argument('-i', '--interval', type=int, default=2, 
                        help='Check interval in seconds (default: 2)')
    parser.add_argument('-w', '--watch', action='store_true', 
                        help='Continuously watch for changes (default: False)')
    args = parser.parse_args()
    
    # Use command-line arguments, then sys.argv, then default
    directory_path = args.directory
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        directory_path = sys.argv[1]
    
    monitor_directory(directory_path, args.interval, args.watch)
