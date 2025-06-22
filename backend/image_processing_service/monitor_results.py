#!/usr/bin/env python3
import time
import os
import sys

def monitor_directory(path):
    """Monitor a directory for new files and print their details."""
    print(f"Monitoring directory: {path}")
    initial_files = set(os.listdir(path))
    print(f"Initial files: {initial_files}")
    
    while True:
        try:
            time.sleep(2)  # Check every 2 seconds
            current_files = set(os.listdir(path))
            new_files = current_files - initial_files
            
            if new_files:
                print(f"\nNew files detected: {new_files}")
                for f in new_files:
                    full_path = os.path.join(path, f)
                    size = os.path.getsize(full_path)
                    print(f"- {f}: {size} bytes")
                    
                    # If it's a numpy file, try to load and print info
                    if f.endswith('.npy'):
                        try:
                            import numpy as np
                            data = np.load(full_path)
                            print(f"  Shape: {data.shape}, dtype: {data.dtype}")
                            print(f"  Values: min={data.min()}, max={data.max()}, unique={np.unique(data)[:10]} (first 10)")
                        except Exception as e:
                            print(f"  Error loading numpy file: {e}")
                
                # Update initial files
                initial_files = current_files
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        directory_path = sys.argv[1]
    else:
        directory_path = "/app/results"  # Default path in Docker
    
    monitor_directory(directory_path)
