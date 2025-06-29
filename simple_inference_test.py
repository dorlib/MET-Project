#!/usr/bin/env python3
"""
Simple inference speed test without visualization
"""
import sys
import os
sys.path.append('.')

import torch
import numpy as np
import time
from self_attention_cv import UNETR

def test_simple_inference():
    """Test raw inference speed without visualization"""
    print("=== Simple Inference Speed Test ===")
    
    # Load data
    scan_path = "./test_image.npy"
    if not os.path.exists(scan_path):
        print(f"❌ Test file not found: {scan_path}")
        return False
        
    scan = np.load(scan_path)
    print(f"Loaded scan with shape: {scan.shape}")
    
    # Extract T1CE channel if multi-channel
    if scan.ndim == 4 and scan.shape[-1] > 1:
        scan = scan[..., 1]
        print(f"Using T1CE channel, shape: {scan.shape}")
    
    # Normalize scan
    scan = (scan - scan.mean()) / (scan.std() + 1e-5)
    
    # Prepare tensor
    device = torch.device('cpu')
    scan_tensor = torch.FloatTensor(scan).unsqueeze(0).unsqueeze(0).to(device)
    print(f"Tensor shape: {scan_tensor.shape}")
    
    # Load model
    print("Loading model...")
    start_load = time.time()
    
    model = UNETR(
        img_shape=(128, 128, 128),
        input_dim=1,
        output_dim=4,
        embed_dim=128,
        patch_size=16,
        num_heads=8,
        ext_layers=[3, 6, 9, 12, 15, 18],
        norm='instance',
        dropout=0.2,
        base_filters=16,
        dim_linear_block=1024
    ).to(device)
    
    state = torch.load("./Data/saved_models/brats_t1ce.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()
    
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Run inference
    print("Running inference...")
    start_inference = time.time()
    
    with torch.no_grad():
        outputs = model(scan_tensor)
    
    prediction = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()
    
    inference_time = time.time() - start_inference
    print(f"Inference completed in {inference_time:.2f} seconds")
    
    print(f"Prediction shape: {prediction.shape}")
    print(f"Unique values in prediction: {np.unique(prediction)}")
    
    total_time = load_time + inference_time
    print(f"Total time: {total_time:.2f} seconds")
    
    return True

if __name__ == "__main__":
    test_simple_inference()
