#!/usr/bin/env python3
"""
Analyze the checkpoint to determine the exact UNETR parameters used during training
"""

import torch
import numpy as np

def analyze_checkpoint(checkpoint_path):
    """Analyze checkpoint to infer model parameters"""
    print(f"Analyzing checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    state = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"Checkpoint keys: {list(state.keys())[:10]}...")  # Show first 10 keys
    
    # Look for key parameters that tell us about the model structure
    key_params = {}
    
    for key, tensor in state.items():
        if 'to_qvk.weight' in key:
            qvk_shape = tensor.shape
            print(f"QVK weight shape: {qvk_shape} for {key}")
            # to_qvk.weight should be [3 * embed_dim, embed_dim] for self-attention
            key_params['qvk_total_dim'] = qvk_shape[0]  # 3 * embed_dim or 3 * num_heads * head_dim
            key_params['embed_dim'] = qvk_shape[1]
            break
    
    for key, tensor in state.items():
        if 'W_0.weight' in key:
            w0_shape = tensor.shape
            print(f"W_0 weight shape: {w0_shape} for {key}")
            # This should tell us about the head dimension
            key_params['w0_out'] = w0_shape[0]  # embed_dim
            key_params['w0_in'] = w0_shape[1]   # num_heads * head_dim
            break
    
    # Analyze patch embedding
    for key, tensor in state.items():
        if 'patch_embeddings.proj.weight' in key or 'patch_embedding' in key:
            patch_shape = tensor.shape
            print(f"Patch embedding shape: {patch_shape} for {key}")
            # Should be [embed_dim, input_channels, patch_size, patch_size, patch_size]
            if len(patch_shape) == 5:
                key_params['patch_embed_dim'] = patch_shape[0]
                key_params['input_channels'] = patch_shape[1]
                key_params['patch_size'] = patch_shape[2]
            break
    
    # Look for position embeddings
    for key, tensor in state.items():
        if 'pos_embed' in key or 'position' in key:
            pos_shape = tensor.shape
            print(f"Position embedding shape: {pos_shape} for {key}")
            # Should tell us about number of patches
            break
    
    print("\nInferred parameters:")
    if 'embed_dim' in key_params:
        embed_dim = key_params['embed_dim']
        print(f"embed_dim: {embed_dim}")
        
        if 'qvk_total_dim' in key_params and 'w0_in' in key_params:
            qvk_dim = key_params['qvk_total_dim']
            w0_in = key_params['w0_in']
            
            # qvk_dim should be 3 * num_heads * head_dim
            # w0_in should be num_heads * head_dim
            # So qvk_dim / 3 should equal w0_in
            head_total_dim = qvk_dim // 3
            print(f"Total head dimension (num_heads * head_dim): {head_total_dim}")
            print(f"W0 input dimension: {w0_in}")
            
            if head_total_dim == w0_in:
                print("✓ QVK and W0 dimensions are consistent")
                
                # Try to infer num_heads
                # Common head dimensions are 32, 64, 128
                for head_dim in [32, 40, 64, 128]:
                    if head_total_dim % head_dim == 0:
                        num_heads = head_total_dim // head_dim
                        print(f"Possible: num_heads={num_heads}, head_dim={head_dim}")
            else:
                print("⚠ QVK and W0 dimensions are inconsistent")
                print(f"Expected W0 input: {head_total_dim}, actual: {w0_in}")
    
    # Look for decoder/CNN parameters
    print("\nLooking for decoder parameters...")
    for key, tensor in state.items():
        if ('conv' in key or 'decoder' in key) and 'weight' in key:
            print(f"{key}: {tensor.shape}")
    
    return key_params

if __name__ == "__main__":
    checkpoint_path = "./Data/saved_models/brats_t1ce.pth"
    analyze_checkpoint(checkpoint_path)
