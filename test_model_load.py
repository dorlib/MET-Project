#!/usr/bin/env python3
"""
Simple Model Test - Check if we can load the UNETR model
"""

import torch
import numpy as np

# Try to import the correct UNETR implementation
try:
    from self_attention_cv import UNETR
    print("✓ self_attention_cv UNETR found")
    
    # Test model creation
    model = UNETR(
        in_channels=1,
        img_size=(128, 128, 128),
        patch_size=16,
        num_classes=4,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4
    )
    print("✓ Model created successfully")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test model loading
    model_path = "/Users/dorliber/MET-Project/Data/saved_models/brats_t1ce.pth"
    try:
        state = torch.load(model_path, map_location='cpu')
        print(f"✓ Model weights loaded, keys: {len(state.keys())}")
        
        # Print some key names to understand the structure
        print("Sample state dict keys:")
        for i, key in enumerate(list(state.keys())[:10]):
            print(f"  {key}")
        if len(state.keys()) > 10:
            print("  ...")
            
        # Try to load state dict
        model.load_state_dict(state)
        print("✓ Model weights loaded successfully")
        
        # Test inference with dummy data
        model.eval()
        dummy_input = torch.randn(1, 1, 128, 128, 128)
        with torch.no_grad():
            output = model(dummy_input)
            print(f"✓ Model inference successful, output shape: {output.shape}")
            
    except Exception as e:
        print(f"✗ Error loading model weights: {e}")
        print("\nState dict keys in the file:")
        for i, key in enumerate(list(state.keys())[:20]):
            print(f"  {key}")
        if len(state.keys()) > 20:
            print("  ...")
            
        print("\nModel state dict keys:")
        model_keys = list(model.state_dict().keys())
        for i, key in enumerate(model_keys[:20]):
            print(f"  {key}")
        if len(model_keys) > 20:
            print("  ...")
        
except ImportError as e:
    print(f"✗ self_attention_cv not found: {e}")
except Exception as e:
    print(f"✗ Error: {e}")
