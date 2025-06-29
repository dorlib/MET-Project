#!/usr/bin/env python3
"""
Quick test to understand the self_attention_cv UNETR parameters
"""

import torch
from self_attention_cv import UNETR
import inspect

print("=== UNETR Parameters ===")
sig = inspect.signature(UNETR.__init__)
print("UNETR.__init__ signature:")
for param_name, param in sig.parameters.items():
    if param_name != 'self':
        print(f"  {param_name}: {param}")

# Try to create a minimal UNETR instance to see what works
print("\n=== Testing minimal UNETR creation ===")
try:
    model = UNETR(
        img_size=128,
        patch_size=16,
        num_classes=4,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4
    )
    print("✓ Basic UNETR creation successful")
    print(f"Model: {model}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n=== Testing with depth parameter ===")
try:
    model = UNETR(
        img_size=128,
        patch_size=16,
        num_classes=4,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4
    )
    print("✓ UNETR with depth parameter successful")
    print(f"Model: {model}")
except Exception as e:
    print(f"✗ Error: {e}")
