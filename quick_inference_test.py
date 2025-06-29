#!/usr/bin/env python3
"""
Quick test script to verify model loading and inference works
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from self_attention_cv import UNETR

def test_model_loading():
    """Test if we can load the model successfully"""
    print("Testing model loading...")
    
    device = torch.device('cpu')
    
    # Use exact parameters from training script
    model = UNETR(
        img_shape=(128, 128, 128),
        input_dim=1,
        output_dim=4,  # NUM_CLASSES = 4
        embed_dim=128,
        patch_size=16,
        num_heads=8,
        ext_layers=[3, 6, 9, 12, 15, 18],
        norm='instance',
        dropout=0.2,
        base_filters=16,
        dim_linear_block=1024
    ).to(device)
    
    print("Model created successfully")
    
    # Load checkpoint
    checkpoint_path = "./Data/saved_models/brats_t1ce.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return False
        
    try:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        print("✅ Model loaded successfully!")
        model.eval()
        
        # Test with dummy input
        dummy_input = torch.randn(1, 1, 128, 128, 128).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✅ Inference test successful! Output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

def test_real_data():
    """Test with actual data if model loading works"""
    print("\nTesting with real data...")
    
    # Load the test scan
    scan_path = "test_image.npy"
    if not os.path.exists(scan_path):
        print(f"Test image not found at {scan_path}")
        return False
        
    scan = np.load(scan_path)
    print(f"Loaded scan with shape: {scan.shape}")
    
    # Take the first channel if multi-channel
    if len(scan.shape) == 4:
        scan = scan[:, :, :, 0]  # Take first channel (T1CE)
    
    print(f"Using scan shape: {scan.shape}")
    
    # Normalize
    scan = (scan - np.mean(scan)) / (np.std(scan) + 1e-8)
    
    # Convert to tensor
    device = torch.device('cpu')
    scan_tensor = torch.FloatTensor(scan).unsqueeze(0).unsqueeze(0).to(device)
    print(f"Tensor shape: {scan_tensor.shape}")
    
    # Load model
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
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model(scan_tensor)
    
    prediction = torch.softmax(outputs, dim=1)
    prediction = torch.argmax(prediction, dim=1).squeeze().cpu().numpy()
    
    print(f"Prediction shape: {prediction.shape}")
    print(f"Unique values in prediction: {np.unique(prediction)}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Get middle slices
    z_mid = scan.shape[2] // 2
    y_mid = scan.shape[1] // 2
    x_mid = scan.shape[0] // 2
    
    # Original images
    axes[0, 0].imshow(scan[:, :, z_mid], cmap='gray')
    axes[0, 0].set_title('Original - Axial')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(scan[:, y_mid, :], cmap='gray')
    axes[0, 1].set_title('Original - Coronal')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(scan[x_mid, :, :], cmap='gray')
    axes[0, 2].set_title('Original - Sagittal')
    axes[0, 2].axis('off')
    
    # Predictions overlaid
    axes[1, 0].imshow(scan[:, :, z_mid], cmap='gray')
    axes[1, 0].imshow(prediction[:, :, z_mid], cmap='jet', alpha=0.5)
    axes[1, 0].set_title('Segmentation - Axial')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(scan[:, y_mid, :], cmap='gray')
    axes[1, 1].imshow(prediction[:, y_mid, :], cmap='jet', alpha=0.5)
    axes[1, 1].set_title('Segmentation - Coronal')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(scan[x_mid, :, :], cmap='gray')
    axes[1, 2].imshow(prediction[x_mid, :, :], cmap='jet', alpha=0.5)
    axes[1, 2].set_title('Segmentation - Sagittal')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    output_path = 'quick_inference_result.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    # Don't show plot to avoid hanging
    # plt.show()
    plt.close()  # Close the figure to free memory
    
    print(f"✅ Visualization saved to {output_path}")
    return True

if __name__ == "__main__":
    print("=== Quick Inference Test ===")
    
    # Test model loading first
    if test_model_loading():
        # If successful, test with real data
        test_real_data()
    else:
        print("Model loading failed - skipping real data test")
