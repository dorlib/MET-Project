#!/usr/bin/env python3
# test_model_segmentation.py - Load saved model and run inference on a .npy file

import os
import sys
import numpy as np
import torch
import matplotlib
# Use TkAgg backend for interactive display
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import importlib.util
import subprocess
import pkg_resources
import argparse

# Function to check and install required packages
def check_and_install_packages():
    required_packages = [
        'self-attention-cv',
        'matplotlib',
        'numpy',
        'torch',
    ]
    
    for package in required_packages:
        try:
            dist = pkg_resources.get_distribution(package)
            print(f"{package} ({dist.version}) is installed")
        except pkg_resources.DistributionNotFound:
            print(f"{package} is NOT installed. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Installed {package}")

# Check and install required packages
check_and_install_packages()

# Now import the UNETR model after ensuring the package is installed
from self_attention_cv import UNETR

# Add project root to path to import model adapter
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.model_service.unetr_adapter import UnetrModelAdapter

# Configuration
MODEL_PATH = '../Data/saved_models/brats_t1ce.pth'
INPUT_NPY = 'image_7_00009.npy'
OUTPUT_DIR = 'test_segmentation_results'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="Run UNETR model inference on .npy file with interactive visualization")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to model weights")
    parser.add_argument("--input", default=INPUT_NPY, help="Path to input .npy file")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory for results")
    parser.add_argument("--skip-interactive", action="store_true", help="Skip interactive visualization")
    parser.add_argument("--num-classes", default=NUM_CLASSES, type=int, help="Number of segmentation classes")
    
    return parser.parse_args()

def main():
    """
    Main function to load model, run inference, and display results
    """
    # Parse arguments
    args = parse_arguments()
    
    # Update paths from arguments
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.model))
    input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.input))
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), args.output_dir))
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Using device: {DEVICE}")
    print(f"Loading model from: {model_path}")
    print(f"Processing image: {input_path}")
    print(f"Output directory: {output_dir}")
    
    # Initialize model adapter
    model_adapter = UnetrModelAdapter(
        model_path=model_path,
        device=DEVICE,
        num_classes=args.num_classes
    )
    
    # Load model
    if not model_adapter.load_model():
        print("Failed to load model, exiting.")
        return
    
    # Run inference
    try:
        print("Running inference...")
        results = model_adapter.predict(input_path)
        pred_mask = results['prediction']
        orig_vol = results['original_image']
        
        # Save prediction to file
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        pred_path = os.path.join(output_dir, f"{base_name}_prediction.npy")
        np.save(pred_path, pred_mask)
        print(f"Saved prediction to: {pred_path}")
        
        # Display results
        if not args.skip_interactive:
            display_results(orig_vol, pred_mask, base_name)
        else:
            # Just save the static figure
            save_static_figure(orig_vol, pred_mask, base_name)
            print("Interactive visualization skipped.")
        
    except Exception as e:
        print(f"Error during inference: {e}")

def display_results(orig_vol, pred_mask, base_name):
    """
    Display interactive visualization of original volume and segmentation mask
    """
    # Save static figure first
    save_static_figure(orig_vol, pred_mask, base_name)
    
    # Interactive display
    display_interactive_viewer(orig_vol, pred_mask)

def save_static_figure(orig_vol, pred_mask, base_name):
    """
    Create and save a static figure with middle slices
    """
    # Prepare mid-slices for each axis
    D, H, W = orig_vol.shape
    mids = {'axial': D//2, 'coronal': H//2, 'sagittal': W//2}
    
    # Extract mid-slices from volume and prediction
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
    
    # Create figure with subplots
    fig, axs = plt.subplots(3, 2, figsize=(12, 15), dpi=150)
    
    # Plot slices for each view
    for r, axis in enumerate(['axial', 'coronal', 'sagittal']):
        # Original image
        axs[r, 0].imshow(slices_vol[axis], interpolation='nearest', cmap='gray')
        axs[r, 0].set_title(f"Original {axis}")
        axs[r, 0].axis('off')
        
        # Segmentation mask
        axs[r, 1].imshow(slices_pred[axis], interpolation='nearest', cmap='viridis')
        axs[r, 1].set_title(f"Segmentation {axis}")
        axs[r, 1].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(OUTPUT_DIR, f"{base_name}_segmentation.png")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved static visualization to: {fig_path}")

def display_interactive_viewer(image, prediction):
    """
    Display interactive viewer with sliders for navigating through the volume
    """
    # Get image dimensions
    D, H, W = image.shape
    
    # Initialize with middle slices
    axial_idx = D // 2
    coronal_idx = H // 2
    sagittal_idx = W // 2
    
    # Create figure
    plt.ion()  # Turn on interactive mode
    fig = plt.figure(figsize=(12, 10))
    plt.suptitle("Interactive 3D Segmentation Viewer\nUse sliders to navigate through volume", fontsize=16)
    
    # Layout with 3 rows (axial, coronal, sagittal) and 2 columns (image, prediction)
    ax1 = plt.subplot(3, 2, 1)
    plt.title('Original - Axial')
    img1 = ax1.imshow(image[axial_idx, :, :], cmap='gray', aspect='equal')
    ax1.axis('off')
    
    ax2 = plt.subplot(3, 2, 3)
    plt.title('Original - Coronal')
    img2 = ax2.imshow(image[:, coronal_idx, :], cmap='gray', aspect='equal')
    ax2.axis('off')
    
    ax3 = plt.subplot(3, 2, 5)
    plt.title('Original - Sagittal')
    img3 = ax3.imshow(image[:, :, sagittal_idx], cmap='gray', aspect='equal')
    ax3.axis('off')
    
    # Prediction views
    ax4 = plt.subplot(3, 2, 2)
    plt.title('Segmentation - Axial')
    img4 = ax4.imshow(prediction[axial_idx, :, :], cmap='viridis', aspect='equal')
    ax4.axis('off')
    
    ax5 = plt.subplot(3, 2, 4)
    plt.title('Segmentation - Coronal')
    img5 = ax5.imshow(prediction[:, coronal_idx, :], cmap='viridis', aspect='equal')
    ax5.axis('off')
    
    ax6 = plt.subplot(3, 2, 6)
    plt.title('Segmentation - Sagittal')
    img6 = ax6.imshow(prediction[:, :, sagittal_idx], cmap='viridis', aspect='equal')
    ax6.axis('off')
    
    # Add sliders for navigation
    plt.subplots_adjust(bottom=0.25)
    
    axcolor = 'lightgoldenrodyellow'
    ax_axial = plt.axes([0.15, 0.15, 0.7, 0.03], facecolor=axcolor)
    ax_coronal = plt.axes([0.15, 0.10, 0.7, 0.03], facecolor=axcolor)
    ax_sagittal = plt.axes([0.15, 0.05, 0.7, 0.03], facecolor=axcolor)
    
    s_axial = Slider(ax_axial, 'Axial Slice', 0, D-1, valinit=axial_idx, valstep=1)
    s_coronal = Slider(ax_coronal, 'Coronal Slice', 0, H-1, valinit=coronal_idx, valstep=1)
    s_sagittal = Slider(ax_sagittal, 'Sagittal Slice', 0, W-1, valinit=sagittal_idx, valstep=1)
    
    def update(val):
        axial_slice = int(s_axial.val)
        coronal_slice = int(s_coronal.val)
        sagittal_slice = int(s_sagittal.val)
        
        # Update image data
        img1.set_data(image[axial_slice, :, :])
        img2.set_data(image[:, coronal_slice, :])
        img3.set_data(image[:, :, sagittal_slice])
        
        img4.set_data(prediction[axial_slice, :, :])
        img5.set_data(prediction[:, coronal_slice, :])
        img6.set_data(prediction[:, :, sagittal_slice])
        
        fig.canvas.draw_idle()
    
    s_axial.on_changed(update)
    s_coronal.on_changed(update)
    s_sagittal.on_changed(update)
    
    plt.tight_layout(rect=[0, 0.25, 1, 1])  # Adjust layout to make room for sliders
    
    print("\nInteractive visualization displayed.")
    print("Use the sliders at the bottom to navigate through the 3D volume.")
    print("Close the plot window when done.")
    
    plt.draw()
    plt.pause(0.001)  # Small pause to ensure the figure is displayed
    plt.show(block=True)  # Keep the plot open until closed

if __name__ == "__main__":
    # Give clear instructions for the user
    print("=" * 80)
    print("UNETR Model Segmentation with Interactive Visualization")
    print("=" * 80)
    print("This script will:")
    print("1. Load the UNETR model from Data/saved_models/")
    print("2. Run inference on a .npy file")
    print("3. Display an interactive 3D visualization with slider controls")
    print("4. Save both the segmentation results and visualization images")
    print("\nUsage examples:")
    print("  python test_model_segmentation.py")
    print("  python test_model_segmentation.py --input image_7_00009.npy")
    print("  python test_model_segmentation.py --model '../Data/saved_models/brats_t1ce.pth'")
    print("  python test_model_segmentation.py --skip-interactive")
    print("-" * 80)
    
    try:
        main()
        print("\nScript completed successfully!")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease check the error message above and ensure all dependencies are installed correctly.")
