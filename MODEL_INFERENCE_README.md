# MET Model Inference Guide

This guide explains how to use the model inference script to run the pre-trained UNETR model on brain MRI scans.

## Prerequisites

1. Python 3.6+
2. Required packages:
   - numpy
   - torch
   - monai
   - matplotlib

You can install these with:
```
pip install numpy torch monai matplotlib
```

## Using the Model Inference Script

The `model_inference.py` script allows you to run the pre-trained model on .npy files:

```bash
python model_inference.py <input_npy_file> [--output_file OUTPUT_FILE] [--model_path MODEL_PATH] [--visualize]
```

### Arguments:

- `input_file` (required): Path to the input .npy file containing the brain MRI scan
- `--output_file` (optional): Path to save the prediction result. Defaults to `<input_file>_prediction.npy`
- `--model_path` (optional): Path to the model weights file. Defaults to `./Data/saved_models/brats_t1ce.pth`
- `--visualize` (optional): Generate and save visualization of the results

### Examples:

1. Basic usage:
   ```
   python model_inference.py test_image.npy
   ```
   This will save the prediction to `test_image_prediction.npy`.

2. With custom output path:
   ```
   python model_inference.py test_image.npy --output_file my_prediction.npy
   ```

3. With visualization:
   ```
   python model_inference.py test_image.npy --visualize
   ```
   This will save both the prediction and a visualization image.

## About the Model

The model used is a UNETR (UNet Transformer) architecture trained on brain MRI scans for MET segmentation. It expects input scans with shape compatible with (128, 128, 128) dimensions.

If your input data has multiple channels (e.g., from preprocessing.py), the script will automatically select the T1CE channel (index 1).

## Input Data Format

The input should be a numpy `.npy` file containing a brain MRI scan. It can be:
1. A single-channel 3D volume with shape (H, W, D)
2. A multi-channel 3D volume with shape (H, W, D, C) where C is the number of channels

For multi-channel inputs, the script will automatically extract the T1CE channel (index 1).

## Output Format

The output is a numpy `.npy` file containing the segmentation prediction. Each voxel value represents a class:
- 0: Background
- 1, 2, 3: Different MET classes

The output has the same spatial dimensions as the input.
