# Using Real Segmentation Model for MET Project

## Overview

The project has been updated to use real model-based segmentation using the UNETR (U-Net Transformer) architecture for MRI brain metastasis segmentation.

## Available Model

The system includes a pre-trained UNETR model for T1c MRI segmentation:
- Model file: `/Data/saved_models/brats_t1ce.pth` (26MB)
- Model architecture: UNETR (Transformer-based U-Net)
- Input: T1-weighted contrast-enhanced MRI scans
- Output: Segmentation masks with the following classes:
  - Class 0: Background
  - Class 1: Metastasis
  - Class 2: Edema

## How to Use Real Segmentation

1. Run the provided script to enable real model-based segmentation:
   ```bash
   ./use_real_model_segmentation.sh
   ```

2. This script will:
   - Stop the current model service
   - Configure the environment to use the real UNETR model
   - Rebuild and restart the model service
   - Copy the model file into the container

3. After the model service is running, you can test it with:
   ```bash
   ./test_real_model_segmentation.py
   ```
   
4. To use a specific test file:
   ```bash
   ./test_real_model_segmentation.py path/to/your/mri_scan.npy
   ```

## Implementation Details

The real model segmentation uses:
1. `model_service.py` - The main service that handles HTTP endpoints
2. `unetr_adapter.py` - Adapter for the UNETR model that handles:
   - Model loading and initialization
   - Preprocessing of input data
   - Running inference with the model
   - Postprocessing of results

## Docker Configuration

The Dockerfile has been updated to:
1. Copy the model file from the host to the container
2. Use proper memory management settings for PyTorch
3. Use the real model instead of the mock implementation

## Troubleshooting

If you encounter issues:

1. Check Docker logs:
   ```bash
   docker-compose logs model-service
   ```

2. Verify the model file is properly copied:
   ```bash
   docker exec -it met-project_model-service_1 ls -la /app/models/
   ```

3. Ensure the image processing service can access the model service:
   ```bash
   docker exec -it met-project_image-processing-service_1 curl -X GET http://model-service:5001/health
   ```
