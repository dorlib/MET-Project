#!/bin/bash
# Script to verify model file paths and create necessary directories

# Define variables
MODEL_DIR="./Data/saved_models"
SAMPLE_MODEL="brats_t1ce.pth"
SAMPLE_MODEL_PATH="${MODEL_DIR}/${SAMPLE_MODEL}"

# Check if model directory exists, create if not
if [ ! -d "$MODEL_DIR" ]; then
    echo "Creating model directory: $MODEL_DIR"
    mkdir -p "$MODEL_DIR"
fi

# Check if model file exists
if [ ! -f "$SAMPLE_MODEL_PATH" ]; then
    echo "Warning: Model file $SAMPLE_MODEL_PATH not found."
    echo "Creating an empty placeholder file for testing."
    echo "In production, please replace with actual trained model file."
    
    # Create a directory-specific README
    cat > "${MODEL_DIR}/README.md" << EOL
# Model Directory

This directory should contain the following trained model files:

- \`brats_t1ce.pth\`: UNETR model trained on BraTS dataset for T1ce modality

If you're seeing this file, it means the actual model files may not have been properly
installed or configured yet.

## Model Acquisition

Please follow the instructions in the main project README for obtaining and placing
the proper model files in this directory.
EOL

    # Create empty placeholder file (1KB) for development/testing purposes
    dd if=/dev/zero of="$SAMPLE_MODEL_PATH" bs=1024 count=1 2>/dev/null
    
    echo "Created placeholder file. Size: $(du -h "$SAMPLE_MODEL_PATH" | cut -f1)"
    echo "NOTE: This is NOT a working model! Replace with actual model file before use."
else
    echo "Model file exists: $SAMPLE_MODEL_PATH"
    echo "Size: $(du -h "$SAMPLE_MODEL_PATH" | cut -f1)"
fi

echo "Model verification complete."
