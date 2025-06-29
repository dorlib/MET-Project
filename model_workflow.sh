#!/bin/bash
# model_workflow.sh - Demonstrates the workflow for using the MET model

set -e  # Exit on any error

# Print colored status messages
function print_status() {
  echo -e "\n\033[1;34m==== $1 ====\033[0m"
}

# Check if the input file was provided
if [ $# -lt 1 ]; then
  echo "Usage: $0 <input_npy_file>"
  echo "Example: $0 test_image.npy"
  exit 1
fi

INPUT_FILE="$1"
MODEL_PATH="./Data/saved_models/brats_t1ce.pth"
OUTPUT_FILE="${INPUT_FILE%.*}_prediction.npy"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
  echo "Error: Input file not found: $INPUT_FILE"
  exit 1
fi

# Check if model file exists
if [ ! -f "$MODEL_PATH" ]; then
  echo "Error: Model file not found: $MODEL_PATH"
  echo "Make sure you have a model file at $MODEL_PATH"
  exit 1
fi

# Step 1: Inspect the input file
print_status "INSPECTING INPUT FILE"
python inspect_npy.py "$INPUT_FILE"

# Step 2: Run the model inference
print_status "RUNNING MODEL INFERENCE"
python model_inference.py "$INPUT_FILE" --visualize

# Step 3: Inspect the output file
print_status "INSPECTING PREDICTION OUTPUT"
python inspect_npy.py "$OUTPUT_FILE"

print_status "WORKFLOW COMPLETE"
echo "Input file: $INPUT_FILE"
echo "Output file: $OUTPUT_FILE"
echo "Visualization saved: ${OUTPUT_FILE%.*}_visualization.png"
