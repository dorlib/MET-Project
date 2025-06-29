#!/bin/bash

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
  python3 -m venv venv
  echo "Created new virtual environment"
fi

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Ensure model directory exists
mkdir -p /app/models

# Check if model file exists in Data/saved_models
if [ -f "../Data/saved_models/brats_t1ce.pth" ]; then
  echo "Using existing model file"
  mkdir -p /app/models
  cp ../Data/saved_models/brats_t1ce.pth /app/models/
else
  echo "Warning: Model file not found. Create a placeholder"
  mkdir -p /app/models
  dd if=/dev/zero of=/app/models/brats_t1ce.pth bs=1024 count=1
fi

# Run the service
python model_service.py
