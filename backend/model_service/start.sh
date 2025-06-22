#!/bin/bash
# Startup script for model service with proper signal handling and memory optimization

# Set memory optimization environment variables if not already set
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:128"}

echo "Starting model service with memory optimization"
echo "Environment configuration:"
echo "- Device: ${DEVICE:-CPU}"
echo "- Workers: ${WORKERS:-1}"
echo "- Mock Model: ${MOCK_MODEL:-False}"
echo "- Thread limits: OMP=${OMP_NUM_THREADS}, MKL=${MKL_NUM_THREADS}"

# Report memory at startup
free -m || echo "free command not available"

# Check model file
echo "Checking model file at ${MODEL_PATH:-/app/models/brats_t1ce.pth}"
if [ ! -f "${MODEL_PATH:-/app/models/brats_t1ce.pth}" ] || [ ! -s "${MODEL_PATH:-/app/models/brats_t1ce.pth}" ]; then
    echo 'Warning: Model file not found or empty. Creating placeholder.'
    mkdir -p $(dirname "${MODEL_PATH:-/app/models/brats_t1ce.pth}")
    dd if=/dev/zero of="${MODEL_PATH:-/app/models/brats_t1ce.pth}" bs=1024 count=1
    echo 'Created placeholder model file. Replace with actual model in production.'
else
    echo 'Model file exists and is non-empty.'
    ls -lh "${MODEL_PATH:-/app/models/brats_t1ce.pth}"
fi

# Start Gunicorn with proper signal handling
exec gunicorn --bind 0.0.0.0:5001 \
    --timeout 600 \
    --workers ${WORKERS:-1} \
    --threads 2 \
    --max-requests 1 \
    --max-requests-jitter 0 \
    --graceful-timeout 300 \
    --log-level info \
    model_service:app
