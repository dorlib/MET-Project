FROM python:3.9-slim

WORKDIR /app

# Set environment variables for better memory management
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    MALLOC_TRIM_THRESHOLD_=65536

# Install system dependencies for matplotlib with minimal size
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    procps \
    gcc \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for data
RUN mkdir -p /app/uploads /app/results /app/models /app/data

# Copy application code
COPY . .

# Copy model files to the container
COPY ./../../Data/saved_models/brats_t1ce.pth /app/models/brats_t1ce.pth

# Create a startup script to properly handle signals
RUN echo '#!/bin/bash' > /app/start.sh && \
    echo '# Start Gunicorn with proper signal handling' >> /app/start.sh && \
    echo 'exec gunicorn --bind 0.0.0.0:5001 \\' >> /app/start.sh && \
    echo '  --timeout 600 \\' >> /app/start.sh && \
    echo '  --workers 1 \\' >> /app/start.sh && \
    echo '  --threads 2 \\' >> /app/start.sh && \
    echo '  --max-requests 1 \\' >> /app/start.sh && \
    echo '  --max-requests-jitter 0 \\' >> /app/start.sh && \
    echo '  --graceful-timeout 300 \\' >> /app/start.sh && \
    echo '  --log-level info \\' >> /app/start.sh && \
    echo '  model_service:app' >> /app/start.sh && \
    chmod +x /app/start.sh

EXPOSE 5001

# Use shell form of CMD to ensure proper signal handling
CMD ["/app/start.sh"]
