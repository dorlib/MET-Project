#!/bin/bash

# Test script to check if the model service can handle smaller files

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Creating small test NPY file for testing...${NC}"

# Navigate to the test_data directory
cd /Users/dorliber/Library/CloudStorage/OneDrive-IntelCorporation/Desktop/private/MET-project/test_data

# Use Python to create a smaller test file
python3 -c '
import numpy as np
import os

# Create a small 32x32x32 volume
small_volume = np.zeros((32, 32, 32), dtype=np.float32)

# Add some simple patterns for visual interest
for i in range(32):
    small_volume[i, :, :] = i / 32.0
    
    # Add a "lesion" sphere
    if 10 <= i <= 20:
        center_y, center_z = 16, 16
        radius = 5
        for y in range(32):
            for z in range(32):
                if ((y - center_y) ** 2 + (z - center_z) ** 2) <= (radius ** 2):
                    small_volume[i, y, z] = 1.0

# Save the file
np.save("small_test_volume.npy", small_volume)
print(f"Created small_test_volume.npy with shape {small_volume.shape}")
'

echo -e "${GREEN}Created test file. Now testing upload...${NC}"

# Optional: Restart model service to ensure clean state
echo -e "${YELLOW}Restarting model service...${NC}"
docker-compose restart model-service
echo -e "${GREEN}Model service restarted.${NC}"

# Wait for service to start
echo -e "${YELLOW}Waiting for service to initialize...${NC}"
sleep 10

# Upload the file using curl
echo -e "${YELLOW}Uploading test file...${NC}"
curl -X POST -F "file=@/Users/dorliber/Library/CloudStorage/OneDrive-IntelCorporation/Desktop/private/MET-project/test_data/small_test_volume.npy" http://localhost:8000/upload

echo
echo -e "${GREEN}Test complete. Check model service logs for processing status.${NC}"
echo -e "${YELLOW}Run: docker logs -f met-project_model-service_1${NC}"
