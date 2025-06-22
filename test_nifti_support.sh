#!/bin/bash
# Script to test NIfTI file upload and processing

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=========================================${NC}"
echo -e "${YELLOW}MET Brain Metastasis Analysis System${NC}"
echo -e "${YELLOW}NIfTI File Support Test Script${NC}"
echo -e "${YELLOW}=========================================${NC}"

# Check if curl is installed
if ! command -v curl &> /dev/null; then
    echo -e "${RED}Error: curl is not installed. Please install curl and try again.${NC}"
    exit 1
fi

API_GATEWAY_URL="http://localhost:8000"
TEST_DATA_DIR="./test_data"
SAMPLE_FILE_PATH="${TEST_DATA_DIR}/sample_nifti.nii.gz"

# Check if sample file exists
if [ ! -f "$SAMPLE_FILE_PATH" ]; then
    echo -e "${YELLOW}Sample NIfTI file not found. Creating a test file...${NC}"
    
    # Generate a simple test NIfTI file using python (if available)
    if command -v python3 &> /dev/null; then
        echo "Creating a sample NIfTI file using Python..."
        python3 -c '
import numpy as np
import nibabel as nib
import os

try:
    # Create a simple 3D array (simulating a brain scan)
    data = np.zeros((128, 128, 128), dtype=np.float32)
    
    # Add some "structures" to make it look somewhat like a brain scan
    # Center sphere (simulating brain)
    center = np.array([64, 64, 64])
    for i in range(128):
        for j in range(128):
            for k in range(128):
                dist = np.sqrt(np.sum((np.array([i, j, k]) - center)**2))
                if dist < 50:
                    data[i, j, k] = 0.8 - dist/100
    
    # Add a few "lesions" (brighter spots)
    for pos in [(40, 70, 60), (80, 50, 70), (60, 90, 50)]:
        x, y, z = pos
        r = 5  # radius
        for i in range(max(0, x-r), min(128, x+r)):
            for j in range(max(0, y-r), min(128, y+r)):
                for k in range(max(0, z-r), min(128, z+r)):
                    dist = np.sqrt(np.sum((np.array([i, j, k]) - np.array([x, y, z]))**2))
                    if dist < r:
                        data[i, j, k] = 1.0 - dist/(r*2)
    
    # Create NIfTI image and save it
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    os.makedirs("./test_data", exist_ok=True)
    nib.save(img, "./test_data/sample_nifti.nii.gz")
    print("Successfully created sample NIfTI file: ./test_data/sample_nifti.nii.gz")
except ImportError as e:
    print(f"Error: {e}. Please install nibabel with: pip install nibabel")
except Exception as e:
    print(f"Error creating sample NIfTI file: {e}")
'
    else
        echo -e "${RED}Python3 not found. Cannot create sample NIfTI file.${NC}"
        echo -e "${YELLOW}Please download a sample NIfTI file manually and place it at: ${SAMPLE_FILE_PATH}${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}Found existing sample NIfTI file: ${SAMPLE_FILE_PATH}${NC}"
fi

# Verify the file exists after attempting to create it
if [ ! -f "$SAMPLE_FILE_PATH" ]; then
    echo -e "${RED}Failed to create or find sample NIfTI file.${NC}"
    exit 1
fi

# Check the size of the sample file
FILE_SIZE=$(du -k "$SAMPLE_FILE_PATH" | cut -f1)
echo -e "${GREEN}Sample file size: ${FILE_SIZE} KB${NC}"

# Test the API Gateway health endpoint
echo -e "\n${YELLOW}Testing API Gateway health...${NC}"
HEALTH_RESPONSE=$(curl -s "${API_GATEWAY_URL}/health")

if [[ "$HEALTH_RESPONSE" == *"healthy"* ]]; then
    echo -e "${GREEN}API Gateway is healthy.${NC}"
else
    echo -e "${RED}API Gateway health check failed. Response: ${HEALTH_RESPONSE}${NC}"
    echo -e "${YELLOW}Make sure your containers are running properly.${NC}"
    exit 1
fi

# Upload the NIfTI file
echo -e "\n${YELLOW}Uploading NIfTI file to the API Gateway...${NC}"
echo -e "${YELLOW}This may take a while for larger files...${NC}"

UPLOAD_RESPONSE=$(curl -s -X POST \
    -F "file=@${SAMPLE_FILE_PATH}" \
    "${API_GATEWAY_URL}/upload")

if [[ "$UPLOAD_RESPONSE" == *"job_id"* ]]; then
    JOB_ID=$(echo $UPLOAD_RESPONSE | grep -o '"job_id":"[^"]*' | sed 's/"job_id":"//')
    echo -e "${GREEN}File uploaded successfully. Job ID: ${JOB_ID}${NC}"
    
    # Poll for results
    echo -e "\n${YELLOW}Waiting for processing to complete...${NC}"
    echo -e "${YELLOW}This may take several minutes for NIfTI files...${NC}"
    
    MAX_ATTEMPTS=20
    ATTEMPT=0
    SLEEP_SECONDS=15
    
    while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
        ATTEMPT=$((ATTEMPT + 1))
        echo -e "${YELLOW}Checking status (attempt ${ATTEMPT}/${MAX_ATTEMPTS})...${NC}"
        
        RESULT_RESPONSE=$(curl -s "${API_GATEWAY_URL}/results/${JOB_ID}")
        
        if [[ "$RESULT_RESPONSE" == *"completed"* ]]; then
            echo -e "\n${GREEN}Processing completed successfully!${NC}"
            echo -e "${GREEN}Result: ${RESULT_RESPONSE}${NC}"
            exit 0
        elif [[ "$RESULT_RESPONSE" == *"processing"* ]]; then
            echo -e "${YELLOW}Still processing... Waiting ${SLEEP_SECONDS} seconds.${NC}"
        else
            echo -e "${RED}Unexpected response: ${RESULT_RESPONSE}${NC}"
            echo -e "${YELLOW}Continuing to wait...${NC}"
        fi
        
        sleep $SLEEP_SECONDS
    done
    
    echo -e "\n${RED}Processing timed out after ${MAX_ATTEMPTS} attempts.${NC}"
    echo -e "${YELLOW}The system may still be processing your file. Check the logs for more information.${NC}"
    
else
    echo -e "${RED}Upload failed. Response: ${UPLOAD_RESPONSE}${NC}"
    exit 1
fi
