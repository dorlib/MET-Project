#!/bin/bash
# test_advanced_processing.sh - Test script for advanced image processing features

set -e  # Exit on error

# ANSI color codes
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Define API Gateway URL
API_URL="http://localhost:5000"

echo -e "${CYAN}==== Testing Advanced Image Processing Features ====${NC}"
echo ""

# Obtain auth token (adjust based on your auth implementation)
echo -e "${YELLOW}Step 1: Obtaining authentication token...${NC}"
AUTH_RESPONSE=$(curl -s -X POST -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"password123"}' \
  "${API_URL}/auth/login")
TOKEN=$(echo ${AUTH_RESPONSE} | grep -o '"token":"[^"]*' | cut -d'"' -f4)

if [ -z "$TOKEN" ]; then
  echo -e "${RED}Failed to get auth token. Using test mode without auth.${NC}"
  AUTH_HEADER=""
else
  AUTH_HEADER="Authorization: Bearer ${TOKEN}"
  echo -e "${GREEN}Authentication token obtained.${NC}"
fi

# Test upload with sample file
echo -e "\n${YELLOW}Step 2: Uploading a sample file...${NC}"
RESPONSE=$(curl -s -X POST -H "Content-Type: multipart/form-data" \
  -H "${AUTH_HEADER}" \
  -F "file=@./test_data/sample_nifti.nii.gz" \
  "${API_URL}/upload")

# Extract job_id
JOB_ID=$(echo ${RESPONSE} | grep -o '"job_id":"[^"]*' | cut -d'"' -f4)

if [ -z "$JOB_ID" ]; then
  echo -e "${RED}Failed to upload file or retrieve job ID.${NC}"
  echo "Response was: ${RESPONSE}"
  exit 1
fi

echo -e "${GREEN}File uploaded successfully. Job ID: ${JOB_ID}${NC}"

# Wait for processing to complete (may take some time)
echo -e "\n${YELLOW}Step 3: Waiting for the model to process the scan...${NC}"
echo "This may take several minutes. Checking status every 5 seconds."

STATUS="processing"
RETRIES=0
MAX_RETRIES=60  # 5 minutes

while [ "$STATUS" != "completed" ] && [ $RETRIES -lt $MAX_RETRIES ]; do
  RESPONSE=$(curl -s -H "${AUTH_HEADER}" "${API_URL}/results/${JOB_ID}")
  STATUS=$(echo ${RESPONSE} | grep -o '"status":"[^"]*' | cut -d'"' -f4)
  
  if [ "$STATUS" == "completed" ]; then
    echo -e "${GREEN}Processing completed!${NC}"
    break
  elif [ "$STATUS" == "failed" ]; then
    echo -e "${RED}Processing failed.${NC}"
    echo "Response was: ${RESPONSE}"
    exit 1
  else
    echo -n "."
    sleep 5
    RETRIES=$((RETRIES + 1))
  fi
done

if [ $RETRIES -eq $MAX_RETRIES ]; then
  echo -e "${RED}\nProcessing timed out after 5 minutes.${NC}"
  exit 1
fi

echo ""
echo -e "${GREEN}Basic segmentation and analysis completed successfully.${NC}"

# Test advanced analysis API
echo -e "\n${YELLOW}Step 4: Testing Advanced Analysis API...${NC}"
RESPONSE=$(curl -s -H "${AUTH_HEADER}" "${API_URL}/advanced-analysis/${JOB_ID}")

if echo "${RESPONSE}" | grep -q "total_classes_found"; then
  echo -e "${GREEN}Advanced analysis successful!${NC}"
  echo "Classes found: $(echo ${RESPONSE} | grep -o '"total_classes_found":[^,]*' | cut -d':' -f2)"
  echo "Total regions: $(echo ${RESPONSE} | grep -o '"total_regions":[^,]*' | cut -d':' -f2)"
else
  echo -e "${RED}Advanced analysis failed.${NC}"
  echo "Response was: ${RESPONSE}"
fi

# Test lesion analysis API
echo -e "\n${YELLOW}Step 5: Testing Lesion Analysis API...${NC}"
RESPONSE=$(curl -s -H "${AUTH_HEADER}" "${API_URL}/lesion-analysis/${JOB_ID}")

if echo "${RESPONSE}" | grep -q "count"; then
  echo -e "${GREEN}Lesion analysis successful!${NC}"
else
  echo -e "${RED}Lesion analysis failed.${NC}"
  echo "Response was: ${RESPONSE}"
fi

# Test slice summary API
echo -e "\n${YELLOW}Step 6: Testing Slice Summary API...${NC}"
RESPONSE=$(curl -s -H "${AUTH_HEADER}" "${API_URL}/slice-summary/${JOB_ID}")

if echo "${RESPONSE}" | grep -q "slices_with_segmentation"; then
  echo -e "${GREEN}Slice summary analysis successful!${NC}"
  echo "Total slices: $(echo ${RESPONSE} | grep -o '"total_slices":[^,]*' | cut -d':' -f2)"
  echo "Slices with segmentation: $(echo ${RESPONSE} | grep -o '"slices_with_segmentation":[^,]*' | cut -d':' -f2)"
else
  echo -e "${RED}Slice summary analysis failed.${NC}"
  echo "Response was: ${RESPONSE}"
fi

# Test visualization APIs (saving images)
echo -e "\n${YELLOW}Step 7: Testing Advanced Visualizations...${NC}"

# Test slice visualization
echo "Downloading slice visualization..."
curl -s -H "${AUTH_HEADER}" "${API_URL}/advanced-visualization/${JOB_ID}?type=slice" > "${JOB_ID}_slice.png"
if [ -s "${JOB_ID}_slice.png" ]; then
  echo -e "${GREEN}Slice visualization downloaded successfully.${NC}"
else
  echo -e "${RED}Failed to download slice visualization.${NC}"
fi

# Test projection visualization
echo "Downloading projection visualization..."
curl -s -H "${AUTH_HEADER}" "${API_URL}/advanced-visualization/${JOB_ID}?type=projection" > "${JOB_ID}_projection.png"
if [ -s "${JOB_ID}_projection.png" ]; then
  echo -e "${GREEN}Projection visualization downloaded successfully.${NC}"
else
  echo -e "${RED}Failed to download projection visualization.${NC}"
fi

# Test multi-slice visualization
echo "Downloading multi-slice visualization..."
curl -s -H "${AUTH_HEADER}" "${API_URL}/advanced-visualization/${JOB_ID}?type=multi-slice&num_slices=3" > "${JOB_ID}_multi_slice.png"
if [ -s "${JOB_ID}_multi_slice.png" ]; then
  echo -e "${GREEN}Multi-slice visualization downloaded successfully.${NC}"
else
  echo -e "${RED}Failed to download multi-slice visualization.${NC}"
fi

# Test lesion visualization
echo "Downloading lesion visualization..."
curl -s -H "${AUTH_HEADER}" "${API_URL}/advanced-visualization/${JOB_ID}?type=lesions" > "${JOB_ID}_lesions.png"
if [ -s "${JOB_ID}_lesions.png" ]; then
  echo -e "${GREEN}Lesion visualization downloaded successfully.${NC}"
else
  echo -e "${RED}Failed to download lesion visualization.${NC}"
fi

# Test updating metadata
echo -e "\n${YELLOW}Step 8: Testing Metadata Updates...${NC}"
RESPONSE=$(curl -s -X POST -H "Content-Type: application/json" \
  -H "${AUTH_HEADER}" \
  -d '{"voxel_volume_mm3": 1.2, "tissue_names": {"1": "Tumor Core", "2": "Edema Zone", "3": "Active Metastasis"}}' \
  "${API_URL}/analysis-metadata")

if echo "${RESPONSE}" | grep -q "Metadata updated successfully"; then
  echo -e "${GREEN}Metadata updated successfully!${NC}"
else
  echo -e "${RED}Metadata update failed.${NC}"
  echo "Response was: ${RESPONSE}"
fi

echo -e "\n${CYAN}==== Advanced Image Processing Tests Complete ====${NC}"
echo -e "${GREEN}Test images saved:${NC}"
echo "- ${JOB_ID}_slice.png"
echo "- ${JOB_ID}_projection.png"
echo "- ${JOB_ID}_multi_slice.png"
echo "- ${JOB_ID}_lesions.png"
echo ""
echo "You can open these images to verify the visualizations."
