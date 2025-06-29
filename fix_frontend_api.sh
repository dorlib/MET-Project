#!/bin/bash

echo "Fixing frontend API configuration..."

# Check the frontend API configuration
echo "Examining frontend API.js file..."
API_FILE=$(find /Users/dorliber/MET-Project/frontend/src -name "api.js" -o -name "api.ts")

if [ -z "$API_FILE" ]; then
    echo "Could not find frontend API file"
    exit 1
fi

echo "Found API file: $API_FILE"
cat "$API_FILE"

# Check the frontend upload component
echo "Examining UploadForm component..."
UPLOAD_COMPONENT=$(find /Users/dorliber/MET-Project/frontend/src -name "UploadForm.js" -o -name "UploadForm.jsx" -o -name "UploadForm.tsx")

if [ -z "$UPLOAD_COMPONENT" ]; then
    echo "Could not find UploadForm component"
    exit 1
fi

echo "Found UploadForm component: $UPLOAD_COMPONENT"
cat "$UPLOAD_COMPONENT"

echo "Creating a custom curl upload test..."
# Create a simple file for upload testing
mkdir -p /tmp/met_project_test
dd if=/dev/urandom of=/tmp/met_project_test/test_scan.bin bs=1k count=128
echo "Created test file: /tmp/met_project_test/test_scan.bin"

# Test upload with curl
echo "Testing upload with curl..."
curl -v -F "file=@/tmp/met_project_test/test_scan.bin" http://localhost:8000/upload

# Test with different field name
echo "Testing upload with different field name..."
curl -v -F "scan=@/tmp/met_project_test/test_scan.bin" http://localhost:8000/upload

# Test with both file field and JSON data
echo "Testing upload with file and JSON data..."
curl -v -F "file=@/tmp/met_project_test/test_scan.bin" -F "metadata={\"patient_id\":\"test123\"};type=application/json" http://localhost:8000/upload

echo "Tests complete."
