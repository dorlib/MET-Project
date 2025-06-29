#!/bin/bash

echo "Testing upload via API..."
SCAN_ID="test_$(date +%s)"
echo "Using scan_id: $SCAN_ID"

echo "Submitting request..."
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d "{\"scan_id\": \"$SCAN_ID\"}" | jq

echo "Waiting 10 seconds before checking status..."
sleep 10

# Extract job_id from the output of the previous curl command
JOB_ID=$(curl -s -X POST "http://localhost:8000/predict" \
         -H "Content-Type: application/json" \
         -d "{\"scan_id\": \"$SCAN_ID\"}" | jq -r '.job_id')

if [ -z "$JOB_ID" ] || [ "$JOB_ID" = "null" ]; then
  echo "Failed to get job_id"
  exit 1
fi

echo "Got job_id: $JOB_ID"
echo "Checking job status..."

curl "http://localhost:8000/status/$JOB_ID" | jq

echo "Waiting 60 seconds for job to finish..."
sleep 60

echo "Checking final status..."
curl "http://localhost:8000/status/$JOB_ID" | jq

echo "Test complete!"
