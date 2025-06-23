#!/bin/bash
# Script to test the 3D projection visualization with different slice indices

# Define constants
API_URL="http://localhost:8000"
AUTH_HEADER="Authorization: Bearer YOUR_TOKEN_HERE"

# Create a test folder if it doesn't exist
mkdir -p test_3d_projection_slices

# Function to generate a simple test job ID
generate_job_id() {
  echo "test-job-$(date +%s)"
}

# Get a job ID from previous test or generate a new one
JOB_ID=$(ls test_3d_projection_slices/*.png 2>/dev/null | head -n 1 | sed 's/.*\/\(.*\)_.*\.png/\1/' || generate_job_id)

echo "Using job ID: $JOB_ID"

# Test the 3D projection with different slice indices
test_slice_indices() {
  local slice_indices=("10" "50" "90")
  
  echo "Testing 3D projection visualization with different slice indices..."
  
  for slice_idx in "${slice_indices[@]}"; do
    echo "Testing slice index: $slice_idx"
    
    # Construct the API URL for 3D projection with the specified slice index
    local url="${API_URL}/advanced-visualization/${JOB_ID}?type=projection&slice_idx=${slice_idx}"
    local output_file="test_3d_projection_slices/${JOB_ID}_projection_slice_${slice_idx}.png"
    
    echo "Requesting: $url"
    echo "Saving to: $output_file"
    
    # Make the API request
    curl -s -H "${AUTH_HEADER}" "$url" > "$output_file"
    
    # Check if the file was created and is a valid image
    if file "$output_file" | grep -q "PNG image data"; then
      echo "Success! Generated 3D projection with slice index $slice_idx"
    else
      echo "Error: Failed to generate 3D projection with slice index $slice_idx"
      cat "$output_file"  # Display error message if any
    fi
    
    echo "--------------------"
  done
}

# Run the tests
test_slice_indices

echo "Test completed. Check the test_3d_projection_slices folder for the generated images."
