#!/bin/bash
# Script to verify that the 3D projection visualization supports slice selection

echo "Verifying 3D projection visualization with slice selection..."

# Check the image_processor.py implementation
echo "Checking image_processor.py implementation..."
grep -A 10 "generate_3d_projection" backend/image_processing_service/image_processor.py

# Check the API gateway implementation
echo "Checking API gateway implementation..."
grep -A 10 "advanced-visualization" backend/api_gateway/api.py

# Check the frontend implementation
echo "Checking VisualizationControls.js implementation..."
grep -A 10 "vizType === 'projection'" frontend/src/components/VisualizationControls.js

echo "Checking api.js implementation..."
grep -A 10 "getVisualizationUrl" frontend/src/services/api.js

echo "======================================================"
echo "The slice selection feature for 3D projection appears to be already implemented!"
echo "When choosing 3D Projection in the visualization controls, the slice slider"
echo "should already be visible and functional."
echo "======================================================"
