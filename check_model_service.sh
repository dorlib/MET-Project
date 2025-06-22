#!/bin/bash
# check_model_service.sh - Test the model service health endpoint

echo "Checking model service health..."

# Call the health endpoint
RESPONSE=$(curl -s http://localhost:8000/model/health)

# Check if curl command succeeded
if [ $? -ne 0 ]; then
    echo "Error: Could not connect to model service"
    exit 1
fi

# Print the response
echo "Model service health check response:"
echo $RESPONSE

# Pretty print JSON if jq is available
if command -v jq &> /dev/null; then
    echo "Formatted response:"
    echo $RESPONSE | jq
fi

# Check if model service is using the mock model
if echo $RESPONSE | grep -q "mock"; then
    echo "Note: Model service is using the mock model implementation"
fi

echo ""
echo "To check the model service logs:"
echo "docker logs met-project_model-service_1"
