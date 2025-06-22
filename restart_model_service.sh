#!/bin/bash
# restart_model_service.sh - Safely restart the model service container

echo "Restarting model service..."

# Check if docker-compose is available
if ! command -v docker compose &> /dev/null; then
    if ! command -v docker-compose &> /dev/null; then
        echo "Error: Neither docker compose nor docker-compose found"
        exit 1
    else
        DOCKER_COMPOSE="docker-compose"
    fi
else
    DOCKER_COMPOSE="docker compose"
fi

# Get current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Clean up any Docker orphans
echo "Removing any orphaned containers..."
$DOCKER_COMPOSE rm -f model-service

# Stop the model service gracefully
echo "Stopping model-service container..."
$DOCKER_COMPOSE stop model-service

# Wait a moment
sleep 2

# Build the model service with latest changes
echo "Building the model service..."
$DOCKER_COMPOSE build model-service

# Start the model service again
echo "Starting model-service container..."
$DOCKER_COMPOSE up -d model-service

# Check if it's running
echo "Checking model-service status..."
$DOCKER_COMPOSE ps model-service

echo "Model service restart completed"
echo "To check logs: docker compose logs -f model-service"

# Check for errors after restart
echo "Checking for errors:"
$DOCKER_COMPOSE logs model-service 2>&1 | grep -i error | tail -10
