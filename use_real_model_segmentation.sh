#!/bin/bash
# Script to switch from mock model to real UNETR model for segmentation

echo "Stopping the current model-service container..."
docker-compose stop model-service

echo "Setting up environment to use real model..."
# Create .env file to configure model settings if it doesn't exist
if [ ! -f .env ]; then
  echo "Creating .env file..."
  touch .env
fi

# Update the .env file to use the real model
grep -v "MOCK_MODEL" .env > .env.tmp
echo "MOCK_MODEL=false" >> .env.tmp
mv .env.tmp .env

echo "Rebuilding model-service container with real UNETR model..."
docker-compose build model-service

echo "Starting model-service with real model..."
docker-compose up -d model-service

echo "Waiting for the model service to initialize..."
sleep 5

echo "Checking model-service container logs..."
docker-compose logs --tail=20 model-service

echo "Done! The system is now using real model-based segmentation."
echo "Check the logs to ensure the model loaded correctly."
