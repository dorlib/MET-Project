#!/bin/bash
# Script to apply the visualization updates

echo "Applying visualization updates for 3D projection slice selection..."

echo "Copying updated image_processor.py to the image processing service container..."
docker cp /Users/dorliber/Library/CloudStorage/OneDrive-IntelCorporation/Desktop/private/MET-project/backend/image_processing_service/image_processor.py met-project_image-processing-service_1:/app/image_processor.py

echo "Restarting image processing service..."
docker-compose restart image-processing-service

echo "Waiting for service to restart..."
sleep 5

echo "Checking image processing service logs..."
docker-compose logs --tail=20 image-processing-service

echo "3D projection slice selection feature has been applied."
echo "You can now use the slice selector when in 3D Projection visualization mode."
