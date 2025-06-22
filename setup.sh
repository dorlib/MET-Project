#!/bin/bash
# setup.sh - Setup script for MET-project

set -e  # Exit on error

# Color definitions
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  Brain Metastasis Analysis System Setup${NC}"
echo -e "${BLUE}======================================${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

# Create necessary directories
echo -e "${YELLOW}Creating necessary directories...${NC}"
mkdir -p ./Data/saved_models
mkdir -p ./Data/uploads
mkdir -p ./Data/results

# Check for model file
MODEL_PATH="./Data/saved_models/brats_t1ce.pth"
if [[ ! -f "$MODEL_PATH" ]]; then
    echo -e "${YELLOW}Warning: Model file not found at $MODEL_PATH${NC}"
    echo -e "${YELLOW}You will need to place your trained model file there before running the services.${NC}"
fi

# Build the Docker images
echo -e "${YELLOW}Building Docker images...${NC}"
docker-compose build

echo -e "${GREEN}Setup completed successfully!${NC}"
echo -e "${YELLOW}To start the services, run:${NC}"
echo -e "${BLUE}docker-compose up${NC}"
echo -e "${YELLOW}Access the web interface at:${NC} http://localhost:3000"
