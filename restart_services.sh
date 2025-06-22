#!/bin/bash
# Script to restart and monitor services

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Restarting Docker services...${NC}"

# Stop all services
echo -e "${YELLOW}Stopping services...${NC}"
docker-compose down
echo -e "${GREEN}Services stopped.${NC}"

# Clean up any unused volumes and networks
echo -e "${YELLOW}Cleaning up Docker resources...${NC}"
docker system prune -f

# Start services
echo -e "${YELLOW}Starting services...${NC}"
docker-compose up -d
echo -e "${GREEN}Services started.${NC}"

# Monitor logs for both model service and API gateway
echo -e "${YELLOW}Monitoring logs for model service and API gateway...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop monitoring...${NC}"

# Function to check if a service is running
check_service() {
  local service_name="$1"
  if docker ps | grep -q "$service_name"; then
    echo -e "${GREEN}$service_name is running.${NC}"
    return 0
  else
    echo -e "${RED}$service_name is not running!${NC}"
    return 1
  fi
}

# Check services
check_service "met-project_model-service_1"
check_service "met-project_api-gateway_1"
check_service "met-project_frontend_1"

# Open frontend in browser
echo -e "${YELLOW}Opening frontend in browser...${NC}"
open http://localhost:3000

# Show logs
docker-compose logs -f model-service api-gateway
