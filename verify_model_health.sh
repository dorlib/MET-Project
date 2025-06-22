#!/bin/bash

# Script to check model service health and restart if needed

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Checking model service health...${NC}"

# Check if model service is running
if ! docker ps | grep -q "met-project_model-service"; then
  echo -e "${RED}Model service is not running!${NC}"
  echo -e "${YELLOW}Starting model service...${NC}"
  docker-compose up -d model-service
  echo -e "${GREEN}Model service started.${NC}"
else
  # Check memory usage
  echo -e "${YELLOW}Checking model service memory usage...${NC}"
  MEMORY_USAGE=$(docker stats --no-stream --format "{{.MemUsage}}" met-project_model-service_1)
  echo -e "${GREEN}Memory usage: $MEMORY_USAGE${NC}"
  
  # Check if service is healthy
  echo -e "${YELLOW}Testing model service health endpoint...${NC}"
  HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5001/health)
  
  if [ "$HEALTH_RESPONSE" -eq 200 ]; then
    echo -e "${GREEN}Model service is healthy.${NC}"
  else
    echo -e "${RED}Model service is not responding properly. Status: $HEALTH_RESPONSE${NC}"
    echo -e "${YELLOW}Restarting model service...${NC}"
    docker-compose restart model-service
    echo -e "${GREEN}Model service restarted.${NC}"
  fi
fi

echo -e "${YELLOW}Checking for worker timeouts in logs...${NC}"
if docker logs --tail 50 met-project_model-service_1 | grep -q "WORKER TIMEOUT"; then
  echo -e "${RED}Worker timeouts detected. Service may be overloaded.${NC}"
  echo -e "${YELLOW}Restarting model service...${NC}"
  docker-compose restart model-service
  echo -e "${GREEN}Model service restarted.${NC}"
else
  echo -e "${GREEN}No worker timeouts detected.${NC}"
fi

echo -e "${GREEN}Done.${NC}"
