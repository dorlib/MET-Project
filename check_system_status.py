#!/usr/bin/env python3
"""
MET Project Status Report
This script checks the status of all services and ensures the system is properly set up.
"""

import requests
import os
import sys
import json
from datetime import datetime

# Define colors for terminal output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'
END = '\033[0m'

def print_header(text):
    """Print a formatted header"""
    print(f"\n{BOLD}{BLUE}{text}{END}")
    print(f"{BLUE}{'=' * len(text)}{END}")

def print_status(label, status, message=""):
    """Print a status message with color coding"""
    if status == "OK":
        status_color = f"{GREEN}✓ {status}{END}"
    elif status == "WARNING":
        status_color = f"{YELLOW}⚠ {status}{END}"
    else:
        status_color = f"{RED}✗ {status}{END}"
    
    label_width = 25
    status_width = 10
    
    print(f"{BOLD}{label:{label_width}}{END} {status_color:{status_width}} {message}")

def check_services():
    """Check all MET project services for health and connectivity"""
    print_header("MET Project Status Report")
    print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Check API Gateway
    api_up = False
    try:
        api_response = requests.get("http://localhost:8000/health", timeout=5)
        if api_response.status_code == 200:
            print_status("API Gateway", "OK", "Running and responding to requests")
            api_up = True
        else:
            print_status("API Gateway", "ERROR", f"Unhealthy response: {api_response.status_code}")
    except Exception as e:
        print_status("API Gateway", "ERROR", f"Connection error: {str(e)}")
    
    # Check Model Service (via API Gateway)
    if api_up:
        try:
            model_response = requests.get("http://localhost:8000/model/health", timeout=5)
            if model_response.status_code == 200:
                model_data = model_response.json()
                print_status("Model Service", "OK", f"Version: {model_data.get('version')} - Active Jobs: {model_data.get('active_jobs')}")
            else:
                print_status("Model Service", "ERROR", f"Unhealthy response: {model_response.status_code}")
        except Exception as e:
            print_status("Model Service", "ERROR", f"Connection error: {str(e)}")
    
    # Check Frontend
    try:
        frontend_response = requests.get("http://localhost:3000", timeout=5)
        if frontend_response.status_code == 200:
            print_status("Frontend", "OK", "Serving content")
        else:
            print_status("Frontend", "ERROR", f"Unhealthy response: {frontend_response.status_code}")
    except Exception as e:
        print_status("Frontend", "ERROR", f"Connection error: {str(e)}")
    
    # Check MySQL (this is just a connectivite check, not a full health check)
    try:
        mysql_response = os.system("docker exec -i met-project_mysql_1 mysqladmin -u root -ppassword ping >/dev/null 2>&1")
        if mysql_response == 0:
            print_status("MySQL Database", "OK", "Accepting connections")
        else:
            print_status("MySQL Database", "ERROR", "Not responding to connections")
    except Exception as e:
        print_status("MySQL Database", "ERROR", f"Connection error: {str(e)}")
    
    # Check for shared volumes
    print_header("Shared Volumes")
    
    try:
        uploads_response = os.system("docker exec -i met-project_model-service_1 ls /app/uploads >/dev/null 2>&1")
        if uploads_response == 0:
            print_status("Uploads Volume", "OK", "Properly mounted")
        else:
            print_status("Uploads Volume", "ERROR", "Not properly mounted")
    except Exception as e:
        print_status("Uploads Volume", "ERROR", f"Check failed: {str(e)}")
    
    try:
        results_response = os.system("docker exec -i met-project_model-service_1 ls /app/results >/dev/null 2>&1")
        if results_response == 0:
            # Count results files
            count_cmd = "docker exec -i met-project_model-service_1 find /app/results -name '*prediction.npy' | wc -l"
            result_count = os.popen(count_cmd).read().strip()
            print_status("Results Volume", "OK", f"Properly mounted, contains {result_count} prediction results")
        else:
            print_status("Results Volume", "ERROR", "Not properly mounted")
    except Exception as e:
        print_status("Results Volume", "ERROR", f"Check failed: {str(e)}")
    
    print_header("End-to-End Flow")
    # Check for recent successful uploads
    try:
        recent_files_cmd = "docker exec -i met-project_model-service_1 find /app/results -name '*_prediction.npy' -mmin -60 | wc -l"
        recent_count = os.popen(recent_files_cmd).read().strip()
        if int(recent_count) > 0:
            print_status("Recent Processing", "OK", f"{recent_count} files processed in the last hour")
        else:
            print_status("Recent Processing", "WARNING", "No files processed in the last hour")
    except Exception as e:
        print_status("Recent Processing", "ERROR", f"Check failed: {str(e)}")
    
    print("\n")
    print(f"{BLUE}To verify the upload and polling flow:{END}")
    print(f"  python test_upload_and_poll.py -v")
    
    print(f"\n{BLUE}To restart all services:{END}")
    print(f"  ./restart_services.sh")
    
    print(f"\n{BLUE}To restart just the model service:{END}")
    print(f"  ./restart_model_service.sh")

if __name__ == "__main__":
    check_services()
