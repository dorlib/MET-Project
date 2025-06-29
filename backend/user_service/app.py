#!/usr/bin/env python3
# User Service for MET Brain Metastasis Segmentation Service

from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
import time
import traceback
import sys
from utils.database import init_db
from controllers.controllers import auth_bp, user_bp, scan_bp

# Configure logging - set to DEBUG level for more details
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)

# Configure Flask app
app.config['DEBUG'] = True
app.config['ENV'] = 'development'
app.config['PROPAGATE_EXCEPTIONS'] = True  # Make sure exceptions are propagated to see them

# Initialize database (with retry mechanism)
retry_count = 5
for i in range(retry_count):
    try:
        logger.info("Attempting to initialize database tables...")
        init_db()
        logger.info("Database tables initialized successfully")
        break
    except Exception as e:
        if i < retry_count - 1:
            wait_time = (i + 1) * 2
            logger.warning(f"Database initialization failed, retrying in {wait_time} seconds. Error: {str(e)}")
            time.sleep(wait_time)
        else:
            logger.error(f"Failed to initialize database after {retry_count} attempts: {str(e)}")
            raise

# Root health check endpoint
@app.route('/health', methods=['GET'])
def root_health():
    return jsonify({
        "status": "healthy",
        "service": "user-service"
    })

# Test endpoint
@app.route('/test', methods=['GET'])
def test_endpoint():
    return jsonify({
        "message": "This is a test endpoint"
    })

# Direct implementation of /scans endpoint
@app.route('/scans', methods=['GET', 'POST'])
def direct_scans():
    """Handle GET and POST requests for the /scans endpoint."""
    if request.method == 'POST':
        # Create a new scan
        data = request.json
        
        if not data or not all(k in data for k in ('job_id', 'file_name')):
            return jsonify({"error": "Job ID and file name are required"}), 400
        
        from services.services import ScanService
        from utils.database import get_db
        db = next(get_db())
        
        result = ScanService.create_scan(
            db,
            data['job_id'],
            data['file_name'],
            data.get('user_email')
        )
        
        if result["success"]:
            return jsonify(result["data"]), result["status_code"]
        else:
            return jsonify({"error": result["error"]}), result["status_code"]
    else:
        # GET method - Get all scans for the authenticated user
        # Check for authentication header
        auth_header = request.headers.get('Authorization')
        
        # If there's an auth header, try to verify it
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            from utils.jwt_util import verify_token
            
            payload = verify_token(token)
            if payload:
                # Get user ID from token
                user_id = payload.get('user_id')
                
                # Get pagination parameters
                try:
                    page = int(request.args.get('page', 1))
                    per_page = int(request.args.get('per_page', 10))
                except ValueError:
                    page = 1
                    per_page = 10
                
                # Get scans for the user
                from services.services import ScanService
                from utils.database import get_db
                db = next(get_db())
                
                try:
                    result = ScanService.get_user_scans(db, user_id, page, per_page)
                    
                    if result["success"]:
                        return jsonify(result["data"])
                    else:
                        return jsonify({"error": result["error"]}), 500
                except Exception as e:
                    return jsonify({"error": f"Error retrieving scans: {str(e)}"}), 500
        
        # Default response for unauthenticated requests
        return jsonify({
            "scans": [],
            "pagination": {
                "total_items": 0,
                "total_pages": 0, 
                "current_page": 1,
                "per_page": 10
            }
        })
    """Get all scans for the authenticated user with pagination."""
    # Check for authentication header
    auth_header = request.headers.get('Authorization')
    
    # If there's an auth header, try to verify it
    if auth_header and auth_header.startswith('Bearer '):
        token = auth_header.split(' ')[1]
        from utils.jwt_util import verify_token
        
        payload = verify_token(token)
        if payload:
            # Get user ID from token
            user_id = payload.get('user_id')
            
            # Get pagination parameters
            try:
                page = int(request.args.get('page', 1))
                per_page = int(request.args.get('per_page', 10))
            except ValueError:
                page = 1
                per_page = 10
            
            # Get scans for the user
            from services.services import ScanService
            from utils.database import get_db
            db = next(get_db())
            
            try:
                result = ScanService.get_user_scans(db, user_id, page, per_page)
                
                if result["success"]:
                    return jsonify(result["data"])
                else:
                    return jsonify({"error": result["error"]}), 500
            except Exception as e:
                return jsonify({"error": f"Error retrieving scans: {str(e)}"}), 500
    
    # Default response for unauthenticated requests
    return jsonify({
        "scans": [],
        "pagination": {
            "total_items": 0,
            "total_pages": 0, 
            "current_page": 1,
            "per_page": 10
        }
    })

# Direct implementation of /scans/filter endpoint
@app.route('/scans/filter', methods=['GET'])
def direct_filter_scans():
    """Filter scans for the authenticated user."""
    # Check for authentication header
    auth_header = request.headers.get('Authorization')
    
    # If there's an auth header, try to verify it
    if auth_header and auth_header.startswith('Bearer '):
        token = auth_header.split(' ')[1]
        from utils.jwt_util import verify_token
        
        payload = verify_token(token)
        if payload:
            # Get user ID from token
            user_id = payload.get('user_id')
            
            # Get pagination parameters
            try:
                page = int(request.args.get('page', 1))
                per_page = int(request.args.get('per_page', 10))
            except ValueError:
                page = 1
                per_page = 10
            
            # Extract filter parameters
            filters = {}
            filter_params = [
                'min_metastasis', 'max_metastasis', 
                'min_volume', 'max_volume',
                'start_date', 'end_date'
            ]
            
            for param in filter_params:
                if param in request.args:
                    filters[param] = request.args.get(param)
            
            # Get filtered scans for the user
            from services.services import ScanService
            from utils.database import get_db
            db = next(get_db())
            
            try:
                result = ScanService.filter_scans(db, user_id, filters, page, per_page)
                
                if result["success"]:
                    return jsonify(result["data"])
                else:
                    return jsonify({"error": result["error"]}), 500
            except Exception as e:
                return jsonify({"error": f"Error filtering scans: {str(e)}"}), 500
    
    # Default response for unauthenticated requests
    return jsonify({
        "scans": [],
        "pagination": {
            "total_items": 0,
            "total_pages": 0, 
            "current_page": 1,
            "per_page": 10
        }
    })

# Direct implementation of /user endpoint
@app.route('/user', methods=['GET'])
def direct_get_user():
    """Get the authenticated user's profile."""
    # Check for authentication header
    auth_header = request.headers.get('Authorization')
    
    # If there's an auth header, try to verify it
    if auth_header and auth_header.startswith('Bearer '):
        token = auth_header.split(' ')[1]
        from utils.jwt_util import verify_token
        
        payload = verify_token(token)
        if payload:
            # Get user by ID from token
            user_id = payload.get('user_id')
            from services.services import UserService
            from utils.database import get_db
            db = next(get_db())
            
            user_data = UserService.get_user_by_id(db, user_id)
            if user_data:
                return jsonify(user_data)
    
    # Return error for unauthenticated requests
    return jsonify({"error": "Authentication required"}), 401

# Direct implementation of /scans/<job_id> endpoint
@app.route('/scans/<job_id>', methods=['PUT'])
def direct_update_scan(job_id):
    """Update a scan record."""
    data = request.json
    
    from services.services import ScanService
    from utils.database import get_db
    db = next(get_db())
    
    result = ScanService.update_scan(
        db,
        job_id,
        data.get('status'),
        data.get('metastasis_count'),
        data.get('total_volume'),
        data.get('metastasis_volumes')
    )
    
    if result["success"]:
        return jsonify(result["data"]), result["status_code"]
    else:
        return jsonify({"error": result["error"]}), result["status_code"]

# Register blueprints
# No prefix is needed as the API gateway calls endpoints directly
app.register_blueprint(auth_bp)
app.register_blueprint(user_bp)
app.register_blueprint(scan_bp)

# Error handling
@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all exceptions during request processing."""
    logger.error(f"Unhandled exception: {str(e)}")
    logger.error(traceback.format_exc())
    
    # Always return a valid Flask response
    response = jsonify({"error": "An internal error occurred"})
    return response, 500

@app.before_request
def log_request():
    """Log details about each incoming request."""
    logger.debug(f"Received request: {request.method} {request.path}")
    logger.debug(f"Headers: {request.headers}")

@app.after_request
def log_response(response):
    """Log details about each outgoing response."""
    logger.debug(f"Response status: {response.status_code}")
    logger.debug(f"Response headers: {response.headers}")
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)
