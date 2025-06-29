from flask import Blueprint, request, jsonify
import json
import logging
from sqlalchemy.orm import Session
from utils.database import get_db
from utils.jwt_util import verify_token
from services.services import UserService, ScanService

# Create blueprints
auth_bp = Blueprint('auth', __name__, url_prefix='')
user_bp = Blueprint('user', __name__, url_prefix='')
scan_bp = Blueprint('scan', __name__, url_prefix='')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper function to get current user from token
def get_current_user(db: Session):
    """
    Get the current user based on the Bearer token in the Authorization header.
    Returns a tuple (payload, user_data) if successful, or raises an exception if not.
    """
    auth_header = request.headers.get('Authorization')
    
    if not auth_header or not auth_header.startswith('Bearer '):
        raise ValueError("Authentication token is missing")
    
    token = auth_header.split(' ')[1]
    payload = verify_token(token)
    
    if not payload:
        raise ValueError("Invalid authentication token")
    
    user_id = payload.get('user_id')
    user_data = UserService.get_user_by_id(db, user_id)
    
    if not user_data:
        raise ValueError("User not found")
    
    return payload, user_data

# Auth endpoints
@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.json
    
    if not data or not all(k in data for k in ('email', 'name', 'password')):
        return jsonify({"error": "Name, email and password are required"}), 400
    
    db = next(get_db())
    result = UserService.register_user(db, data['email'], data['name'], data['password'])
    
    if result["success"]:
        return jsonify({
            "message": "User registered successfully",
            **result["data"]
        }), result["status_code"]
    else:
        return jsonify({"error": result["error"]}), result["status_code"]

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.json
    
    if not data or not all(k in data for k in ('email', 'password')):
        return jsonify({"error": "Email and password are required"}), 400
    
    db = next(get_db())
    result = UserService.login_user(db, data['email'], data['password'])
    
    if result["success"]:
        return jsonify({
            "message": "Login successful",
            **result["data"]
        }), result["status_code"]
    else:
        return jsonify({"error": result["error"]}), result["status_code"]

# User endpoints
@user_bp.route('/user', methods=['GET'])
def get_user():
    """Get the authenticated user's profile."""
    try:
        auth_header = request.headers.get('Authorization')
        
        # Check if auth header exists and has the correct format
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Authentication token is missing"}), 401
        
        # Extract and verify the token
        token = auth_header.split(' ')[1]
        payload = verify_token(token)
        
        if not payload:
            return jsonify({"error": "Invalid authentication token"}), 401
        
        # Get user by ID from the token
        db = next(get_db())
        user_id = payload.get('user_id')
        user_data = UserService.get_user_by_id(db, user_id)
        
        if not user_data:
            return jsonify({"error": "User not found"}), 404
        
        return jsonify(user_data), 200
    except Exception as e:
        logger.error(f"Error in get_user: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

# Scan endpoints
@scan_bp.route('/scans', methods=['POST'])
def create_scan():
    data = request.json
    db = next(get_db())
    
    if not data or not all(k in data for k in ('job_id', 'file_name')):
        return jsonify({"error": "Job ID and file name are required"}), 400
    
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

@scan_bp.route('/scans/<job_id>', methods=['PUT'])
def update_scan(job_id):
    data = request.json
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

@scan_bp.route('/scans', methods=['GET'])
def get_scans():
    """Get all scans for the authenticated user with pagination."""
    try:
        auth_header = request.headers.get('Authorization')
        
        # Check if auth header exists and has the correct format
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Authentication token is missing"}), 401
        
        # Extract and verify the token
        token = auth_header.split(' ')[1]
        payload = verify_token(token)
        
        if not payload:
            return jsonify({"error": "Invalid authentication token"}), 401
        
        # Get user by ID from the token
        db = next(get_db())
        user_id = payload.get('user_id')
        user_data = UserService.get_user_by_id(db, user_id)
        
        if not user_data:
            return jsonify({"error": "User not found"}), 404
        
        # Get pagination parameters
        try:
            page = int(request.args.get('page', 1))
            per_page = int(request.args.get('per_page', 10))
        except ValueError:
            return jsonify({"error": "Invalid pagination parameters"}), 400
        
        # Get scans for the user
        result = ScanService.get_user_scans(db, user_id, page, per_page)
        
        if result["success"]:
            return jsonify(result["data"]), result["status_code"]
        else:
            return jsonify({"error": result["error"]}), result["status_code"]
    except Exception as e:
        logger.error(f"Error in get_scans: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@scan_bp.route('/test', methods=['GET'])
def scan_test():
    """Test endpoint on the scan blueprint."""
    return jsonify({"message": "This is a test endpoint on the scan blueprint"}), 200

@scan_bp.route('/scans/filter', methods=['GET'])
def filter_scans():
    """Filter scans for the authenticated user."""
    try:
        auth_header = request.headers.get('Authorization')
        
        # Check if auth header exists and has the correct format
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Authentication token is missing"}), 401
        
        # Extract and verify the token
        token = auth_header.split(' ')[1]
        payload = verify_token(token)
        
        if not payload:
            return jsonify({"error": "Invalid authentication token"}), 401
        
        # Get user by ID from the token
        db = next(get_db())
        user_id = payload.get('user_id')
        user_data = UserService.get_user_by_id(db, user_id)
        
        if not user_data:
            return jsonify({"error": "User not found"}), 404
        
        # Get pagination parameters
        try:
            page = int(request.args.get('page', 1))
            per_page = int(request.args.get('per_page', 10))
        except ValueError:
            return jsonify({"error": "Invalid pagination parameters"}), 400
        
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
        
        # Filter scans for the user
        result = ScanService.filter_scans(db, user_id, filters, page, per_page)
        
        if result["success"]:
            return jsonify(result["data"]), result["status_code"]
        else:
            return jsonify({"error": result["error"]}), result["status_code"]
    except Exception as e:
        logger.error(f"Error in filter_scans: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

# Health check endpoint
@user_bp.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "service": "user-service"
    })
