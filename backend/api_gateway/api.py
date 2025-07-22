#!/usr/bin/env python3
# api_gateway/api.py - API Gateway for MET Brain Metastasis Segmentation Service

from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import os
import requests
import json
import logging
import uuid
import numpy as np
from werkzeug.utils import secure_filename
from functools import wraps
import io

# Import preprocessing utilities
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocess_nifti_t1ce_for_model

app = Flask(__name__)

# Configure CORS with explicit settings
CORS(app, origins=['http://localhost:3000', 'http://127.0.0.1:3000'], 
     methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization'],
     supports_credentials=True)

logging.basicConfig(level=logging.INFO)

MODEL_SERVICE_URL = os.environ.get('MODEL_SERVICE_URL', 'http://model-service:5001')
IMAGE_PROCESSING_SERVICE_URL = os.environ.get('IMAGE_PROCESSING_SERVICE_URL', 'http://image-processing-service:5002')
USER_SERVICE_URL = os.environ.get('USER_SERVICE_URL', 'http://user-service:5003')
UPLOAD_FOLDER = '/app/uploads'
RESULTS_FOLDER = '/app/results'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Authentication decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get('Authorization')
        
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            
        if not token:
            return jsonify({'error': 'Authentication token is missing'}), 401
            
        # Verify token with user service
        try:
            response = requests.get(
                f"{USER_SERVICE_URL}/user",
                headers={"Authorization": f"Bearer {token}"}
            )
            
            if response.status_code != 200:
                return jsonify({'error': 'Invalid authentication token'}), 401
                
            # Add user to request context
            request.user = response.json()
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
        return f(*args, **kwargs)
    
    return decorated

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "api-gateway"})

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Endpoint to upload MRI scan (.npy format)
    Optional authentication to associate scan with user
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Get user email from auth header if available
    user_email = None
    auth_header = request.headers.get('Authorization')
    
    if auth_header and auth_header.startswith('Bearer '):
        token = auth_header.split(' ')[1]
        try:
            # Verify token with user service
            response = requests.get(
                f"{USER_SERVICE_URL}/user",
                headers={"Authorization": f"Bearer {token}"}
            )
            
            if response.status_code == 200:
                user_email = response.json().get('email')
        except Exception as e:
            # Continue without user association if auth fails
            logging.warning(f"Authentication failed: {str(e)}")
    
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    
    # Generate unique ID for this job and secure filename
    job_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    
    # Check file extension and handle accordingly - handle special case for .nii.gz
    if filename.lower().endswith('.nii.gz'):
        file_ext = '.nii.gz'
    else:
        file_ext = os.path.splitext(filename)[1].lower()
    
    # Log for debugging purposes
    logging.info(f"Uploaded file: {filename}, detected extension: {file_ext}")
    
    try:
        if file_ext in ['.nii', '.nii.gz']:
            
            temp_nifti_path = os.path.join(UPLOAD_FOLDER, f"temp_{job_id}_{filename}")
            npy_filename = f"{job_id}_converted.npy"
            file_path = os.path.join(UPLOAD_FOLDER, npy_filename)
            
            # Save the uploaded NIfTI file
            try:
                file.save(temp_nifti_path)
                if not os.path.exists(temp_nifti_path) or os.path.getsize(temp_nifti_path) == 0:
                    return jsonify({"error": "NIfTI file upload failed or empty file received"}), 400
            except Exception as e:
                logging.error(f"Error saving uploaded NIfTI file: {str(e)}")
                return jsonify({"error": f"Failed to save uploaded NIfTI file: {str(e)}"}), 500
            
            # Convert to NPY with preprocessing
            if not preprocess_nifti_t1ce_for_model(temp_nifti_path, file_path):
                return jsonify({"error": "Failed to convert NIfTI to NPY format"}), 500
                
            # Remove temporary NIfTI file
            try:
                os.remove(temp_nifti_path)
            except Exception as e:
                logging.warning(f"Could not remove temporary NIfTI file: {str(e)}")
                
            logging.info(f"Successfully converted NIfTI to NPY: {file_path}")
            
        elif file_ext == '.npy':
            # Handle NPY format directly
            file_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_{filename}")
            try:
                file.save(file_path)
            except Exception as e:
                logging.error(f"Error saving uploaded NPY file: {str(e)}")
                return jsonify({"error": f"Failed to save uploaded NPY file: {str(e)}"}), 500
        else:
            return jsonify({
                "error": "Invalid file format. Supported formats: NIfTI (.nii, .nii.gz) and NumPy (.npy)",
                "supported_formats": [".npy", ".nii", ".nii.gz"],
                "received_format": file_ext if file_ext else "unknown"
            }), 400
            
        # Validate file exists and has non-zero size
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return jsonify({"error": "File processing failed or empty file received"}), 400
        
        # Forward to model service for prediction
        try:
            response = requests.post(
                f"{MODEL_SERVICE_URL}/predict",
                json={"file_path": file_path, "job_id": job_id},
                timeout=10  # Short timeout since we just need to submit the job, not wait for completion
            )
        except requests.RequestException as e:
            logging.error(f"Error communicating with model service: {str(e)}")
            return jsonify({"error": "Failed to connect to model service"}), 503
        
        if response.status_code == 200:
            # Register scan with user service (with or without user association)
            try:
                scan_response = requests.post(
                    f"{USER_SERVICE_URL}/scans",
                    json={
                        "job_id": job_id,
                        "file_name": filename,
                        "user_email": user_email,
                        "status": "processing"
                    }
                )
                if scan_response.status_code >= 400:
                    logging.warning(f"User service returned error when registering scan: {scan_response.text}")
            except Exception as e:
                logging.error(f"Error registering scan with user service: {str(e)}")
            
            return jsonify({
                "message": "File uploaded successfully",
                "job_id": job_id,
                "status": "processing",
                "user_associated": user_email is not None
            })
        else:
            error_msg = "Unknown error" 
            try:
                error_msg = response.json().get('error', response.text)
            except:
                error_msg = response.text
                
            return jsonify({
                "error": "Model service error", 
                "details": error_msg
            }), response.status_code
    except Exception as e:
        logging.exception(f"Unexpected error during file upload: {str(e)}")
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route('/results/<job_id>', methods=['GET'])
def get_results(job_id):
    """
    Endpoint to get segmentation results and metastasis analysis
    """
    # Check if the model prediction is complete
    model_status_response = requests.get(f"{MODEL_SERVICE_URL}/status/{job_id}")
    
    # If model service knows about the job, check its status
    if model_status_response.status_code == 200:
        model_status = model_status_response.json()
        
        if model_status.get("status") != "completed":
            return jsonify({
                "job_id": job_id,
                "status": model_status.get("status", "unknown"),
                "message": "Segmentation still processing"
            })
    
    # If model service doesn't know about the job or job is completed, 
    # check if results exist in image processing service
    analysis_response = requests.get(
        f"{IMAGE_PROCESSING_SERVICE_URL}/analyze/{job_id}"
    )
    
    if analysis_response.status_code != 200:
        # Neither service knows about this job or has results
        if model_status_response.status_code != 200:
            return jsonify({"error": "Invalid job ID or results not found"}), 404
        else:
            return jsonify({
                "job_id": job_id,
                "status": "segmentation_complete", 
                "message": "Segmentation complete, analysis pending or failed"
            }), 202
    
    # Get analysis data
    analysis_data = analysis_response.json()
    
    # Update scan information in user service
    try:
        requests.put(
            f"{USER_SERVICE_URL}/scans/{job_id}",
            json={
                "status": "completed",
                "metastasis_count": analysis_data.get("metastasis_count"),
                "total_volume": analysis_data.get("total_volume"),
                "metastasis_volumes": analysis_data.get("metastasis_volumes")
            }
        )
    except Exception as e:
        logging.error(f"Error updating scan in user service: {str(e)}")
    
    # Return full results including segmentation visualization and metastasis analysis
    return jsonify({
        "job_id": job_id,
        "status": "completed",
        "segmentation_path": f"/visualization/{job_id}",
        "prediction_download_url": f"/download/prediction/{job_id}",
        "metastasis_count": analysis_data.get("metastasis_count"),
        "metastasis_volumes": analysis_data.get("metastasis_volumes"),
        "total_volume": analysis_data.get("total_volume"),
        "confidence_metrics": analysis_data.get("confidence_metrics"),
    })

@app.route('/download/prediction/<job_id>', methods=['GET'])
def download_prediction(job_id):
    """
    Endpoint to download the raw prediction/mask file (.npy format)
    """
    # Validate job_id format to prevent path traversal
    if '/' in job_id or '\\' in job_id or '..' in job_id:
        return jsonify({"error": "Invalid job ID format"}), 400
    
    try:
        # First, try to get the file directly from the image processing service
        # This approach is more robust as it doesn't depend on model service state
        download_response = requests.get(
            f"{IMAGE_PROCESSING_SERVICE_URL}/download/prediction/{job_id}",
            stream=True  # Stream the file to avoid loading it all into memory
        )
        
        if download_response.status_code == 200:
            # File exists and can be downloaded
            
            # Create a streaming response to forward the file
            def generate():
                for chunk in download_response.iter_content(chunk_size=8192):
                    if chunk:
                        yield chunk
            
            logging.info(f"Prediction file download initiated for job {job_id}")
            
            # Get content type and filename from the downstream response
            content_type = download_response.headers.get('Content-Type', 'application/octet-stream')
            content_disposition = download_response.headers.get('Content-Disposition', f'attachment; filename="{job_id}_prediction.npy"')
            
            return Response(
                generate(),
                content_type=content_type,
                headers={
                    'Content-Disposition': content_disposition,
                    'Content-Length': download_response.headers.get('Content-Length', '')
                }
            )
        elif download_response.status_code == 404:
            # File doesn't exist - check if we should give more specific error info
            
            # Optional: Check model service status for more detailed error message
            try:
                model_status_response = requests.get(f"{MODEL_SERVICE_URL}/status/{job_id}", timeout=5)
                
                if model_status_response.status_code == 200:
                    model_status = model_status_response.json()
                    if model_status.get("status") != "completed":
                        return jsonify({
                            "error": "Prediction not ready",
                            "status": model_status.get("status", "unknown"),
                            "message": "Model prediction still processing"
                        }), 202
                
                return jsonify({"error": "Prediction file not found"}), 404
                        
            except requests.exceptions.RequestException:
                # Model service is not available, but that's okay for download
                return jsonify({"error": "Prediction file not found"}), 404
        else:
            # Other error from image processing service
            logging.error(f"Image processing service returned error {download_response.status_code}: {download_response.text}")
            return jsonify({
                "error": "Failed to retrieve prediction file",
                "status_code": download_response.status_code
            }), download_response.status_code
        
    except Exception as e:
        logging.error(f"Error downloading prediction file: {str(e)}")
        return jsonify({"error": f"Download failed: {str(e)}"}), 500

@app.route('/advanced-analysis/<job_id>', methods=['GET'])
def get_advanced_analysis(job_id):
    """
    Endpoint to get advanced multi-class tissue analysis
    """
    # Validate JWT token first
    token = request.headers.get('Authorization')
    if not token:
        return jsonify({"error": "Authorization token required"}), 401
    
    # Check if the job_id is valid
    model_status_response = requests.get(f"{MODEL_SERVICE_URL}/status/{job_id}")
    if model_status_response.status_code != 200:
        return jsonify({"error": "Invalid job ID"}), 404
        
    model_status = model_status_response.json()
    if model_status.get("status") != "completed":
        return jsonify({
            "job_id": job_id,
            "status": model_status.get("status", "unknown"),
            "message": "Segmentation still processing"
        })
    
    # Request advanced analysis from image processing service
    analysis_response = requests.get(
        f"{IMAGE_PROCESSING_SERVICE_URL}/advanced-analysis/{job_id}"
    )
    
    if analysis_response.status_code != 200:
        return jsonify({
            "error": "Advanced analysis failed",
            "details": analysis_response.json() if analysis_response.content else "No details available"
        }), analysis_response.status_code
    
    # Return the analysis data
    return jsonify(analysis_response.json())

@app.route('/visualization/<job_id>', methods=['GET'])
def get_basic_visualization(job_id):
    """
    Endpoint to get basic segmentation visualization image
    """
    # Validate job_id format to prevent path traversal
    if '/' in job_id or '\\' in job_id or '..' in job_id:
        return jsonify({"error": "Invalid job ID format"}), 400
    
    try:
        # Log the request
        logging.info(f"Basic visualization request for job {job_id}")
        
        # Forward the request to the image processing service
        viz_response = requests.get(
            f"{IMAGE_PROCESSING_SERVICE_URL}/visualization/{job_id}",
            params=request.args
        )
        
        if viz_response.status_code != 200:
            # Log the error response from the image processing service
            logging.error(f"Image processing service returned error {viz_response.status_code}: {viz_response.text}")
            return jsonify({
                "error": "Visualization generation failed",
                "status_code": viz_response.status_code,
                "details": viz_response.text
            }), viz_response.status_code
        
        # Return the image directly
        logging.info(f"Visualization successfully generated for job {job_id}")
        return Response(
            viz_response.content,
            mimetype=viz_response.headers['Content-Type'],
            headers={
                'Content-Disposition': viz_response.headers.get('Content-Disposition', f'inline; filename="{job_id}_visualization.png"')
            }
        )
    except Exception as e:
        logging.error(f"Error generating visualization for job {job_id}: {str(e)}")
        return jsonify({"error": f"Visualization failed: {str(e)}"}), 500

@app.route('/advanced-visualization/<job_id>', methods=['GET'])
def get_advanced_visualization(job_id):
    """
    Endpoint to get advanced visualizations of segmentation results
    
    Query parameters:
    - type: Type of visualization (slice, projection, multi-slice, lesions)
    - quality: Quality level (standard, high)
    - slice_idx: Optional slice index for slice visualizations
    - num_slices: Number of slices for multi-slice visualization
    - upscale: Upscaling factor for high-res visualizations
    - enhance_contrast: Whether to enhance contrast
    - enhance_edges: Whether to enhance edges
    """
    # Validate job_id format
    if '/' in job_id or '\\' in job_id or '..' in job_id:
        return jsonify({"error": "Invalid job ID format"}), 400
        
    # Log the visualization request parameters
    logging.info(f"Advanced visualization request for job {job_id} with params: {request.args}")
    logging.info(f"Request slice_idx: {request.args.get('slice_idx')}")
    
    try:
        # Forward the visualization request with all query parameters to the image processing service
        viz_response = requests.get(
            f"{IMAGE_PROCESSING_SERVICE_URL}/visualization/{job_id}",
            params=request.args
        )
        
        if viz_response.status_code != 200:
            # Log the error response
            logging.error(f"Image processing service returned error {viz_response.status_code}: {viz_response.text}")
            return jsonify({
                "error": "Visualization generation failed",
                "status_code": viz_response.status_code,
                "details": viz_response.text
            }), viz_response.status_code
        
        # Return the image directly with appropriate headers
        logging.info(f"Advanced visualization successfully generated for job {job_id}")
        return Response(
            viz_response.content,
            mimetype=viz_response.headers['Content-Type'],
            headers={
                'Content-Disposition': viz_response.headers.get('Content-Disposition', f'inline; filename="{job_id}_visualization.png"')
            }
        )
    except Exception as e:
        logging.error(f"Error requesting visualization from image processing service: {str(e)}")
        return jsonify({"error": f"Visualization request failed: {str(e)}"}), 500

@app.route('/segmentation-data/<job_id>', methods=['GET'])
def get_segmentation_data(job_id):
    """
    Endpoint to get raw segmentation data for 3D visualization
    
    Query parameters:
    - downsample: Downsampling factor (1-4, default: 2)
    - max_voxels: Maximum number of voxels to return (default: 50000)
    """
    # Validate job_id format
    if '/' in job_id or '\\' in job_id or '..' in job_id:
        return jsonify({"error": "Invalid job ID format"}), 400
    
    # Check if the job exists and is completed (similar logic to results endpoint)
    model_status_response = requests.get(f"{MODEL_SERVICE_URL}/status/{job_id}")
    
    # If model service knows about the job, check its status
    if model_status_response.status_code == 200:
        model_status = model_status_response.json()
        if model_status.get("status") != "completed":
            return jsonify({
                "error": "Segmentation not ready",
                "status": model_status.get("status", "unknown")
            }), 202
    
    # Check if segmentation results exist in image processing service
    # (even if model service doesn't know about the job)
    
    try:
        # Forward the request to the image processing service
        response = requests.get(
            f"{IMAGE_PROCESSING_SERVICE_URL}/segmentation-data/{job_id}",
            params=request.args
        )
        
        if response.status_code != 200:
            logging.error(f"Image processing service returned error {response.status_code}: {response.text}")
            
            # If neither service knows about this job, return proper error
            if model_status_response.status_code != 200 and response.status_code == 404:
                return jsonify({"error": "Invalid job ID or results not found"}), 404
                
            return jsonify({
                "error": "Failed to get segmentation data",
                "status_code": response.status_code
            }), response.status_code
        
        logging.info(f"Segmentation data successfully retrieved for job {job_id}")
        return response.json()
        
    except Exception as e:
        logging.error(f"Error requesting segmentation data: {str(e)}")
        return jsonify({"error": f"Request failed: {str(e)}"}), 500

@app.route('/api/volume-dimensions/<job_id>', methods=['GET'])
def get_volume_dimensions(job_id):
    """
    Endpoint to get volume dimensions for slice navigation
    
    Returns dimensions [width, height, depth] and optional spacing information
    """
    # Validate job_id format
    if '/' in job_id or '\\' in job_id or '..' in job_id:
        return jsonify({"error": "Invalid job ID format"}), 400
    
    # Check if the job exists and is completed (similar logic to other endpoints)
    model_status_response = requests.get(f"{MODEL_SERVICE_URL}/status/{job_id}")
    
    # If model service knows about the job, check its status
    if model_status_response.status_code == 200:
        model_status = model_status_response.json()
        if model_status.get("status") != "completed":
            return jsonify({
                "error": "Volume data not ready",
                "status": model_status.get("status", "unknown")
            }), 202
    
    try:
        # Forward the request to the image processing service
        response = requests.get(f"{IMAGE_PROCESSING_SERVICE_URL}/volume-dimensions/{job_id}")
        
        if response.status_code != 200:
            logging.error(f"Image processing service returned error {response.status_code}: {response.text}")
            
            # If neither service knows about this job, return proper error
            if model_status_response.status_code != 200 and response.status_code == 404:
                return jsonify({"error": "Invalid job ID or results not found"}), 404
                
            return jsonify({
                "error": "Failed to get volume dimensions",
                "status_code": response.status_code
            }), response.status_code
        
        logging.info(f"Volume dimensions successfully retrieved for job {job_id}")
        return response.json()
        
    except Exception as e:
        logging.error(f"Error requesting volume dimensions: {str(e)}")
        return jsonify({"error": f"Request failed: {str(e)}"}), 500

@app.route('/lesion-analysis/<job_id>', methods=['GET'])
def get_lesion_analysis(job_id):
    """
    Endpoint to get detailed lesion analysis
    
    Query parameters:
    - class_id: Optional specific class ID to analyze
    """
    # Validate JWT token first
    token = request.headers.get('Authorization')
    if not token:
        return jsonify({"error": "Authorization token required"}), 401
    
    # Check if the job_id is valid
    model_status_response = requests.get(f"{MODEL_SERVICE_URL}/status/{job_id}")
    if model_status_response.status_code != 200:
        return jsonify({"error": "Invalid job ID"}), 404
        
    model_status = model_status_response.json()
    if model_status.get("status") != "completed":
        return jsonify({
            "job_id": job_id,
            "status": model_status.get("status", "unknown"),
            "message": "Segmentation still processing"
        })
    
    # Forward the lesion analysis request with query parameters
    analysis_response = requests.get(
        f"{IMAGE_PROCESSING_SERVICE_URL}/lesion-analysis/{job_id}",
        params=request.args
    )
    
    if analysis_response.status_code != 200:
        return jsonify({
            "error": "Lesion analysis failed",
            "details": analysis_response.json() if analysis_response.content else "No details available"
        }), analysis_response.status_code
    
    # Return the analysis data
    return jsonify(analysis_response.json())

@app.route('/slice-summary/<job_id>', methods=['GET'])
def get_slice_summary(job_id):
    """
    Endpoint to get a summary of class distribution across slices
    
    Query parameters:
    - with_graph: Optional boolean to include distribution graph
    """
    # Validate JWT token first
    token = request.headers.get('Authorization')
    if not token:
        return jsonify({"error": "Authorization token required"}), 401
    
    # Check if the job_id is valid
    model_status_response = requests.get(f"{MODEL_SERVICE_URL}/status/{job_id}")
    if model_status_response.status_code != 200:
        return jsonify({"error": "Invalid job ID"}), 404
        
    model_status = model_status_response.json()
    if model_status.get("status") != "completed":
        return jsonify({
            "job_id": job_id,
            "status": model_status.get("status", "unknown"),
            "message": "Segmentation still processing"
        })
    
    # Forward the slice summary request with query parameters
    summary_response = requests.get(
        f"{IMAGE_PROCESSING_SERVICE_URL}/slice-summary/{job_id}",
        params=request.args
    )
    
    if summary_response.status_code != 200:
        return jsonify({
            "error": "Slice summary generation failed",
            "details": summary_response.json() if summary_response.content else "No details available"
        }), summary_response.status_code
    
    # Return the summary data
    return jsonify(summary_response.json())

@app.route('/analysis-metadata', methods=['POST'])
def set_metadata():
    """
    Endpoint to set metadata for analysis calculation (like voxel size)
    """
    # Validate JWT token first
    token = request.headers.get('Authorization')
    if not token:
        return jsonify({"error": "Authorization token required"}), 401
    
    # Forward the metadata request with body to the image processing service
    metadata_response = requests.post(
        f"{IMAGE_PROCESSING_SERVICE_URL}/metadata",
        json=request.json
    )
    
    if metadata_response.status_code != 200:
        return jsonify({
            "error": "Metadata update failed",
            "details": metadata_response.json() if metadata_response.content else "No details available"
        }), metadata_response.status_code
    
    # Return the updated metadata
    return jsonify(metadata_response.json())

# User authentication endpoints
@app.route('/auth/register', methods=['POST'])
def register():
    """
    Register a new user
    """
    try:
        response = requests.post(
            f"{USER_SERVICE_URL}/register",
            json=request.json
        )
        return response.json(), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/auth/login', methods=['POST'])
def login():
    """
    Login a user
    """
    try:
        response = requests.post(
            f"{USER_SERVICE_URL}/login",
            json=request.json
        )
        return response.json(), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/user/profile', methods=['GET'])
@token_required
def get_profile():
    """
    Get authenticated user profile
    """
    # User data is already attached to request by token_required decorator
    return jsonify(request.user)

@app.route('/auth/logout', methods=['POST'])
@token_required
def logout():
    """
    Logout a user - invalidate their token
    In a production environment, this should add the token to a blacklist
    """
    # In a real implementation, you would add the token to a blacklist/revocation list
    # For now, we'll just return success as the frontend handles token removal
    return jsonify({
        "message": "Successfully logged out"
    })

@app.route('/user/scans', methods=['GET'])
@token_required
def get_user_scans():
    """
    Get authenticated user's scan history with pagination support
    
    Query parameters:
    - page: Page number (default: 1)
    - per_page: Items per page (default: 10)
    """
    try:
        # Forward to user service with token
        auth_header = request.headers.get('Authorization')
        
        # Forward pagination parameters if provided
        params = {}
        if 'page' in request.args:
            params['page'] = request.args.get('page')
        if 'per_page' in request.args:
            params['per_page'] = request.args.get('per_page')
            
        response = requests.get(
            f"{USER_SERVICE_URL}/scans",
            headers={"Authorization": auth_header},
            params=params
        )
        return response.json(), response.status_code
    except Exception as e:
        logging.error(f"Error fetching user scans: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/user/scans/<job_id>', methods=['DELETE'])
@token_required
def delete_user_scan(job_id):
    """
    Delete a specific scan for the authenticated user
    """
    try:
        # Forward to user service with token
        auth_header = request.headers.get('Authorization')
        
        response = requests.delete(
            f"{USER_SERVICE_URL}/scans/{job_id}",
            headers={"Authorization": auth_header}
        )
        return response.json(), response.status_code
    except Exception as e:
        logging.error(f"Error deleting scan: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/user/settings', methods=['GET'])
@token_required
def get_user_settings():
    """
    Get authenticated user's settings including 2FA status
    """
    try:
        # Forward to user service with token
        auth_header = request.headers.get('Authorization')
        
        response = requests.get(
            f"{USER_SERVICE_URL}/user/settings",
            headers={"Authorization": auth_header}
        )
        return response.json(), response.status_code
    except Exception as e:
        logging.error(f"Error fetching user settings: {str(e)}")
        return jsonify({"error": str(e)}), 500

# 2FA endpoints have been removed

@app.route('/export/csv/<job_id>', methods=['GET'])
@token_required
def export_csv(job_id):
    """
    Export scan results as CSV
    """
    try:
        # First, get result data from the analyze endpoint
        response = requests.get(f"{IMAGE_PROCESSING_SERVICE_URL}/analyze/{job_id}")
        
        if response.status_code != 200:
            return jsonify({"error": "Result not found or not processed yet"}), 404
            
        result_data = response.json()
        
        # Create CSV content
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Brain Metastasis Analysis Results'])
        writer.writerow(['Job ID', job_id])
        writer.writerow(['Total Metastasis Count', result_data.get('metastasis_count', 0)])
        writer.writerow(['Total Volume (mm³)', result_data.get('total_volume', 0)])
        writer.writerow([])  # Empty row
        
        # Write individual metastases data
        writer.writerow(['Metastasis #', 'Volume (mm³)', '% of Total'])
        total_volume = result_data.get('total_volume', 0)
        
        for i, volume in enumerate(result_data.get('metastasis_volumes', [])):
            percentage = (volume / total_volume * 100) if total_volume > 0 else 0
            writer.writerow([i + 1, round(volume, 2), f"{round(percentage, 1)}%"])
        
        # Create response
        output.seek(0)
        
        return output.getvalue(), 200, {
            'Content-Type': 'text/csv',
            'Content-Disposition': f'attachment; filename=metastasis_results_{job_id}.csv'
        }
    except Exception as e:
        logging.error(f"Error exporting CSV: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/export/pdf/<job_id>', methods=['GET'])
@token_required
def export_pdf(job_id):
    """
    Export scan results as PDF
    """
    try:
        # First, get result data from the analyze endpoint
        response = requests.get(f"{IMAGE_PROCESSING_SERVICE_URL}/analyze/{job_id}")
        
        if response.status_code != 200:
            return jsonify({"error": "Result not found or not processed yet"}), 404
            
        result_data = response.json()
        
        # Create PDF content using ReportLab
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        import io
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        
        elements = []
        
        # Title
        title = Paragraph("Brain Metastasis Analysis Results", styles['Title'])
        elements.append(title)
        elements.append(Spacer(1, 12))
        
        # Summary data
        summary_data = [
            ["Job ID:", job_id],
            ["Total Metastasis Count:", str(result_data.get('metastasis_count', 0))],
            ["Total Volume (mm³):", str(round(result_data.get('total_volume', 0), 2))]
        ]
        
        summary_table = Table(summary_data, colWidths=[120, 300])
        summary_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ]))
        
        elements.append(summary_table)
        elements.append(Spacer(1, 24))
        
        # Metastasis data
        metastasis_title = Paragraph("Individual Metastasis Analysis", styles['Heading2'])
        elements.append(metastasis_title)
        elements.append(Spacer(1, 12))
        
        metastasis_data = [["Metastasis #", "Volume (mm³)", "% of Total"]]
        total_volume = result_data.get('total_volume', 0)
        
        for i, volume in enumerate(result_data.get('metastasis_volumes', [])):
            percentage = (volume / total_volume * 100) if total_volume > 0 else 0
            metastasis_data.append([i + 1, round(volume, 2), f"{round(percentage, 1)}%"])
        
        metastasis_table = Table(metastasis_data, colWidths=[100, 100, 100])
        metastasis_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('ALIGN', (1, 1), (2, -1), 'RIGHT'),
        ]))
        
        elements.append(metastasis_table)
        
        # Build PDF
        doc.build(elements)
        
        # Create response
        buffer.seek(0)
        
        return buffer.getvalue(), 200, {
            'Content-Type': 'application/pdf',
            'Content-Disposition': f'attachment; filename=metastasis_results_{job_id}.pdf'
        }
    except Exception as e:
        logging.error(f"Error exporting PDF: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/share/<job_id>', methods=['POST'])
@token_required
def create_share_link(job_id):
    """
    Create a shareable link for scan results
    
    Request body:
    - expires_in: Expiration time in hours (default: 24, max: 168 for 7 days)
    - allow_download: Whether to allow PDF/CSV downloads (default: false)
    - include_detailed_analysis: Whether to include detailed analysis (default: false)
    """
    try:
        # Validate that the user owns this scan or has access to it
        auth_header = request.headers.get('Authorization')
        
        # Check if scan exists and user has access
        response = requests.get(
            f"{USER_SERVICE_URL}/scans",
            headers={"Authorization": auth_header},
            params={"job_id": job_id}
        )
        
        if response.status_code != 200:
            return jsonify({"error": "Scan not found or access denied"}), 404
            
        scan_data = response.json()
        user_scans = scan_data.get('scans', [])
        
        # Check if job_id exists in user's scans
        if not any(scan.get('job_id') == job_id for scan in user_scans):
            return jsonify({"error": "Scan not found or access denied"}), 404
        
        # Get request parameters
        data = request.get_json() or {}
        expires_in = min(int(data.get('expires_in', 24)), 168)  # Max 7 days
        allow_download = bool(data.get('allow_download', False))
        include_detailed = bool(data.get('include_detailed_analysis', False))
        
        # Generate unique share token
        import secrets
        import time
        
        share_token = secrets.token_urlsafe(32)
        expires_at = int(time.time()) + (expires_in * 3600)  # Convert hours to seconds
        
        # Store share information (you could use a database, Redis, or file storage)
        # For now, I'll use a simple approach with user service
        share_data = {
            "share_token": share_token,
            "job_id": job_id,
            "created_by": request.user.get('user_id'),
            "expires_at": expires_at,
            "allow_download": allow_download,
            "include_detailed_analysis": include_detailed,
            "created_at": int(time.time())
        }
        
        # Store the share link in user service
        try:
            store_response = requests.post(
                f"{USER_SERVICE_URL}/shares",
                json=share_data,
                headers={"Authorization": auth_header}
            )
            
            if store_response.status_code != 200:
                logging.warning(f"Failed to store share data: {store_response.text}")
        except Exception as e:
            logging.warning(f"Could not store share data: {str(e)}")
        
        # Create shareable URL
        base_url = request.host_url.rstrip('/')
        share_url = f"{base_url}/shared/{share_token}"
        
        return jsonify({
            "share_url": share_url,
            "share_token": share_token,
            "expires_at": expires_at,
            "expires_in_hours": expires_in,
            "allow_download": allow_download,
            "include_detailed_analysis": include_detailed,
            "message": f"Shareable link created. Expires in {expires_in} hours."
        }), 200
        
    except Exception as e:
        logging.error(f"Error creating share link: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/shared/<share_token>', methods=['GET'])
def get_shared_results(share_token):
    """
    Get shared scan results without authentication
    """
    try:
        # Retrieve share information
        # For now, try to get it from user service
        try:
            share_response = requests.get(f"{USER_SERVICE_URL}/shares/{share_token}")
            
            if share_response.status_code != 200:
                return jsonify({"error": "Share link not found or expired"}), 404
                
            share_data = share_response.json()
        except Exception as e:
            return jsonify({"error": "Invalid share link"}), 404
        
        # Check if share link has expired
        import time
        current_time = int(time.time())
        
        if current_time > share_data.get('expires_at', 0):
            return jsonify({"error": "Share link has expired"}), 410
        
        job_id = share_data.get('job_id')
        if not job_id:
            return jsonify({"error": "Invalid share data"}), 400
        
        # Get the analysis results
        analysis_response = requests.get(f"{IMAGE_PROCESSING_SERVICE_URL}/analyze/{job_id}")
        
        if analysis_response.status_code != 200:
            return jsonify({"error": "Results not available"}), 404
            
        analysis_data = analysis_response.json()
        
        # Build response based on share permissions
        shared_results = {
            "job_id": job_id,
            "shared_at": share_data.get('created_at'),
            "expires_at": share_data.get('expires_at'),
            "metastasis_count": analysis_data.get("metastasis_count"),
            "total_volume": analysis_data.get("total_volume"),
            "metastasis_volumes": analysis_data.get("metastasis_volumes"),
            "segmentation_path": f"/shared/{share_token}/visualization",
            "permissions": {
                "allow_download": share_data.get('allow_download', False),
                "include_detailed_analysis": share_data.get('include_detailed_analysis', False)
            }
        }
        
        # Add detailed analysis if permitted
        if share_data.get('include_detailed_analysis', False):
            shared_results["confidence_metrics"] = analysis_data.get("confidence_metrics")
            shared_results["detailed_analysis_available"] = True
        
        return jsonify(shared_results), 200
        
    except Exception as e:
        logging.error(f"Error retrieving shared results: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/shared/<share_token>/visualization', methods=['GET'])
def get_shared_visualization(share_token):
    """
    Get visualization for shared results
    """
    try:
        # Verify share token and get job_id
        try:
            share_response = requests.get(f"{USER_SERVICE_URL}/shares/{share_token}")
            
            if share_response.status_code != 200:
                return jsonify({"error": "Share link not found or expired"}), 404
                
            share_data = share_response.json()
        except Exception as e:
            return jsonify({"error": "Invalid share link"}), 404
        
        # Check expiration
        import time
        if int(time.time()) > share_data.get('expires_at', 0):
            return jsonify({"error": "Share link has expired"}), 410
        
        job_id = share_data.get('job_id')
        if not job_id:
            return jsonify({"error": "Invalid share data"}), 400
        
        # Forward visualization request to image processing service
        viz_response = requests.get(
            f"{IMAGE_PROCESSING_SERVICE_URL}/visualization/{job_id}",
            params=request.args
        )
        
        if viz_response.status_code != 200:
            return jsonify({"error": "Visualization not available"}), 404
        
        # Return the visualization
        return Response(
            viz_response.content,
            mimetype=viz_response.headers['Content-Type'],
            headers={
                'Content-Disposition': viz_response.headers.get('Content-Disposition', f'inline; filename="shared_{job_id}_visualization.png"')
            }
        )
        
    except Exception as e:
        logging.error(f"Error retrieving shared visualization: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/shared/<share_token>/download/<format>', methods=['GET'])
def download_shared_results(share_token, format):
    """
    Download shared results in PDF or CSV format
    
    Args:
        share_token: The share token
        format: 'pdf' or 'csv'
    """
    try:
        # Verify share token and permissions
        try:
            share_response = requests.get(f"{USER_SERVICE_URL}/shares/{share_token}")
            
            if share_response.status_code != 200:
                return jsonify({"error": "Share link not found or expired"}), 404
                
            share_data = share_response.json()
        except Exception as e:
            return jsonify({"error": "Invalid share link"}), 404
        
        # Check expiration
        import time
        if int(time.time()) > share_data.get('expires_at', 0):
            return jsonify({"error": "Share link has expired"}), 410
        
        # Check download permission
        if not share_data.get('allow_download', False):
            return jsonify({"error": "Downloads not allowed for this share"}), 403
        
        job_id = share_data.get('job_id')
        if not job_id:
            return jsonify({"error": "Invalid share data"}), 400
        
        if format not in ['pdf', 'csv']:
            return jsonify({"error": "Invalid format. Use 'pdf' or 'csv'"}), 400
        
        # Get the analysis data
        analysis_response = requests.get(f"{IMAGE_PROCESSING_SERVICE_URL}/analyze/{job_id}")
        
        if analysis_response.status_code != 200:
            return jsonify({"error": "Results not available"}), 404
            
        result_data = analysis_response.json()
        
        if format == 'csv':
            # Generate CSV
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            writer.writerow(['Brain Metastasis Analysis Results (Shared)'])
            writer.writerow(['Job ID', job_id])
            writer.writerow(['Total Metastasis Count', result_data.get('metastasis_count', 0)])
            writer.writerow(['Total Volume (mm³)', result_data.get('total_volume', 0)])
            writer.writerow([])  # Empty row
            
            writer.writerow(['Metastasis #', 'Volume (mm³)', '% of Total'])
            total_volume = result_data.get('total_volume', 0)
            
            for i, volume in enumerate(result_data.get('metastasis_volumes', [])):
                percentage = (volume / total_volume * 100) if total_volume > 0 else 0
                writer.writerow([i + 1, round(volume, 2), f"{round(percentage, 1)}%"])
            
            output.seek(0)
            
            return output.getvalue(), 200, {
                'Content-Type': 'text/csv',
                'Content-Disposition': f'attachment; filename=shared_metastasis_results_{job_id}.csv'
            }
            
        elif format == 'pdf':
            # Generate PDF
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet
            import io
            
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            
            elements = []
            
            # Title
            title = Paragraph("Brain Metastasis Analysis Results (Shared)", styles['Title'])
            elements.append(title)
            elements.append(Spacer(1, 12))
            
            # Summary data
            summary_data = [
                ["Job ID:", job_id],
                ["Total Metastasis Count:", str(result_data.get('metastasis_count', 0))],
                ["Total Volume (mm³):", str(round(result_data.get('total_volume', 0), 2))]
            ]
            
            summary_table = Table(summary_data, colWidths=[120, 300])
            summary_table.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ]))
            
            elements.append(summary_table)
            elements.append(Spacer(1, 24))
            
            # Metastasis data
            metastasis_title = Paragraph("Individual Metastasis Analysis", styles['Heading2'])
            elements.append(metastasis_title)
            elements.append(Spacer(1, 12))
            
            metastasis_data = [["Metastasis #", "Volume (mm³)", "% of Total"]]
            total_volume = result_data.get('total_volume', 0)
            
            for i, volume in enumerate(result_data.get('metastasis_volumes', [])):
                percentage = (volume / total_volume * 100) if total_volume > 0 else 0
                metastasis_data.append([i + 1, round(volume, 2), f"{round(percentage, 1)}%"])
            
            metastasis_table = Table(metastasis_data, colWidths=[100, 100, 100])
            metastasis_table.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('ALIGN', (1, 1), (2, -1), 'RIGHT'),
            ]))
            
            elements.append(metastasis_table)
            
            # Add shared notice
            elements.append(Spacer(1, 24))
            shared_notice = Paragraph("Note: This report was generated from a shared link.", styles['Normal'])
            elements.append(shared_notice)
            
            # Build PDF
            doc.build(elements)
            buffer.seek(0)
            
            return buffer.getvalue(), 200, {
                'Content-Type': 'application/pdf',
                'Content-Disposition': f'attachment; filename=shared_metastasis_results_{job_id}.pdf'
            }
            
    except Exception as e:
        logging.error(f"Error downloading shared results: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/shares', methods=['GET'])
@token_required
def list_user_shares():
    """
    List all share links created by the authenticated user
    """
    try:
        auth_header = request.headers.get('Authorization')
        
        response = requests.get(
            f"{USER_SERVICE_URL}/shares",
            headers={"Authorization": auth_header},
            params=request.args
        )
        
        return response.json(), response.status_code
        
    except Exception as e:
        logging.error(f"Error listing shares: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/shares/<share_token>', methods=['DELETE'])
@token_required
def revoke_share(share_token):
    """
    Revoke/delete a share link
    """
    try:
        auth_header = request.headers.get('Authorization')
        
        response = requests.delete(
            f"{USER_SERVICE_URL}/shares/{share_token}",
            headers={"Authorization": auth_header}
        )
        
        return response.json(), response.status_code
        
    except Exception as e:
        logging.error(f"Error revoking share: {str(e)}")
        return jsonify({"error": str(e)}), 500
