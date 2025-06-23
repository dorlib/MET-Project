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
CORS(app)
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
                json={"file_path": file_path, "job_id": job_id}
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
    
    if model_status_response.status_code != 200:
        return jsonify({"error": "Invalid job ID or processing error"}), 404
    
    model_status = model_status_response.json()
    
    if model_status.get("status") != "completed":
        return jsonify({
            "job_id": job_id,
            "status": model_status.get("status", "unknown"),
            "message": "Segmentation still processing"
        })
    
    # Get metastasis analysis from image processing service
    analysis_response = requests.get(
        f"{IMAGE_PROCESSING_SERVICE_URL}/analyze/{job_id}"
    )
    
    if analysis_response.status_code != 200:
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
        "metastasis_count": analysis_data.get("metastasis_count"),
        "metastasis_volumes": analysis_data.get("metastasis_volumes"),
        "total_volume": analysis_data.get("total_volume"),
    })

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

# 2FA endpoints have been removed

@app.route('/export/csv/<job_id>', methods=['GET'])
@token_required
def export_csv(job_id):
    """
    Export scan results as CSV
    """
    try:
        # First, get result data
        response = requests.get(f"{IMAGE_PROCESSING_SERVICE_URL}/results/{job_id}")
        
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
        # First, get result data
        response = requests.get(f"{IMAGE_PROCESSING_SERVICE_URL}/results/{job_id}")
        
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

@app.route('/user/scans/filter', methods=['GET'])
@token_required
def filter_scans():
    """
    Get filtered scan history with advanced filtering
    
    Query parameters:
    - min_metastasis: Minimum number of metastases
    - max_metastasis: Maximum number of metastases
    - min_volume: Minimum total volume (mm³)
    - max_volume: Maximum total volume (mm³)
    - start_date: Filter from this date (YYYY-MM-DD)
    - end_date: Filter until this date (YYYY-MM-DD)
    - page: Page number (default: 1)
    - per_page: Items per page (default: 10)
    """
    try:
        # Forward to user service with token and filters
        auth_header = request.headers.get('Authorization')
        
        # Forward all query parameters
        response = requests.get(
            f"{USER_SERVICE_URL}/scans/filter",
            headers={"Authorization": auth_header},
            params=request.args
        )
        
        return response.json(), response.status_code
    except Exception as e:
        logging.error(f"Error filtering scans: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/volume-dimensions/<job_id>', methods=['GET'])
def get_volume_dimensions(job_id):
    """
    Get the dimensions of the volume for a specific job
    """
    # Validate job_id format to prevent path traversal
    if '/' in job_id or '\\' in job_id or '..' in job_id:
        return jsonify({"error": "Invalid job ID format"}), 400
    
    try:
        # First, try to get dimensions from the image processing service if available
        try:
            response = requests.get(f"{IMAGE_PROCESSING_SERVICE_URL}/volume-info/{job_id}")
            if response.status_code == 200:
                return response.json(), 200
        except Exception as e:
            logging.warning(f"Could not get volume info from image processing service: {str(e)}")
        
        # If the specific endpoint is not available, try to load the prediction file locally
        pred_path = os.path.join(RESULTS_FOLDER, f"{job_id}_prediction.npy")
        
        if not os.path.exists(pred_path):
            return jsonify({"error": "Segmentation not found"}), 404
            
        try:
            # Load the prediction file to get dimensions
            data = np.load(pred_path)
            dimensions = data.shape
            return jsonify({
                "job_id": job_id,
                "dimensions": dimensions,
                "max_slice_index": dimensions[0] - 1 if len(dimensions) > 0 else 0
            }), 200
        except Exception as file_error:
            logging.warning(f"Error loading prediction file: {str(file_error)}")
            
        # If we reach here, provide a reasonable default
        return jsonify({
            "job_id": job_id,
            "dimensions": [128, 128, 128],  # Typical MRI dimensions
            "max_slice_index": 127
        }), 200
    except Exception as e:
        logging.error(f"Error getting volume dimensions: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """
    Endpoint to check the status of a processing job
    """
    try:
        # Forward request to model service
        model_status_response = requests.get(f"{MODEL_SERVICE_URL}/status/{job_id}")
        
        if model_status_response.status_code != 200:
            logging.warning(f"Model service returned status {model_status_response.status_code} for job {job_id}")
            return jsonify({"status": "not_found", "error": "Job not found"}), 404
        
        # Return the status from model service
        return jsonify(model_status_response.json())
    except Exception as e:
        logging.error(f"Error checking status for job {job_id}: {str(e)}")
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/model/health', methods=['GET'])
def model_health_check():
    """
    Endpoint to check the health of the model service
    """
    try:
        response = requests.get(f"{MODEL_SERVICE_URL}/health")
        return jsonify(response.json()), response.status_code
    except Exception as e:
        logging.error(f"Error checking model service health: {str(e)}")
        return jsonify({"status": "error", "error": str(e)}), 503

@app.route('/side-by-side-visualization/<job_id>', methods=['GET'])
def get_side_by_side_visualization(job_id):
    """
    Endpoint to get side-by-side visualization of original image and segmentation mask
    
    Args:
        job_id: Job ID for the scan
        
    Query parameters:
    - slice_idx: Optional slice index for the visualization
    - view_type: View type (axial, coronal, sagittal)
    - upscale_factor: Optional upscale factor for visualization
    - contrast_enhancement: Whether to apply contrast enhancement (true/false)
    - edge_enhancement: Whether to apply edge enhancement (true/false)
    """
    try:
        # Log the visualization request parameters
        logging.info(f"Side-by-side visualization request for job {job_id} with params: {request.args}")
        
        # Forward the request to the image processing service with all query parameters
        viz_response = requests.get(
            f"{IMAGE_PROCESSING_SERVICE_URL}/side-by-side/{job_id}",
            params=request.args,
            stream=True
        )
        
        if viz_response.status_code != 200:
            logging.error(f"Image processing service returned error: {viz_response.status_code}")
            error_data = viz_response.json() if viz_response.headers.get('Content-Type') == 'application/json' else {"error": "Visualization generation failed"}
            return jsonify(error_data), viz_response.status_code
        
        logging.info(f"Side-by-side visualization successfully generated for job {job_id}")
        
        # Stream the visualization image back to the client
        return Response(
            viz_response.iter_content(chunk_size=1024),
            status=viz_response.status_code,
            headers={
                'Content-Type': viz_response.headers.get('Content-Type', 'image/png'),
                'Content-Disposition': viz_response.headers.get('Content-Disposition', f'inline; filename="{job_id}_side_by_side.png"')
            }
        )
    except Exception as e:
        logging.error(f"Error generating side-by-side visualization for job {job_id}: {str(e)}")
        return jsonify({"error": f"Visualization failed: {str(e)}"}), 500

@app.route('/three-plane-visualization/<job_id>', methods=['GET'])
def get_three_plane_visualization(job_id):
    """
    Endpoint to get visualization with all three anatomical planes side by side
    
    Args:
        job_id: Job ID for the scan
        
    Query parameters:
    - axial_slice_idx: Optional axial slice index
    - coronal_slice_idx: Optional coronal slice index
    - sagittal_slice_idx: Optional sagittal slice index
    - contrast_enhancement: Whether to apply contrast enhancement (true/false)
    - edge_enhancement: Whether to apply edge enhancement (true/false)
    """
    try:
        # Log the visualization request parameters
        logging.info(f"Three-plane visualization request for job {job_id} with params: {request.args}")
        
        # Forward the request to the image processing service with all query parameters
        viz_response = requests.get(
            f"{IMAGE_PROCESSING_SERVICE_URL}/three-plane/{job_id}",
            params=request.args,
            stream=True
        )
        
        if viz_response.status_code != 200:
            logging.error(f"Image processing service returned error: {viz_response.status_code}")
            error_data = viz_response.json() if viz_response.headers.get('Content-Type') == 'application/json' else {"error": "Visualization generation failed"}
            return jsonify(error_data), viz_response.status_code
        
        logging.info(f"Three-plane visualization successfully generated for job {job_id}")
        
        # Stream the visualization image back to the client
        return Response(
            viz_response.iter_content(chunk_size=1024),
            status=viz_response.status_code,
            headers={
                'Content-Type': viz_response.headers.get('Content-Type', 'image/png'),
                'Content-Disposition': viz_response.headers.get('Content-Disposition', f'inline; filename="{job_id}_three_plane.png"')
            }
        )
    except Exception as e:
        logging.error(f"Error generating three-plane visualization for job {job_id}: {str(e)}")
        return jsonify({"error": f"Visualization failed: {str(e)}"}), 500
