# API Documentation

This document provides detailed information about the API endpoints for the Brain Metastasis Analysis System.

## Authentication Endpoints

### Register a New User
- **URL:** `/auth/register`
- **Method:** `POST`
- **Auth Required:** No
- **Request Body:**
  ```json
  {
    "name": "User Name",
    "email": "user@example.com",
    "password": "securepassword"
  }
  ```
- **Success Response:**
  - **Code:** 201
  - **Content:** 
    ```json
    {
      "message": "User registered successfully",
      "token": "jwt-token-string",
      "name": "User Name",
      "email": "user@example.com"
    }
    ```
- **Error Responses:**
  - **Code:** 400 BAD REQUEST - Invalid input data
  - **Code:** 409 CONFLICT - User already exists

### Login
- **URL:** `/auth/login`
- **Method:** `POST`
- **Auth Required:** No
- **Request Body:**
  ```json
  {
    "email": "user@example.com",
    "password": "securepassword"
  }
  ```
- **Success Response:**
  - **Code:** 200
  - **Content:** 
    ```json
    {
      "message": "Login successful",
      "token": "jwt-token-string",
      "name": "User Name",
      "email": "user@example.com"
    }
    ```
- **Error Response:**
  - **Code:** 401 UNAUTHORIZED - Invalid credentials

### Logout
- **URL:** `/auth/logout`
- **Method:** `POST`
- **Auth Required:** Yes (Bearer Token)
- **Headers:** 
  - `Authorization: Bearer <token>`
- **Success Response:**
  - **Code:** 200
  - **Content:** 
    ```json
    {
      "message": "Successfully logged out"
    }
    ```
- **Error Response:**
  - **Code:** 401 UNAUTHORIZED - Missing or invalid token

<!-- 2FA Authentication has been removed from the implementation -->

## User Data Endpoints

### Get User Profile
- **URL:** `/user/profile`
- **Method:** `GET`
- **Auth Required:** Yes (Bearer Token)
- **Headers:** 
  - `Authorization: Bearer <token>`
- **Success Response:**
  - **Code:** 200
  - **Content:** 
    ```json
    {
      "name": "User Name",
      "email": "user@example.com",
      "created_at": "2023-06-05T14:48:00.000Z",
      "role": "user"
    }
    ```
- **Error Response:**
  - **Code:** 401 UNAUTHORIZED - Missing or invalid token

### Get User Scan History
- **URL:** `/user/scans`
- **Method:** `GET`
- **Auth Required:** Yes (Bearer Token)
- **Headers:** 
  - `Authorization: Bearer <token>`
- **Success Response:**
  - **Code:** 200
  - **Content:** 
    ```json
    {
      "scans": [
        {
          "job_id": "a1b2c3d4-e5f6-7890-abcd-1234567890ab",
          "file_name": "brain_scan.npy",
          "status": "completed",
          "created_at": "2023-06-05T14:48:00.000Z",
          "metastasis_count": 3,
          "total_volume": 215.6
        },
        {
          "job_id": "b2c3d4e5-f6a7-8901-bcde-2345678901bc",
          "file_name": "follow_up_scan.npy",
          "status": "processing",
          "created_at": "2023-06-10T09:15:00.000Z"
        }
      ]
    }
    ```
- **Error Response:**
  - **Code:** 401 UNAUTHORIZED - Missing or invalid token

## MRI Scan Processing Endpoints

### Upload MRI Scan
- **URL:** `/upload`
- **Method:** `POST`
- **Auth Required:** Optional (Bearer Token for user association)
- **Content-Type:** `multipart/form-data`
- **Form Data:**
  - `file`: The MRI scan file in supported formats:
    - **NIfTI format (.nii or .nii.gz)** - Will be automatically converted and preprocessed (up to 100MB)
    - **NumPy format (.npy)** - Direct input format (up to 50MB)
- **Success Response:**
  - **Code:** 200
  - **Content:** 
    ```json
    {
      "message": "File uploaded successfully",
      "job_id": "a1b2c3d4-e5f6-7890-abcd-1234567890ab",
      "status": "processing",
      "user_associated": true
    }
    ```
- **Error Responses:**
  - **Code:** 400 BAD REQUEST - Missing file or unsupported format
  - **Code:** 500 SERVER ERROR - Processing error

### Get Scan Results
- **URL:** `/results/{job_id}`
- **Method:** `GET`
- **URL Parameters:**
  - `job_id`: The ID of the processing job
- **Success Response (Processing):**
  - **Code:** 200
  - **Content:** 
    ```json
    {
      "job_id": "a1b2c3d4-e5f6-7890-abcd-1234567890ab",
      "status": "processing",
      "message": "Segmentation still processing"
    }
    ```
- **Success Response (Completed):**
  - **Code:** 200
  - **Content:** 
    ```json
    {
      "job_id": "a1b2c3d4-e5f6-7890-abcd-1234567890ab",
      "status": "completed",
      "segmentation_path": "/visualization/a1b2c3d4-e5f6-7890-abcd-1234567890ab",
      "metastasis_count": 3,
      "metastasis_volumes": [125.6, 75.0, 15.0],
      "total_volume": 215.6
    }
    ```
- **Error Responses:**
  - **Code:** 404 NOT FOUND - Invalid job ID
  - **Code:** 500 SERVER ERROR - Processing error

### Get Visualization Image
- **URL:** `/visualization/{job_id}`
- **Method:** `GET`
- **URL Parameters:**
  - `job_id`: The ID of the processing job
- **Success Response:**
  - **Code:** 200
  - **Content:** Image file (PNG)
- **Error Response:**
  - **Code:** 404 NOT FOUND - Visualization not found

## Export Endpoints

### Export Results as CSV
- **URL:** `/export/csv/{job_id}`
- **Method:** `GET`
- **Auth Required:** Yes (Token)
- **Success Response:**
  - **Code:** 200
  - **Content Type:** `text/csv`
  - **Content Disposition:** `attachment; filename=metastasis_results_{job_id}.csv`
- **Error Responses:**
  - **Code:** 401 UNAUTHORIZED - Invalid or missing authentication token
  - **Code:** 404 NOT FOUND - Results not found or not processed yet
  - **Code:** 500 INTERNAL SERVER ERROR - Server error during export

### Export Results as PDF
- **URL:** `/export/pdf/{job_id}`
- **Method:** `GET`
- **Auth Required:** Yes (Token)
- **Success Response:**
  - **Code:** 200
  - **Content Type:** `application/pdf`
  - **Content Disposition:** `attachment; filename=metastasis_results_{job_id}.pdf`
- **Error Responses:**
  - **Code:** 401 UNAUTHORIZED - Invalid or missing authentication token
  - **Code:** 404 NOT FOUND - Results not found or not processed yet
  - **Code:** 500 INTERNAL SERVER ERROR - Server error during export

### Filter Scan History
- **URL:** `/user/scans/filter`
- **Method:** `GET`
- **Auth Required:** Yes (Token)
- **Query Parameters:**
  - `min_metastasis` (optional) - Minimum number of metastases
  - `max_metastasis` (optional) - Maximum number of metastases
  - `min_volume` (optional) - Minimum total volume (mm³)
  - `max_volume` (optional) - Maximum total volume (mm³)
  - `start_date` (optional) - Filter from this date (YYYY-MM-DD)
  - `end_date` (optional) - Filter until this date (YYYY-MM-DD)
  - `page` (optional) - Page number (default: 1)
  - `per_page` (optional) - Items per page (default: 10)
- **Success Response:**
  - **Code:** 200
  - **Content:** 
    ```json
    {
      "scans": [
        {
          "job_id": "abc123",
          "file_name": "patient_scan.npy",
          "created_at": "2025-06-10T14:32:17",
          "status": "completed",
          "metastasis_count": 3,
          "total_volume": 12.45
        }
      ],
      "pagination": {
        "total_items": 25,
        "total_pages": 3,
        "current_page": 1,
        "per_page": 10
      }
    }
    ```
- **Error Responses:**
  - **Code:** 401 UNAUTHORIZED - Invalid or missing authentication token
  - **Code:** 500 INTERNAL SERVER ERROR - Server error during filtering

## Health Check Endpoints

### API Gateway Health Check
- **URL:** `/health`
- **Method:** `GET`
- **Success Response:**
  - **Code:** 200
  - **Content:** 
    ```json
    {
      "status": "healthy",
      "service": "api-gateway"
    }
    ```

### Model Service Health Check
- **URL:** `http://model-service:5001/health`
- **Method:** `GET`
- **Success Response:**
  - **Code:** 200
  - **Content:** 
    ```json
    {
      "status": "healthy",
      "service": "model-service"
    }
    ```

### Image Processing Service Health Check
- **URL:** `http://image-processing-service:5002/health`
- **Method:** `GET`
- **Success Response:**
  - **Code:** 200
  - **Content:** 
    ```json
    {
      "status": "healthy",
      "service": "image-processing-service"
    }
    ```

### User Service Health Check
- **URL:** `http://user-service:5003/health`
- **Method:** `GET`
- **Success Response:**
  - **Code:** 200
  - **Content:** 
    ```json
    {
      "status": "healthy",
      "service": "user-service"
    }
    ```

### Basic Analysis
- **URL:** `/analyze/{job_id}`
- **Method:** `GET` 
- **Auth Required:** Yes
- **URL Parameters:**
  - `job_id` - The ID of the processing job to analyze
- **Success Response:**
  - **Code:** 200
  - **Content:**
    ```json
    {
      "job_id": "example_job_id",
      "metastasis_count": 3,
      "metastasis_volumes": [245.5, 123.2, 78.9],
      "total_volume": 447.6,
      "average_volume": 149.2
    }
    ```
- **Error Responses:**
  - **Code:** 404 NOT FOUND - Segmentation not found
  - **Code:** 400 BAD REQUEST - Invalid job ID
  - **Code:** 500 INTERNAL ERROR - Analysis failed

### Advanced Multi-Class Analysis
- **URL:** `/advanced-analysis/{job_id}`
- **Method:** `GET`
- **Auth Required:** Yes
- **URL Parameters:**
  - `job_id` - The ID of the processing job to analyze
- **Success Response:**
  - **Code:** 200
  - **Content:**
    ```json
    {
      "job_id": "example_job_id",
      "total_classes_found": 3,
      "classes": {
        "Metastasis": {
          "count": 3,
          "regions": [
            {
              "id": 1,
              "volume_mm3": 245.5,
              "centroid": [45.2, 67.8, 23.1],
              "equivalent_diameter_mm": 7.8,
              "sphericity": 0.87
            },
            ...
          ],
          "total_volume": 447.6,
          "average_volume": 149.2
        },
        "Edema": {
          "count": 2,
          "regions": [...],
          "total_volume": 1240.5,
          "average_volume": 620.25
        },
        ...
      },
      "overall_summary": {
        "total_regions": 6,
        "total_volume": 1915.2
      }
    }
    ```
- **Error Responses:**
  - **Code:** 404 NOT FOUND - Segmentation not found
  - **Code:** 400 BAD REQUEST - Invalid job ID
  - **Code:** 500 INTERNAL ERROR - Analysis failed

### Visualization Generation
- **URL:** `/visualization/{job_id}`
- **Method:** `GET`
- **Auth Required:** Yes
- **URL Parameters:**
  - `job_id` - The ID of the processing job
- **Query Parameters:**
  - `type` - Visualization type (slice, projection, multi-slice, lesions)
  - `class_id` - Optional class ID to highlight
  - `slice_idx` - Optional slice index for slice visualizations
  - `num_slices` - Optional number of slices for multi-slice view (default: 5)
- **Success Response:**
  - **Code:** 200
  - **Content:** PNG image data
- **Error Responses:**
  - **Code:** 404 NOT FOUND - Segmentation not found
  - **Code:** 400 BAD REQUEST - Invalid parameters
  - **Code:** 500 INTERNAL ERROR - Visualization generation failed

### Lesion Analysis
- **URL:** `/lesion-analysis/{job_id}`
- **Method:** `GET`
- **Auth Required:** Yes
- **URL Parameters:**
  - `job_id` - The ID of the processing job
- **Query Parameters:**
  - `class_id` - Optional specific class ID to analyze
- **Success Response:**
  - **Code:** 200
  - **Content:**
    ```json
    {
      "job_id": "example_job_id",
      "Metastasis": {
        "count": 3,
        "regions": [
          {
            "id": 1,
            "volume_mm3": 245.5,
            "voxel_count": 245,
            "centroid": [45.2, 67.8, 23.1]
          },
          ...
        ],
        "total_volume": 447.6,
        "average_volume": 149.2
      },
      "Edema": {
        "count": 2,
        "regions": [...],
        "total_volume": 1240.5,
        "average_volume": 620.25
      }
    }
    ```
- **Error Responses:**
  - **Code:** 404 NOT FOUND - Segmentation not found
  - **Code:** 400 BAD REQUEST - Invalid parameters
  - **Code:** 500 INTERNAL ERROR - Analysis failed

### Slice Summary
- **URL:** `/slice-summary/{job_id}`
- **Method:** `GET`
- **Auth Required:** Yes
- **URL Parameters:**
  - `job_id` - The ID of the processing job
- **Query Parameters:**
  - `with_graph` - Optional boolean to include distribution graph (true/false)
- **Success Response:**
  - **Code:** 200
  - **Content:**
    ```json
    {
      "job_id": "example_job_id",
      "total_slices": 128,
      "slices_with_segmentation": 45,
      "best_slices_per_class": {
        "Metastasis": {
          "slice_idx": 64,
          "count": 1240
        },
        "Edema": {
          "slice_idx": 65,
          "count": 3560
        }
      },
      "slice_data": [
        {
          "slice_idx": 62,
          "classes": {
            "Metastasis": 982,
            "Edema": 2140
          },
          "total_segmented_voxels": 3122
        },
        ...
      ],
      "distribution_graph": "base64_encoded_image_data" 
    }
    ```
- **Error Responses:**
  - **Code:** 404 NOT FOUND - Segmentation not found
  - **Code:** 400 BAD REQUEST - Invalid parameters
  - **Code:** 500 INTERNAL ERROR - Summary generation failed

### Set Metadata
- **URL:** `/metadata`
- **Method:** `POST`
- **Auth Required:** Yes
- **Request Body:**
  ```json
  {
    "voxel_volume_mm3": 1.5,
    "metastasis_class": 3,
    "tissue_names": {
      "1": "Tumor Core",
      "2": "Edema",
      "3": "Metastasis"
    },
    "tissue_colors": {
      "1": [0.0, 0.0, 1.0],
      "2": [0.0, 1.0, 0.0],
      "3": [1.0, 0.0, 0.0]
    }
  }
  ```
- **Success Response:**
  - **Code:** 200
  - **Content:**
    ```json
    {
      "message": "Metadata updated successfully",
      "voxel_volume_mm3": 1.5,
      "metastasis_class": 3,
      "tissue_names": {
        "1": "Tumor Core",
        "2": "Edema",
        "3": "Metastasis"
      },
      "tissue_colors": {
        "1": [0.0, 0.0, 1.0],
        "2": [0.0, 1.0, 0.0],
        "3": [1.0, 0.0, 0.0]
      }
    }
    ```
- **Error Responses:**
  - **Code:** 400 BAD REQUEST - Invalid parameters
  - **Code:** 500 INTERNAL ERROR - Update failed
