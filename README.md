# Brain Metastasis Analysis System

This project provides a complete system for analyzing brain metastases in MRI scans using deep learning segmentation with a UNETR model. The system includes microservices for model inference, image processing for quantifying metastases, user authentication, and a comprehensive web frontend for user interaction.

## Supported File Formats

The system now supports multiple file formats for MRI brain scan inputs:
* **NumPy (.npy)** - Original format for preprocessed data
* **NIfTI (.nii, .nii.gz)** - Standard neuroimaging format

## Project Structure

```
MET-project/
├── API_DOCS.md                      # API documentation
├── backend/
│   ├── api_gateway/                 # API gateway service
│   ├── model_service/               # UNETR model inference service
│   ├── image_processing_service/    # Image analysis service for metastasis quantification
│   └── user_service/                # User management and authentication service
├── frontend/                        # React-based web interface
├── docker-compose.yml               # Docker services configuration
├── test_services.py                 # End-to-end testing script
└── unetr_t1c.py                     # Original UNETR model implementation
```

## Services

### 1. API Gateway
- Entry point for all frontend requests
- Handles file uploads and routes requests to appropriate services
- Aggregates results from various services

### 2. Model Service
- Runs the UNETR segmentation model on uploaded MRI scans
- Converts the model input/output for service use
- Generates visualizations of segmentation results

### 3. Image Processing Service
- Analyzes segmentation masks to identify individual metastases
- Calculates the count and volume of metastases
- Provides detailed measurements and visualization for clinical decision support
- Features advanced multi-class tissue analysis with 3D visualizations
- Generates detailed lesion analysis with individual metastases characterization
- Supports analysis of different tissue types (metastases, edema, tumor core)

### 4. User Service
- Manages user authentication (login/register)
- Stores user profiles and scan history in MySQL database
- Provides filtering and retrieval of scan results for users

### 5. Frontend
- Provides an intuitive interface for uploading MRI scans
- Visualizes segmentation results
- Displays metastasis statistics (count, volume, distribution)
- Provides user authentication and scan history

## Getting Started

### Prerequisites
- Docker and Docker Compose
- NVIDIA GPU (recommended) with CUDA support
- Trained model weights file (`brats_t1ce.pth`)

### Setup and Installation

1. Place your trained model in the appropriate location:
   ```
   /Data/saved_models/brats_t1ce.pth
   ```

2. Build and start the services:
   ```bash
   docker-compose up --build
   ```

3. Access the web interface:
   ```
   http://localhost:3000
   ```

## Usage

1. Upload a T1 contrast-enhanced MRI scan in one of the supported formats:
   - NIfTI format (`.nii` or `.nii.gz`) - automatically converted and preprocessed
   - NumPy format (`.npy`)
2. The system processes the scan using the UNETR segmentation model
3. View the segmentation results with advanced analysis capabilities:
   - Multi-class tissue analysis (metastases, edema, tumor core)
   - 3D projections and multi-slice visualizations
   - Detailed lesion characterization with volumetrics
   - Slice-by-slice distribution analysis
4. Examine and interact with comprehensive analysis results:
   - Count, volume, and spatial distribution of detected metastases
   - Shape characteristics and location data for each lesion
   - Interactive visualizations with customizable parameters

## API Endpoints

### API Gateway (`http://localhost:5000`)

- `POST /upload` - Upload an MRI scan file
- `GET /results/{job_id}` - Get processing results and analysis
- `GET /visualization/{job_id}` - Get segmentation visualization image
- `GET /advanced-analysis/{job_id}` - Get comprehensive multi-class tissue analysis
- `GET /advanced-visualization/{job_id}` - Get advanced visualizations with customizable parameters
- `GET /lesion-analysis/{job_id}` - Get detailed analysis of individual lesions
- `GET /slice-summary/{job_id}` - Get slice-by-slice tissue distribution analysis
- `POST /analysis-metadata` - Update analysis settings (voxel size, tissue classes)
- `GET /health` - Service health check

## Development

### Adding New Features

- **Model improvements**: Update the `unetr_adapter.py` file in model_service
- **Frontend enhancements**: Modify React components in the frontend/src directory
- **Analysis algorithms**: Extend functionality in the image_processing_service

### Running Services Individually

Each service can be run independently during development:

```bash
# API Gateway
cd backend/api_gateway
python api.py

# Model Service
cd backend/model_service
python model_service.py

# Image Processing Service
cd backend/image_processing_service
python image_processor.py

# Frontend
cd frontend
npm start
```

## Security Features

### Two-Factor Authentication

This application supports Two-Factor Authentication (2FA) to provide an additional layer of security:

- Users can enable 2FA from their profile page
- 2FA uses Time-based One-Time Passwords (TOTP) compatible with apps like Google Authenticator
- Once enabled, login requires both a password and a 2FA verification code

### Enhanced Data Security

- Secure authentication with JWT tokens
- Proper password hashing with bcrypt
- HTTPS support (when deployed)
- Input validation and sanitization

## Advanced Features

### Data Export

Results can be exported in multiple formats:
- CSV export for spreadsheet analysis
- PDF reports with detailed metastasis information

### Search and Filtering

- Filter scans by date range
- Filter by metastasis count and volume
- Advanced sorting capabilities

### 3D Visualization

- Interactive 3D rendering of brain and metastasis location
- Customizable visualization controls (opacity, rotation)
- Volume-based size representation
