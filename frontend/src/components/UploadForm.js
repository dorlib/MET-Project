import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { 
  Box, 
  Typography, 
  Paper, 
  Button, 
  CircularProgress,
  Alert,
  LinearProgress
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import api from '../services/api';

const UploadForm = ({ onUploadSuccess }) => {
  const [uploading, setUploading] = React.useState(false);
  const [error, setError] = React.useState(null);
  const [file, setFile] = React.useState(null);
  const [uploadProgress, setUploadProgress] = React.useState(0);
  const [processingMessage, setProcessingMessage] = React.useState('');

  const onDrop = useCallback(acceptedFiles => {
    // Accept .npy, .nii, and .nii.gz files
    const validFiles = acceptedFiles.filter(file => 
      file.name.endsWith('.npy') || 
      file.name.endsWith('.nii') || 
      file.name.endsWith('.nii.gz')
    );
    
    if (validFiles.length === 0) {
      setError('Please upload a supported file (.npy, .nii, .nii.gz)');
      return;
    }
    
    const selectedFile = validFiles[0];
    
    // Check file size (limit to 100MB as NIfTI files can be larger)
    if (selectedFile.size > 100 * 1024 * 1024) {
      setError('File size exceeds 100MB limit');
      return;
    }
    
    // Check if file name contains invalid characters
    const invalidCharsRegex = /[<>:"/\\|?*]/g;
    if (invalidCharsRegex.test(selectedFile.name)) {
      setError('Filename contains invalid characters');
      return;
    }
    
    setFile(selectedFile);
    setError(null);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/octet-stream': ['.npy', '.nii', '.nii.gz'],
      'application/x-nifti-1': ['.nii'],
      'application/gzip': ['.nii.gz']
    },
    maxFiles: 1
  });

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file first');
      return;
    }
    
    // Additional validation before upload - check for supported file types
    const isValidFile = file.name.endsWith('.npy') || 
                       file.name.endsWith('.nii') || 
                       file.name.endsWith('.nii.gz');
    if (!isValidFile) {
      setError('Only .npy, .nii, and .nii.gz files are supported');
      return;
    }
    
    // Double check file size - different limit for different formats
    const sizeLimit = file.name.endsWith('.npy') ? 50 : 100; // MB
    if (file.size > sizeLimit * 1024 * 1024) {
      setError(`File size exceeds ${sizeLimit}MB limit`);
      return;
    }
    
    // Check if file is empty
    if (file.size === 0) {
      setError('File appears to be empty');
      return;
    }

    setUploading(true);
    setError(null);
    setUploadProgress(0);
    
    // Set initial processing message based on file type
    if (file.name.endsWith('.nii') || file.name.endsWith('.nii.gz')) {
      setProcessingMessage('Preparing to process NIfTI file. This may take longer than NPY files.');
    }

    try {
      const formData = new FormData();
      formData.append('file', file);

      // Create a simulated progress for better UX
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 95) {
            clearInterval(progressInterval);
            return 95;
          }
          const isNifti = file.name.endsWith('.nii') || file.name.endsWith('.nii.gz');
          // Slow down progress for NIfTI files to reflect longer processing time
          const increment = isNifti ? 
            (prev < 40 ? 1 : 0.5) : 
            (prev < 60 ? 3 : 1.5);
          return Math.min(95, prev + increment);
        });
        
        // Update processing message based on progress
        if (file.name.endsWith('.nii') || file.name.endsWith('.nii.gz')) {
          if (uploadProgress > 40 && uploadProgress < 70) {
            setProcessingMessage('Converting NIfTI to the required format...');
          } else if (uploadProgress >= 70) {
            setProcessingMessage('Analyzing brain scan for metastases...');
          }
        }
      }, 500);

      const response = await api.uploadScan(formData, (progress) => {
        setUploadProgress(progress);
      });
      
      clearInterval(progressInterval);
      setUploadProgress(100);
      setFile(null); // Clear the file after successful upload
      onUploadSuccess(response.data);
    } catch (err) {
      console.error('Error uploading file:', err);
      // Detailed error handling
      let errorMessage = 'Failed to upload file';
      
      if (err.response) {
        // Server responded with an error
        errorMessage = err.response.data?.error || `Server error: ${err.response.status}`;
      } else if (err.request) {
        // Request was made but no response was received
        errorMessage = 'No response from server. Please check your internet connection.';
      }
      
      setError(errorMessage);
    } finally {
      setUploading(false);
      setProcessingMessage('');
    }
  };

  return (
    <Paper sx={{ p: 3, mb: 4 }}>
      <Typography variant="h5" gutterBottom>
        Upload MRI Scan
      </Typography>
      
      <Typography variant="body2" color="text.secondary" paragraph>
        Upload a T1 contrast-enhanced MRI scan in .npy, .nii, or .nii.gz format for brain metastasis segmentation and analysis.
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Box
        {...getRootProps()}
        sx={{
          border: '2px dashed #90caf9',
          borderRadius: 2,
          p: 3,
          textAlign: 'center',
          cursor: 'pointer',
          backgroundColor: isDragActive ? 'rgba(144, 202, 249, 0.1)' : 'transparent',
          mb: 3
        }}
      >
        <input {...getInputProps()} />
        <CloudUploadIcon sx={{ fontSize: 48, color: 'primary.main', mb: 1 }} />
        {isDragActive ? (
          <Typography>Drop the file here...</Typography>
        ) : (
          <Typography>Drag and drop a .npy, .nii, .nii.gz file here, or click to select a file</Typography>
        )}
        {file && (
          <Box mt={2}>
            <Alert severity="success">
              Selected file: {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
            </Alert>
          </Box>
        )}
      </Box>

      {processingMessage && (
        <Alert severity="info" sx={{ mb: 2 }}>
          {processingMessage}
        </Alert>
      )}
      
      <Button
        variant="contained"
        onClick={handleUpload}
        disabled={!file || uploading}
        startIcon={uploading ? <CircularProgress size={20} color="inherit" /> : null}
        fullWidth
      >
        {uploading ? 'Uploading...' : 'Upload and Analyze'}
      </Button>
      
      {uploading && (
        <Box sx={{ width: '100%', mt: 2 }}>
          <Typography variant="body2" color="text.secondary" align="center" sx={{ mb: 1 }}>
            {file && (file.name.endsWith('.nii') || file.name.endsWith('.nii.gz')) ? 
              'Converting and analyzing NIfTI file, this may take several minutes...' : 
              'Processing file...'}
          </Typography>
          <LinearProgress variant="determinate" value={uploadProgress} />
        </Box>
      )}
    </Paper>
  );
};

export default UploadForm;
