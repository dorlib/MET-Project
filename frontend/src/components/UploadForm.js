import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { 
  Box, 
  Typography, 
  Paper, 
  Button, 
  CircularProgress,
  Alert,
  LinearProgress,
  Container,
  Grid,
  Card,
  CardContent,
  Fade,
  Chip,
  Avatar
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import BrainIcon from '@mui/icons-material/Psychology';
import SecurityIcon from '@mui/icons-material/Security';
import SpeedIcon from '@mui/icons-material/Speed';
import VisibilityIcon from '@mui/icons-material/Visibility';
import CloudIcon from '@mui/icons-material/Cloud';
import MedicalInformationIcon from '@mui/icons-material/MedicalInformation';
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
    <Container maxWidth="lg">
      {/* Hero Section */}
      <Fade in={true} timeout={1000}>
        <Box
          sx={{
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            borderRadius: 4,
            p: 6,
            mb: 6,
            color: 'white',
            textAlign: 'center',
            position: 'relative',
            overflow: 'hidden',
            '&::before': {
              content: '""',
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              background: 'url("data:image/svg+xml,%3Csvg width="40" height="40" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg"%3E%3Cg fill="white" fill-opacity="0.05"%3E%3Cpath d="M20 20c0-11.046-8.954-20-20-20v20h20zM0 20v20h20c0-11.046-8.954-20-20-20z"/%3E%3C/g%3E%3C/svg%3E") repeat',
              zIndex: 0,
            }
          }}
        >
          <Box sx={{ position: 'relative', zIndex: 1 }}>
            <Avatar
              sx={{
                width: 80,
                height: 80,
                margin: '0 auto 24px',
                background: 'rgba(255,255,255,0.2)',
                backdropFilter: 'blur(10px)',
              }}
            >
              <BrainIcon sx={{ fontSize: 40 }} />
            </Avatar>
            
            <Typography variant="h2" sx={{ fontWeight: 700, mb: 2, textShadow: '0 2px 4px rgba(0,0,0,0.3)' }}>
              AI-Powered Brain Metastasis Detection
            </Typography>
            
            <Typography variant="h5" sx={{ mb: 4, opacity: 0.9, maxWidth: 800, mx: 'auto' }}>
              Advanced machine learning technology for precise identification and analysis of brain metastases in T1-contrast enhanced MRI scans
            </Typography>
            
            <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', flexWrap: 'wrap' }}>
              <Chip 
                icon={<SecurityIcon />} 
                label="HIPAA Compliant" 
                sx={{ bgcolor: 'rgba(255,255,255,0.2)', color: 'white' }} 
              />
              <Chip 
                icon={<SpeedIcon />} 
                label="Fast Analysis" 
                sx={{ bgcolor: 'rgba(255,255,255,0.2)', color: 'white' }} 
              />
              <Chip 
                icon={<MedicalInformationIcon />} 
                label="Clinical Grade" 
                sx={{ bgcolor: 'rgba(255,255,255,0.2)', color: 'white' }} 
              />
            </Box>
          </Box>
        </Box>
      </Fade>

      {/* Features Section */}
      <Fade in={true} timeout={1500}>
        <Grid container spacing={4} sx={{ mb: 6 }}>
          <Grid item xs={12} md={4}>
            <Card 
              sx={{ 
                height: '100%', 
                textAlign: 'center', 
                transition: 'transform 0.3s, box-shadow 0.3s',
                '&:hover': {
                  transform: 'translateY(-8px)',
                  boxShadow: 6
                }
              }}
            >
              <CardContent sx={{ p: 4 }}>
                <Avatar
                  sx={{
                    width: 60,
                    height: 60,
                    margin: '0 auto 16px',
                    background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
                  }}
                >
                  <BrainIcon sx={{ fontSize: 30 }} />
                </Avatar>
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                  Advanced AI Detection
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  State-of-the-art UNETR (UNEt TRansformers) architecture trained on extensive medical datasets for precise metastasis identification
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={4}>
            <Card 
              sx={{ 
                height: '100%', 
                textAlign: 'center',
                transition: 'transform 0.3s, box-shadow 0.3s',
                '&:hover': {
                  transform: 'translateY(-8px)',
                  boxShadow: 6
                }
              }}
            >
              <CardContent sx={{ p: 4 }}>
                <Avatar
                  sx={{
                    width: 60,
                    height: 60,
                    margin: '0 auto 16px',
                    background: 'linear-gradient(45deg, #FF6B6B 30%, #FF8E8E 90%)',
                  }}
                >
                  <VisibilityIcon sx={{ fontSize: 30 }} />
                </Avatar>
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                  3D Visualization
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Interactive 3D visualization with multi-plane views, allowing detailed examination of detected metastases from all angles
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={4}>
            <Card 
              sx={{ 
                height: '100%', 
                textAlign: 'center',
                transition: 'transform 0.3s, box-shadow 0.3s',
                '&:hover': {
                  transform: 'translateY(-8px)',
                  boxShadow: 6
                }
              }}
            >
              <CardContent sx={{ p: 4 }}>
                <Avatar
                  sx={{
                    width: 60,
                    height: 60,
                    margin: '0 auto 16px',
                    background: 'linear-gradient(45deg, #4CAF50 30%, #81C784 90%)',
                  }}
                >
                  <CloudIcon sx={{ fontSize: 30 }} />
                </Avatar>
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                  Secure & Fast
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Cloud-based processing with enterprise-grade security, delivering results in minutes while maintaining patient data privacy
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Fade>

      {/* Upload Section */}
      <Fade in={true} timeout={2000}>
        <Paper 
          sx={{ 
            p: 4, 
            mb: 4,
            background: 'linear-gradient(145deg, #f8f9ff 0%, #ffffff 100%)',
            boxShadow: '0 8px 32px rgba(0,0,0,0.1)',
            borderRadius: 3
          }}
        >
          <Box sx={{ textAlign: 'center', mb: 4 }}>
            <Typography variant="h4" sx={{ fontWeight: 600, mb: 2, color: 'primary.main' }}>
              Upload Your MRI Scan
            </Typography>
            
            <Typography variant="body1" color="text.secondary" sx={{ maxWidth: 600, mx: 'auto' }}>
              Upload a T1 contrast-enhanced MRI scan in .npy, .nii, or .nii.gz format for brain metastasis segmentation and analysis.
            </Typography>
          </Box>

          {error && (
            <Alert severity="error" sx={{ mb: 3, borderRadius: 2 }}>
              {error}
            </Alert>
          )}

          <Box
            {...getRootProps()}
            sx={{
              border: '3px dashed',
              borderColor: isDragActive ? 'primary.main' : 'primary.light',
              borderRadius: 3,
              p: 6,
              textAlign: 'center',
              cursor: 'pointer',
              backgroundColor: isDragActive ? 'primary.50' : 'grey.50',
              transition: 'all 0.3s ease',
              mb: 3,
              '&:hover': {
                borderColor: 'primary.main',
                backgroundColor: 'primary.50',
                transform: 'scale(1.02)'
              }
            }}
          >
            <input {...getInputProps()} />
            <CloudUploadIcon sx={{ fontSize: 64, color: 'primary.main', mb: 2 }} />
            {isDragActive ? (
              <Typography variant="h6" color="primary.main">
                Drop the file here...
              </Typography>
            ) : (
              <>
                <Typography variant="h6" sx={{ mb: 1 }}>
                  Drag and drop your MRI scan here
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  or click to browse files (.npy, .nii, .nii.gz)
                </Typography>
              </>
            )}
            {file && (
              <Box mt={3}>
                <Alert 
                  severity="success" 
                  sx={{ 
                    borderRadius: 2,
                    '& .MuiAlert-message': { textAlign: 'left' }
                  }}
                >
                  <Typography variant="subtitle2">
                    Selected file: {file.name}
                  </Typography>
                  <Typography variant="body2">
                    Size: {(file.size / 1024 / 1024).toFixed(2)} MB
                  </Typography>
                </Alert>
              </Box>
            )}
          </Box>

          {processingMessage && (
            <Alert severity="info" sx={{ mb: 3, borderRadius: 2 }}>
              {processingMessage}
            </Alert>
          )}
          
          <Button
            variant="contained"
            onClick={handleUpload}
            disabled={!file || uploading}
            startIcon={uploading ? <CircularProgress size={20} color="inherit" /> : <BrainIcon />}
            size="large"
            sx={{
              width: '100%',
              py: 2,
              fontSize: '1.1rem',
              fontWeight: 600,
              borderRadius: 2,
              background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
              '&:hover': {
                background: 'linear-gradient(45deg, #1976D2 30%, #1BA3D3 90%)',
                transform: 'translateY(-2px)',
                boxShadow: 6
              },
              transition: 'all 0.3s ease'
            }}
          >
            {uploading ? 'Processing Your Scan...' : 'Start AI Analysis'}
          </Button>
          
          {uploading && (
            <Box sx={{ width: '100%', mt: 3 }}>
              <Typography variant="body2" color="text.secondary" align="center" sx={{ mb: 2 }}>
                Upload Progress: {uploadProgress}%
              </Typography>
              <LinearProgress 
                variant="determinate" 
                value={uploadProgress} 
                sx={{ 
                  height: 8, 
                  borderRadius: 4,
                  backgroundColor: 'grey.200',
                  '& .MuiLinearProgress-bar': {
                    borderRadius: 4,
                    background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)'
                  }
                }} 
              />
            </Box>
          )}
        </Paper>
      </Fade>

      {/* Information Section */}
      <Fade in={true} timeout={2500}>
        <Paper sx={{ p: 4, borderRadius: 3, bgcolor: 'grey.50' }}>
          <Typography variant="h5" sx={{ fontWeight: 600, mb: 3, textAlign: 'center' }}>
            How It Works
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={4}>
              <Box sx={{ textAlign: 'center' }}>
                <Avatar
                  sx={{
                    width: 50,
                    height: 50,
                    margin: '0 auto 16px',
                    bgcolor: 'primary.main',
                    fontSize: '1.5rem',
                    fontWeight: 'bold'
                  }}
                >
                  1
                </Avatar>
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
                  Upload
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Upload your T1-contrast enhanced MRI scan in supported format
                </Typography>
              </Box>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Box sx={{ textAlign: 'center' }}>
                <Avatar
                  sx={{
                    width: 50,
                    height: 50,
                    margin: '0 auto 16px',
                    bgcolor: 'secondary.main',
                    fontSize: '1.5rem',
                    fontWeight: 'bold'
                  }}
                >
                  2
                </Avatar>
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
                  Analyze
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Our AI model processes the scan to detect and segment metastases
                </Typography>
              </Box>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Box sx={{ textAlign: 'center' }}>
                <Avatar
                  sx={{
                    width: 50,
                    height: 50,
                    margin: '0 auto 16px',
                    bgcolor: 'success.main',
                    fontSize: '1.5rem',
                    fontWeight: 'bold'
                  }}
                >
                  3
                </Avatar>
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
                  Review
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  View detailed results with 3D visualization and quantitative analysis
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </Paper>
      </Fade>
    </Container>
  );
};

export default UploadForm;
