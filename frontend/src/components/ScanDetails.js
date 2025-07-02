import React, { useState, useEffect } from 'react';
import {
  Container,
  Paper,
  Typography,
  Box,
  Avatar,
  Divider,
  CircularProgress,
  Alert,
  Button,
  Card,
  CardContent,
  Grid,
  Chip,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Breadcrumbs,
  Link
} from '@mui/material';
import {
  ArrowBack,
  Psychology,
  Timeline,
  Assessment,
  Visibility,
  Download,
  Share,
  MedicalInformation,
  Schedule,
  FilePresent,
  CheckCircleOutline,
  InfoOutlined
} from '@mui/icons-material';
import { format } from 'date-fns';
import ResultViewer from './ResultViewer';
import api from '../services/api';

const ScanDetails = ({ jobId, onNavigateBack, results, status }) => {
  const [scanInfo, setScanInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchScanInfo = async () => {
      try {
        setLoading(true);
        // Fetch additional scan information if needed
        // For now, we'll use the results data that's passed in
        if (results) {
          setScanInfo(results);
        }
      } catch (err) {
        console.error('Error fetching scan info:', err);
        setError('Failed to load scan details. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    if (jobId) {
      fetchScanInfo();
    }
  }, [jobId, results]);

  // Format date for display
  const formatDate = (dateString) => {
    try {
      return format(new Date(dateString), 'MMM dd, yyyy - HH:mm');
    } catch (error) {
      return 'Unknown date';
    }
  };

  // Get status chip for scan
  const getStatusChip = (status) => {
    switch (status) {
      case 'completed':
        return <Chip icon={<CheckCircleOutline />} label="Completed" color="success" size="medium" />;
      case 'processing':
        return <Chip icon={<CircularProgress size={16} />} label="Processing" color="primary" size="medium" />;
      case 'failed':
        return <Chip icon={<InfoOutlined />} label="Failed" color="error" size="medium" />;
      default:
        return <Chip label={status || 'Unknown'} size="medium" />;
    }
  };

  if (loading) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Box sx={{ textAlign: 'center', py: 4 }}>
          <CircularProgress size={48} />
          <Typography variant="h6" sx={{ mt: 2 }}>
            Loading scan details...
          </Typography>
        </Box>
      </Container>
    );
  }

  if (error) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
        <Button
          variant="outlined"
          startIcon={<ArrowBack />}
          onClick={onNavigateBack}
        >
          Back to Profile
        </Button>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      {/* Breadcrumbs */}
      <Breadcrumbs sx={{ mb: 3 }}>
        <Link
          component="button"
          variant="body1"
          onClick={onNavigateBack}
          sx={{ textDecoration: 'none', cursor: 'pointer' }}
        >
          Profile
        </Link>
        <Typography color="text.primary">Scan Details</Typography>
      </Breadcrumbs>

      {/* Header */}
      <Paper sx={{ p: 3, mb: 3, background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', color: 'white' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <Avatar
            sx={{
              width: 60,
              height: 60,
              mr: 3,
              background: 'rgba(255,255,255,0.2)',
              backdropFilter: 'blur(10px)',
            }}
          >
            <Psychology sx={{ fontSize: 30 }} />
          </Avatar>
          <Box sx={{ flex: 1 }}>
            <Typography variant="h4" sx={{ fontWeight: 600, mb: 1 }}>
              Scan Analysis Details
            </Typography>
            <Typography variant="h6" sx={{ opacity: 0.9 }}>
              Job ID: {jobId}
            </Typography>
          </Box>
          <Box sx={{ textAlign: 'right' }}>
            {getStatusChip(status)}
          </Box>
        </Box>
        
        <Button
          variant="outlined"
          startIcon={<ArrowBack />}
          onClick={onNavigateBack}
          sx={{ 
            color: 'white', 
            borderColor: 'rgba(255,255,255,0.5)',
            '&:hover': {
              borderColor: 'white',
              backgroundColor: 'rgba(255,255,255,0.1)'
            }
          }}
        >
          Back to Profile
        </Button>
      </Paper>

      <Grid container spacing={3}>
        {/* Scan Information */}
        <Grid item xs={12} md={4}>
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2, display: 'flex', alignItems: 'center' }}>
                <MedicalInformation sx={{ mr: 1, color: 'primary.main' }} />
                Scan Information
              </Typography>
              
              <List disablePadding>
                <ListItem sx={{ px: 0, py: 1 }}>
                  <ListItemIcon>
                    <FilePresent color="primary" />
                  </ListItemIcon>
                  <ListItemText 
                    primary="File Name" 
                    secondary={results?.file_name || 'Unknown'} 
                  />
                </ListItem>
                
                <ListItem sx={{ px: 0, py: 1 }}>
                  <ListItemIcon>
                    <Schedule color="primary" />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Processed Date" 
                    secondary={results?.created_at ? formatDate(results.created_at) : 'Unknown'} 
                  />
                </ListItem>
                
                <ListItem sx={{ px: 0, py: 1 }}>
                  <ListItemIcon>
                    <Assessment color="primary" />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Status" 
                    secondary={status || 'Unknown'} 
                  />
                </ListItem>
              </List>
            </CardContent>
          </Card>

          {/* Analysis Results Summary */}
          {status === 'completed' && results && (
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 2, display: 'flex', alignItems: 'center' }}>
                  <Timeline sx={{ mr: 1, color: 'primary.main' }} />
                  Analysis Summary
                </Typography>
                
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Metastases Detected
                  </Typography>
                  <Typography variant="h3" color="primary.main" sx={{ fontWeight: 600 }}>
                    {results.metastasis_count || 0}
                  </Typography>
                </Box>
                
                {results.total_volume && (
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Total Volume
                    </Typography>
                    <Typography variant="h5" color="secondary.main" sx={{ fontWeight: 600 }}>
                      {results.total_volume.toFixed(2)} mm³
                    </Typography>
                  </Box>
                )}
                
                {results.metastasis_volumes && results.metastasis_volumes.length > 0 && (
                  <Box>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Individual Volumes
                    </Typography>
                    <Box sx={{ maxHeight: 150, overflow: 'auto' }}>
                      {results.metastasis_volumes.map((volume, index) => (
                        <Chip
                          key={index}
                          label={`${volume.toFixed(2)} mm³`}
                          size="small"
                          sx={{ mr: 1, mb: 1 }}
                        />
                      ))}
                    </Box>
                  </Box>
                )}
              </CardContent>
            </Card>
          )}

          {/* Actions */}
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>
                Actions
              </Typography>
              
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                <Button
                  variant="outlined"
                  startIcon={<Download />}
                  fullWidth
                  onClick={() => {
                    // TODO: Implement download functionality
                    alert('Download functionality coming soon!');
                  }}
                >
                  Download Report
                </Button>
                
                <Button
                  variant="outlined"
                  startIcon={<Share />}
                  fullWidth
                  onClick={() => {
                    // TODO: Implement share functionality
                    alert('Share functionality coming soon!');
                  }}
                >
                  Share Results
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Visualization */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2, display: 'flex', alignItems: 'center' }}>
                <Visibility sx={{ mr: 1, color: 'primary.main' }} />
                3D Visualization & Analysis
              </Typography>
              
              <Divider sx={{ mb: 3 }} />
              
              <ResultViewer 
                jobId={jobId} 
                status={status} 
                results={results}
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  );
};

export default ScanDetails;
