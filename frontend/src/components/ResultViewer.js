import React, { useState, useEffect, useCallback } from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  CircularProgress, 
  Grid,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Divider,
  Tabs,
  Tab,
  Button,
  Stack
} from '@mui/material';
import { Doughnut } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip as ChartTooltip, Legend } from 'chart.js';
import Visualization3DPlaceholder from './Visualization3DPlaceholder';
import VisualizationControls from './VisualizationControls';
import { PictureAsPdf, TableChart } from '@mui/icons-material';
import api from '../services/api';

// Register Chart.js components
ChartJS.register(ArcElement, ChartTooltip, Legend);

const ResultViewer = ({ jobId, status, results }) => {
  const [tabValue, setTabValue] = useState(0);
  const [exporting, setExporting] = useState(false);
  
  // Visualization settings
  const [vizType, setVizType] = useState('slice');
  const [sliceIndex, setSliceIndex] = useState(50); // Start with middle slice = 50
  const [numSlices, setNumSlices] = useState(5);
  const [vizQuality, setVizQuality] = useState('high');
  const [enhanceContrast, setEnhanceContrast] = useState(true);
  const [enhanceEdges, setEnhanceEdges] = useState(true);
  const [upscaleFactor, setUpscaleFactor] = useState(1.2);
  const [maxSliceIndex, setMaxSliceIndex] = useState(100); // Default, updated later
  const [sliceInitialized, setSliceInitialized] = useState(false);
  
  // Additional state for 2D view types (axial, coronal, sagittal)
  const [viewType, setViewType] = useState('axial');
  // Additional state for three-plane view
  const [axialSliceIndex, setAxialSliceIndex] = useState(50);
  const [coronalSliceIndex, setCoronalSliceIndex] = useState(50);
  const [sagittalSliceIndex, setSagittalSliceIndex] = useState(50);
  
  // Utility function to ensure we have a valid slice index
  const ensureValidSlice = useCallback((slice) => {
    if (slice === null || slice === undefined || isNaN(slice)) {
      // Default to middle slice based on maxSliceIndex
      return Math.floor(maxSliceIndex / 2);
    }
    
    // Ensure slice is within bounds
    return Math.max(0, Math.min(slice, maxSliceIndex));
  }, [maxSliceIndex]);

  // Memoize the visualization URL generation to prevent unnecessary re-renders
  const getVisualizationUrl = useCallback(() => {
    // Only generate URL when we have a valid jobId, sliceIndex, and status is completed
    if (!jobId || status !== 'completed') return '#';
    
    // Use the current slice index with validation
    const currentSlice = ensureValidSlice(sliceIndex);
    
    // Choose the appropriate URL builder based on visualization type
    if (vizType === 'side-by-side') {
      return api.getSideBySideUrl(jobId, {
        slice: currentSlice,
        viewType,
        upscale: upscaleFactor,
        enhanceContrast,
        enhanceEdges
      });
    } else if (vizType === 'three-plane') {
      return api.getThreePlaneUrl(jobId, {
        axialSlice: axialSliceIndex,
        coronalSlice: coronalSliceIndex,
        sagittalSlice: sagittalSliceIndex,
        enhanceContrast,
        enhanceEdges
      });
    } else {
      // Default visualization types (slice, multi-slice, projection, lesions)
      return api.getVisualizationUrl(jobId, {
        type: vizType,
        quality: "high", // Force high quality
        slice: currentSlice,
        viewType, // Include viewType for slice view
        upscale: upscaleFactor,
        enhance_contrast: enhanceContrast,
        enhance_edges: enhanceEdges,
        brightness: 1.4,
        contrast: 1.6,
        show_original: true // Show original scan if no segmentation data
      }, results?.segmentation_path);
    }
  }, [jobId, status, vizType, sliceIndex, enhanceContrast, enhanceEdges, upscaleFactor, 
      results?.segmentation_path, ensureValidSlice, viewType, axialSliceIndex, coronalSliceIndex, sagittalSliceIndex]);
  
  // Log debug information when props change - moved to top level to avoid hook rules violation
  useEffect(() => {
    // Only log debugging info when we have completed status
    if (status === 'completed' && jobId && results) {
      console.log("Debug - Component rendering with:", {
        jobId,
        status,
        sliceIndex: ensureValidSlice(sliceIndex),
        sliceInitialized,
        hasSegmentationPath: !!results?.segmentation_path
      });
      
      // Only log URL when we have results
      if (getVisualizationUrl() !== '#') {
        console.log("Visualization URL:", getVisualizationUrl());
      }
    }
  }, [jobId, status, sliceIndex, sliceInitialized, results, ensureValidSlice, getVisualizationUrl]);
  
  // Add a button to manually test the API with useCallback to avoid dependency issues
  const testResultsApi = React.useCallback(async () => {
    if (!jobId) return;
    
    try {
      console.log('Testing getResults API call...');
      const response = await api.getResults(jobId);
      console.log('Raw results response:', response);
      console.log('Results data:', response.data);
      console.log('segmentation_path present:', !!response.data?.segmentation_path);
      
      if (response.data?.segmentation_path) {
        console.log('Segmentation path from response:', response.data.segmentation_path);
      }
      
      // Test the visualization URL directly - ensure we're using the same slice index everywhere
      // Make sure we're using the actual current sliceIndex, not the initial value
      const currentSliceIndex = ensureValidSlice(sliceIndex);
      const vizUrl = api.getVisualizationUrl(jobId, {
        type: vizType,
        quality: vizQuality,
        slice: currentSliceIndex
      }, response.data?.segmentation_path); // Use segmentation_path if available
      console.log('Full visualization URL:', window.location.origin + vizUrl);
      
      // Check if endpoints exist in API gateway
      console.log('Checking API endpoints');
      
      // Check both standard and advanced visualization endpoints
      // Always use the current sliceIndex for consistency
      const standardVizUrl = `/visualization/${jobId}?type=${vizType}&quality=standard&slice_idx=${currentSliceIndex}`;
      const advancedVizUrl = `/advanced-visualization/${jobId}?type=${vizType}&quality=high&slice_idx=${currentSliceIndex}`;
      
      // Get auth token from localStorage for fetch requests
      const token = localStorage.getItem('token');
      const authHeaders = token ? { Authorization: `Bearer ${token}` } : {};

      // Test standard endpoint
      try {
        const standardResponse = await fetch(standardVizUrl, {
          headers: authHeaders
        });
        console.log('Standard visualization status:', standardResponse.status);
        if (!standardResponse.ok) {
          console.error('Standard endpoint not available');
        } else {
          console.log('Standard visualization endpoint works!');
        }
      } catch (e) {
        console.error('Error accessing standard visualization:', e);
      }
      
      // Test advanced endpoint
      try {
        const advancedResponse = await fetch(advancedVizUrl, {
          headers: authHeaders
        });
        console.log('Advanced visualization status:', advancedResponse.status);
        if (!advancedResponse.ok) {
          console.error('Advanced endpoint not available');
        } else {
          console.log('Advanced visualization endpoint works!');
        }
      } catch (e) {
        console.error('Error accessing advanced visualization:', e);
      }
      
      // Test API service endpoint
      console.log('Testing API service generated URL:', vizUrl);
      try {
        const apiResponse = await fetch(vizUrl, {
          headers: authHeaders,
          // Add a timeout to handle potential long-running requests
          signal: AbortSignal.timeout(30000) // 30 second timeout
        });
        console.log('Visualization endpoint response status:', apiResponse.status);
        if (!apiResponse.ok) {
          console.error('Failed to fetch visualization');
          return;
        }
        
        const blob = await apiResponse.blob();
        console.log('Visualization endpoint response type:', blob.type);
        if (blob.type.startsWith('image/')) {
          console.log('Successfully fetched visualization image');
        } else {
          console.error('Response is not an image');
        }
      } catch (error) {
        console.error('Error fetching visualization:', error);
      }
    } catch (error) {
      console.error('Error testing getResults:', error);
    }
  }, [jobId, vizType, vizQuality, sliceIndex, ensureValidSlice]);
  
  // Monitor when results change
  useEffect(() => {
    if (results) {
      console.log("Results updated:", results);
      console.log("Results contains segmentation_path:", results.segmentation_path ? "Yes" : "No");
    }
  }, [results]);

  // Note: sliceInitialized state is now declared at the top of the component
  
  // Monitor slice index changes to debug when and why they might reset
  useEffect(() => {
    // Only log and update session storage when the slice is explicitly changed by user
    // not during the initialization process
    if (sliceInitialized) {
      console.log("Slice index changed to:", sliceIndex);
      // Store the current slice index in session storage to maintain it across API calls
      if (sliceIndex !== null && !isNaN(sliceIndex)) {
        // Store the validated slice index to ensure it's always in bounds
        const validatedSlice = ensureValidSlice(sliceIndex);
        sessionStorage.setItem(`slice_index_${jobId}`, validatedSlice.toString());
      }
    }
  }, [sliceIndex, jobId, sliceInitialized, ensureValidSlice]);
  
  // Function to fetch volume dimensions from the backend
  // Function has been moved inside the useEffect below
  
  // Fetch volume dimensions when jobId changes or status becomes completed
  useEffect(() => {
    const fetchDimensions = async () => {
      try {
        console.log("Fetching volume dimensions for job ID:", jobId);
        const response = await api.getVolumeDimensions(jobId);
        console.log("Volume dimensions response:", response);
        
        if (response.status === 200) {
          const data = response.data;
          console.log("Volume dimensions data:", data);
          
          if (data.dimensions && data.dimensions.length > 0) {
            // Update max slice index based on z-dimension (depth)
            const newMaxSliceIndex = data.dimensions[0] - 1; // 0-based index
            console.log("Setting maxSliceIndex to:", newMaxSliceIndex);
            setMaxSliceIndex(newMaxSliceIndex);
            
            // Only initialize slice index if not already done
            if (!sliceInitialized) {
              // Check for stored slice index in session storage first
              const storedSliceIndex = sessionStorage.getItem(`slice_index_${jobId}`);
              
              if (storedSliceIndex) {
                // Restore from session storage if available
                const parsedIndex = parseInt(storedSliceIndex, 10);
                console.log("Restoring sliceIndex from session storage:", parsedIndex);
                // Ensure the stored index is within the valid range for this volume
                const validIndex = Math.min(parsedIndex, newMaxSliceIndex);
                setSliceIndex(validIndex);
              } else {
                // No stored value, use middle slice
                const middleSlice = Math.floor(newMaxSliceIndex / 2);
                console.log("Initializing sliceIndex to middle slice:", middleSlice);
                setSliceIndex(middleSlice);
              }
              
              // Mark as initialized to prevent further automatic updates
              setSliceInitialized(true);
            }
          }
        } else {
          // If the endpoint doesn't exist or fails, use volume stats from results
          if (results && results.volume_dimensions) {
            const newMaxSliceIndex = results.volume_dimensions[0] - 1;
            console.log("Using results volume_dimensions, setting maxSliceIndex to:", newMaxSliceIndex);
            setMaxSliceIndex(newMaxSliceIndex);
            
            // Only set initial slice if not already initialized
            if (!sliceInitialized) {
              // Check storage first
              const storedSliceIndex = sessionStorage.getItem(`slice_index_${jobId}`);
              if (storedSliceIndex) {
                const parsedIndex = parseInt(storedSliceIndex, 10);
                const validIndex = Math.min(parsedIndex, newMaxSliceIndex);
                setSliceIndex(validIndex);
              } else {
                setSliceIndex(Math.floor(newMaxSliceIndex / 2));
              }
              setSliceInitialized(true);
            }
          }
        }
      } catch (error) {
        console.error('Error fetching volume dimensions:', error);
        // Use a reasonable default based on typical MRI dimensions
        console.log("Using default maxSliceIndex of 128 due to error");
        setMaxSliceIndex(128);
        
        // Only set default slice index if not already initialized
        if (!sliceInitialized) {
          const defaultMiddleSlice = 64; // Half of default maxSliceIndex
          console.log("Setting default sliceIndex to:", defaultMiddleSlice);
          setSliceIndex(defaultMiddleSlice);
          setSliceInitialized(true);
        }
      }
    };
    
    if (jobId && status === 'completed') {
      fetchDimensions();
      // Only test API when job completes for initial load and only once
      if (!sliceInitialized) {
        // Removed direct testResultsApi call here to prevent infinite loop
      }
    }
  }, [jobId, status, results, sliceInitialized]);

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };
  
  // Handle exporting results
  const handleExport = async (format) => {
    try {
      setExporting(true);
      
      let response;
      if (format === 'pdf') {
        response = await api.exportResultsPDF(jobId);
      } else if (format === 'csv') {
        response = await api.exportResultsCSV(jobId);
      }
      
      // Create a download link for the blob
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `metastasis_results_${jobId}.${format}`);
      document.body.appendChild(link);
      link.click();
      
      // Clean up
      link.parentNode.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error(`Error exporting results as ${format}:`, error);
    } finally {
      setExporting(false);
    }
  };

  // For pending status
  if (status !== 'completed') {
    // Handle error states specially
    if (status === 'not_found' || status === 'failed' || status === 'error') {
      return (
        <Paper sx={{ p: 3, textAlign: 'center' }}>
          <Typography variant="h6" color="error" sx={{ mb: 2 }}>
            {status === 'not_found' ? 'Job Not Found' : 'Processing Failed'}
          </Typography>
          <Typography variant="body1">
            {status === 'not_found' 
              ? 'The requested scan could not be found. It may have been deleted or expired.'
              : 'There was an error processing this scan. Please try again or contact support.'
            }
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
            Job ID: {jobId}
          </Typography>
        </Paper>
      );
    }
    
    // Regular pending/processing state
    return (
      <Paper sx={{ p: 3, textAlign: 'center' }}>
        <CircularProgress sx={{ mb: 2 }} />
        <Typography variant="h6">
          {status === 'processing' ? 'Processing MRI Scan...' : 'Analyzing segmentation...'}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          This may take a few minutes. Please wait.
        </Typography>
      </Paper>
    );
  }

  // For completed status
  const { metastasis_count, metastasis_volumes, total_volume } = results || {};
  
  // Prepare chart data
  const chartData = {
    labels: metastasis_volumes.map((_, index) => `Metastasis ${index + 1}`),
    datasets: [
      {
        data: metastasis_volumes,
        backgroundColor: [
          '#FF6384',
          '#36A2EB',
          '#FFCE56',
          '#4BC0C0',
          '#9966FF',
          '#FF9F40',
          '#8AC926',
          '#1982C4',
          '#6A4C93',
          '#E76F51'
        ],
        hoverBackgroundColor: [
          '#FF6384',
          '#36A2EB',
          '#FFCE56',
          '#4BC0C0',
          '#9966FF',
          '#FF9F40',
          '#8AC926',
          '#1982C4',
          '#6A4C93',
          '#E76F51'
        ],
      },
    ],
  };

  // Add debugging for segmentation_path and visualization params
  console.log("Debug - Results object:", results);
  console.log("Debug - Segmentation path exists:", results?.segmentation_path ? "Yes" : "No");
  console.log("Debug - Current visualization settings:", { 
    vizType, vizQuality, sliceIndex, 
    upscaleFactor, enhanceContrast, enhanceEdges, 
    maxSliceIndex
  });
  
  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Metastasis Detection Results
      </Typography>
      
      <Divider sx={{ mb: 3 }} />
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Summary
              </Typography>
              
              <TableContainer component={Paper}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Metastasis Type</TableCell>
                      <TableCell align="right">Count</TableCell>
                      <TableCell align="right">Total Volume (cm³)</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    <TableRow>
                      <TableCell>Detected Metastases</TableCell>
                      <TableCell align="right">{metastasis_count || 0}</TableCell>
                      <TableCell align="right">{total_volume?.toFixed(2) || 0}</TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </TableContainer>
              
              <Divider sx={{ my: 2 }} />
              
              <Typography variant="subtitle1" gutterBottom>
                Volume Breakdown
              </Typography>
              
              <Doughnut data={chartData} />
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Actions
              </Typography>
              
              <Stack spacing={2}>
                <Button 
                  variant="contained" 
                  color="primary" 
                  onClick={() => handleExport('pdf')}
                  disabled={exporting}
                  startIcon={<PictureAsPdf />}
                >
                  {exporting ? 'Exporting as PDF...' : 'Export Results as PDF'}
                </Button>
                
                <Button 
                  variant="contained" 
                  color="secondary" 
                  onClick={() => handleExport('csv')}
                  disabled={exporting}
                  startIcon={<TableChart />}
                >
                  {exporting ? 'Exporting as CSV...' : 'Export Results as CSV'}
                </Button>
                
                <Button 
                  variant="outlined" 
                  color="primary" 
                  onClick={testResultsApi}
                >
                  Test Results API
                </Button>
              </Stack>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      
      <Divider sx={{ my: 4 }} />
      
      <Typography variant="h5" gutterBottom>
        Detailed Results
      </Typography>
      
      <Tabs 
        value={tabValue} 
        onChange={handleTabChange} 
        variant="scrollable" 
        scrollButtons="auto"
        sx={{ mb: 3 }}
      >
        <Tab label="3D Visualization" />
        <Tab label="Slice View" />
        <Tab label="Tabular Data" />
      </Tabs>
      
      {tabValue === 0 && (
        <Visualization3DPlaceholder jobId={jobId} />
      )}
      
      {tabValue === 1 && (
        <Box>
          <Typography variant="subtitle1" gutterBottom>
            Adjust the settings below to configure the slice view.
          </Typography>
          
          <VisualizationControls 
            vizType={vizType}
            setVizType={setVizType}
            sliceIndex={sliceIndex}
            setSliceIndex={setSliceIndex}
            numSlices={numSlices}
            setNumSlices={setNumSlices}
            vizQuality={vizQuality}
            setVizQuality={setVizQuality}
            enhanceContrast={enhanceContrast}
            setEnhanceContrast={setEnhanceContrast}
            enhanceEdges={enhanceEdges}
            setEnhanceEdges={setEnhanceEdges}
            upscaleFactor={upscaleFactor}
            setUpscaleFactor={setUpscaleFactor}
            maxSliceIndex={maxSliceIndex}
            viewType={viewType}
            setViewType={setViewType}
          />
          
          <Divider sx={{ my: 2 }} />
          
          <Typography variant="h6" gutterBottom>
            Slice View
          </Typography>
          
          {/* Render the slice view based on selected settings */}
          <Box 
            sx={{ 
              position: 'relative', 
              width: '100%', 
              height: 500, 
              bgcolor: '#000', // Dark background to improve contrast
              borderRadius: 2,
              overflow: 'hidden',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center'
            }}
          >
            {/* Check if metastasis count is zero and display a message */}
            {metastasis_count === 0 ? (
              <Box sx={{ textAlign: 'center', color: 'white', p: 3 }}>
                <Typography variant="h6" sx={{ mb: 2 }}>
                  No metastases detected in this scan
                </Typography>
                <Typography variant="body1">
                  The analysis did not find any metastatic regions in this scan.
                </Typography>
                <Typography variant="body2" sx={{ mt: 2, color: 'gray' }}>
                  You can still browse through the different slices to inspect the scan data.
                </Typography>
              </Box>
            ) : (
              <>
                {/* Loading indicator */}
                <CircularProgress 
                  size={40}
                  thickness={5}
                  sx={{ 
                    position: 'absolute', 
                    top: '50%', 
                    left: '50%', 
                    marginTop: -3, 
                    marginLeft: -3,
                    color: '#fff',
                    zIndex: 2
                  }} 
                />
                
                {/* Enhanced visualization with additional params */}
                {/* Use memoized source URL to prevent repeated API calls */}
                <img 
                  src={getVisualizationUrl()}
                  alt="Slice View"
                  style={{ 
                    width: '100%', 
                    height: '100%', 
                    objectFit: 'contain',
                    backgroundColor: '#000',
                    filter: 'brightness(110%)' // Subtle brightness enhancement without affecting colors
                  }}
                  onLoad={(e) => {
                    console.log(`Image loaded: ${e.target.naturalWidth}x${e.target.naturalHeight}`);
                    // Hide the spinner when image loads
                    const spinners = document.querySelectorAll('.MuiCircularProgress-root');
                    spinners.forEach(spinner => {
                      spinner.style.display = 'none';
                    });
                    
                    // If the image is extremely small, it might be empty
                    if (e.target.naturalWidth < 10 || e.target.naturalHeight < 10) {
                      console.warn("Image dimensions are very small - might be empty");
                    }
                  }}
                  onError={(e) => {
                    console.error('Error loading visualization');
                    
                    // Only try the fallback once (set a flag on the element)
                    if (!e.target.dataset.fallbackAttempted) {
                      e.target.dataset.fallbackAttempted = 'true';
                      console.log('Retrying with fallback URL');
                      // Try showing the original image without segmentation
                      const fallbackUrl = `/visualization/${jobId}?quality=high&upscale=2&enhance_contrast=true&enhance_edges=true&type=${vizType}&slice_idx=${ensureValidSlice(sliceIndex)}&show_original=true&brightness=1.5&contrast=1.8`;
                      e.target.src = fallbackUrl;
                    } else {
                      // Hide spinner on failure
                      const spinners = document.querySelectorAll('.MuiCircularProgress-root');
                      spinners.forEach(spinner => {
                        spinner.style.display = 'none';
                      });
                      
                      // Display an error placeholder instead of endless retries
                      e.target.src = 'data:image/svg+xml;charset=UTF-8,<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>';
                      e.target.style.padding = '100px';
                    }
                  }}
                />
              </>
            )}
            
            {/* Slice indicator - show even when no metastases */}
            <Typography 
              variant="caption" 
              sx={{
                position: 'absolute',
                bottom: 10,
                right: 10,
                color: 'white',
                backgroundColor: 'rgba(0,0,0,0.5)',
                padding: '2px 6px',
                borderRadius: 1
              }}
            >
              Slice: {sliceIndex} / {maxSliceIndex}
            </Typography>
            
            {/* Additional scan info */}
            <Typography 
              variant="caption" 
              sx={{
                position: 'absolute',
                bottom: 10,
                left: 10,
                color: 'white',
                backgroundColor: 'rgba(0,0,0,0.5)',
                padding: '2px 6px',
                borderRadius: 1
              }}
            >
              Job ID: {jobId.slice(0,8)}...
            </Typography>
          </Box>
        </Box>
      )}
      
      {tabValue === 2 && (
        <Paper sx={{ p: 3 }}>
          <Typography variant="subtitle1" gutterBottom>
            Metastasis Segmentation Results
          </Typography>
          
          <Divider sx={{ my: 2 }} />
          
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Metastasis Type</TableCell>
                  <TableCell align="right">Count</TableCell>
                  <TableCell align="right">Total Volume (cm³)</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell>Detected Metastases</TableCell>
                  <TableCell align="right">{metastasis_count || 0}</TableCell>
                  <TableCell align="right">{total_volume?.toFixed(2) || 0}</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
      )}
    </Box>
  );
};

export default ResultViewer;
