import React, { useEffect, useState } from 'react';
import { Box, Paper, Typography, Button, CircularProgress, Slider } from '@mui/material';
import { Info, Fullscreen, FullscreenExit } from '@mui/icons-material';

/**
 * Enhanced placeholder component for 3D Brain and Metastasis Visualization
 * This version shows a real visualization based on actual segmentation data
 * 
 * @param {Object} props
 * @param {string} props.jobId - Job ID
 * @param {Array} props.metastases - Array of metastasis objects with real position and volume data
 * @param {boolean} props.loading - Whether the 3D data is still loading
 */
const Visualization3DPlaceholder = ({ jobId, metastases = [], loading: externalLoading = false }) => {
  const [internalLoading, setInternalLoading] = useState(false);
  const [fullscreen, setFullscreen] = useState(false);
  const [viewAngle, setViewAngle] = useState(0);
  const [rotationActive, setRotationActive] = useState(false);
  
  // Combined loading state
  const loading = externalLoading || internalLoading;
  
  // Function to toggle fullscreen mode
  const handleFullscreen = () => {
    setFullscreen(!fullscreen);
  };
  
  // Simulates a loading state for the visualization (only when not externally loading)
  useEffect(() => {
    if (!externalLoading) {
      setInternalLoading(true);
      const timer = setTimeout(() => {
        setInternalLoading(false);
      }, 1500);
      return () => clearTimeout(timer);
    }
  }, [externalLoading]);
  
  // Auto-rotation effect
  useEffect(() => {
    if (rotationActive) {
      const intervalId = setInterval(() => {
        setViewAngle(angle => (angle + 1) % 360);
      }, 100);
      return () => clearInterval(intervalId);
    }
  }, [rotationActive]);
  
  // Handle view angle change
  const handleViewAngleChange = (event, newValue) => {
    setViewAngle(newValue);
  };
  
  // Toggle auto-rotation
  const toggleRotation = () => {
    setRotationActive(!rotationActive);
  };
  
  return (
    <Box sx={{ 
      height: fullscreen ? '100vh' : '100%', 
      width: fullscreen ? '100vw' : '100%',
      position: fullscreen ? 'fixed' : 'relative',
      top: fullscreen ? 0 : 'auto',
      left: fullscreen ? 0 : 'auto',
      zIndex: fullscreen ? 9999 : 1,
      bgcolor: fullscreen ? 'background.paper' : 'transparent',
      display: 'flex', 
      flexDirection: 'column'
    }}>
      <Paper elevation={2} sx={{ 
        p: 2, 
        mb: 2,
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <Typography variant="h6">
          3D Visualization
        </Typography>
        <Button 
          variant="outlined" 
          size="small"
          onClick={handleFullscreen}
          startIcon={fullscreen ? <FullscreenExit /> : <Fullscreen />}
        >
          {fullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
        </Button>
      </Paper>
      
      <Box 
        sx={{ 
          flexGrow: 1, 
          display: 'flex', 
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          bgcolor: '#f5f5f5', 
          borderRadius: 1,
          p: 3,
          position: 'relative',
          overflow: 'hidden'
        }}
      >
        {loading ? (
          <CircularProgress />
        ) : metastases.length > 0 ? (
          <>
            <Box
              sx={{
                width: '100%',
                height: '100%',
                position: 'relative',
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                perspective: '1200px',
                backgroundColor: '#000', // Dark medical background
                borderRadius: 2,
                overflow: 'hidden'
              }}
            >
              {/* Brain volume representation - more realistic medical visualization */}
              <Box
                sx={{
                  width: '300px',
                  height: '280px',
                  position: 'absolute',
                  transform: `rotateY(${viewAngle}deg) rotateX(-10deg)`,
                  transition: 'transform 0.1s',
                  transformStyle: 'preserve-3d',
                }}
              >
                {/* Brain tissue - multiple layers for depth */}
                {[0, 1, 2].map((layer) => (
                  <Box
                    key={`brain-layer-${layer}`}
                    sx={{
                      width: `${300 - layer * 20}px`,
                      height: `${280 - layer * 18}px`,
                      position: 'absolute',
                      left: `${layer * 10}px`,
                      top: `${layer * 9}px`,
                      borderRadius: '45% 55% 52% 48% / 48% 45% 55% 52%', // Irregular brain-like shape
                      border: `2px solid rgba(180, 180, 180, ${0.3 - layer * 0.05})`,
                      background: `radial-gradient(ellipse at 35% 40%, 
                        rgba(120, 120, 120, ${0.15 - layer * 0.03}) 0%, 
                        rgba(90, 90, 90, ${0.08 - layer * 0.02}) 60%, 
                        rgba(60, 60, 60, ${0.04 - layer * 0.01}) 100%)`,
                      transform: `translateZ(${layer * 15}px)`,
                    }}
                  />
                ))}
                
                {/* Render metastases at their real 3D positions */}
                {metastases.map((met, index) => {
                  // Use the real normalized positions from the segmentation
                  const x = met.position[0] * 120; // Scale to fit within brain outline
                  const y = met.position[1] * 100; // Scale Y a bit less for brain shape
                  const z = met.position[2] * 50;  // Z depth for 3D effect
                  
                  // Size based on volume - medical scaling
                  const baseSize = Math.max(4, Math.min(25, Math.cbrt(met.volume) * 0.8));
                  const size = baseSize + (z * 0.1); // Slight size variation based on depth
                  
                  // Medical colors - red for metastases (like in your Napari setup)
                  const intensity = Math.min(1, met.volume / 1000); // Intensity based on size
                  const red = Math.floor(255 * (0.7 + intensity * 0.3));
                  const green = Math.floor(50 * (1 - intensity * 0.5));
                  const blue = Math.floor(50 * (1 - intensity * 0.5));
                  const alpha = 0.85 + (intensity * 0.15);
                  
                  return (
                    <Box
                      key={met.id}
                      sx={{
                        width: `${size}px`,
                        height: `${size}px`,
                        borderRadius: '50%',
                        position: 'absolute',
                        left: `${150 + x}px`, // Center + offset
                        top: `${140 + y}px`,  // Center + offset
                        backgroundColor: `rgba(${red}, ${green}, ${blue}, ${alpha})`,
                        transform: `translateZ(${z}px) scale(${1 + z * 0.01})`,
                        boxShadow: `0 0 ${size * 0.8}px rgba(${red}, ${green}, ${blue}, 0.6)`,
                        border: `1px solid rgba(${red + 50}, ${green + 20}, ${blue + 20}, 0.8)`,
                        zIndex: Math.floor(z + 100),
                        transition: 'all 0.1s ease-out',
                        '&::before': {
                          content: '""',
                          position: 'absolute',
                          width: '100%',
                          height: '100%',
                          borderRadius: '50%',
                          background: `radial-gradient(circle at 30% 30%, 
                            rgba(255, 255, 255, 0.4) 0%, 
                            transparent 70%)`,
                          pointerEvents: 'none',
                        }
                      }}
                      title={`Metastasis ${met.id}: ${met.volume.toFixed(1)}mm³`}
                    />
                  );
                })}
                
                {/* Add coordinate system reference lines */}
                <Box
                  sx={{
                    position: 'absolute',
                    width: '2px',
                    height: '200px',
                    backgroundColor: 'rgba(100, 150, 255, 0.3)',
                    left: '149px',
                    top: '40px',
                    transform: 'translateZ(0px)',
                  }}
                />
                <Box
                  sx={{
                    position: 'absolute',
                    width: '200px',
                    height: '2px',
                    backgroundColor: 'rgba(100, 255, 150, 0.3)',
                    left: '50px',
                    top: '139px',
                    transform: 'translateZ(0px)',
                  }}
                />
              </Box>
              
              {/* Volume and count info overlay */}
              <Box
                sx={{
                  position: 'absolute',
                  top: 16,
                  left: 16,
                  backgroundColor: 'rgba(0, 0, 0, 0.7)',
                  color: 'white',
                  padding: '8px 12px',
                  borderRadius: 1,
                  fontSize: '0.875rem',
                  fontFamily: 'monospace',
                }}
              >
                <Typography variant="caption" display="block">
                  Metastases: {metastases.length}
                </Typography>
                <Typography variant="caption" display="block">
                  Total Volume: {metastases.reduce((sum, m) => sum + m.volume, 0).toFixed(1)} mm³
                </Typography>
                <Typography variant="caption" display="block">
                  Largest: {Math.max(...metastases.map(m => m.volume)).toFixed(1)} mm³
                </Typography>
              </Box>
              
              {/* Medical view controls */}
              <Box
                sx={{
                  position: 'absolute',
                  bottom: 16,
                  right: 16,
                  backgroundColor: 'rgba(0, 0, 0, 0.7)',
                  color: 'white',
                  padding: '4px 8px',
                  borderRadius: 1,
                  fontSize: '0.75rem',
                }}
              >
                Medical 3D View
              </Box>
            </Box>
            
            <Box sx={{ 
              width: '100%', 
              mt: 2, 
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center'
            }}>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                3D Brain Visualization - Real Positions from Segmentation
              </Typography>
              <Slider
                value={viewAngle}
                onChange={handleViewAngleChange}
                min={0}
                max={359}
                sx={{ 
                  width: '80%', 
                  mb: 2,
                  '& .MuiSlider-thumb': {
                    backgroundColor: '#ff4444',
                  },
                  '& .MuiSlider-track': {
                    backgroundColor: '#ff4444',
                  }
                }}
                disabled={rotationActive}
              />
              
              <Button
                variant="contained"
                color={rotationActive ? "secondary" : "primary"}
                onClick={toggleRotation}
                size="small"
                sx={{ 
                  backgroundColor: rotationActive ? '#ff4444' : '#2196F3',
                  '&:hover': {
                    backgroundColor: rotationActive ? '#cc3333' : '#1976D2',
                  }
                }}
              >
                {rotationActive ? "Stop Rotation" : "Auto-Rotate"}
              </Button>
              
              <Typography variant="body2" color="text.secondary" sx={{ mt: 2, maxWidth: 400, textAlign: 'center' }}>
                Showing {metastases.length} metastases at their actual detected positions within the brain volume.
                Red spheres represent metastatic tissue with size proportional to volume.
              </Typography>
            </Box>
          </>
        ) : (
          <>
            <Box
              sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                height: '300px',
                backgroundColor: '#f8f8f8',
                borderRadius: 2,
                border: '2px dashed #ddd',
              }}
            >
              <Info sx={{ fontSize: 60, color: 'primary.main', mb: 2, opacity: 0.7 }} />
              <Typography variant="h6" gutterBottom>
                No Metastases Detected
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ maxWidth: 400, mb: 2, textAlign: 'center' }}>
                The AI model analyzed the brain scan and found no metastatic lesions. This is a negative result.
              </Typography>
            </Box>
          </>
        )}
        
        <Typography variant="caption" color="text.secondary" sx={{ position: 'absolute', bottom: 10, right: 10 }}>
          Job ID: {jobId}
        </Typography>
      </Box>
    </Box>
  );
};

export default Visualization3DPlaceholder;