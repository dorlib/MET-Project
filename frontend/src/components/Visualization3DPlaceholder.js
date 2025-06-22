import React, { useEffect, useState } from 'react';
import { Box, Paper, Typography, Button, CircularProgress, Slider } from '@mui/material';
import { Info, Fullscreen, FullscreenExit } from '@mui/icons-material';

/**
 * Enhanced placeholder component for 3D Brain and Metastasis Visualization
 * This version shows a static visualization based on the metadata
 * 
 * Note: This is a placeholder component that simulates a 3D visualization.
 * In the future, this could be replaced with a real 3D rendering using Three.js or similar.
 * 
 * @param {Object} props
 * @param {string} props.jobId - Job ID
 * @param {Array} props.metastases - Array of metastasis objects with position and volume
 */
const Visualization3DPlaceholder = ({ jobId, metastases = [] }) => {
  const [loading, setLoading] = useState(false);
  const [fullscreen, setFullscreen] = useState(false);
  const [viewAngle, setViewAngle] = useState(0);
  const [rotationActive, setRotationActive] = useState(false);
  
  // Function to toggle fullscreen mode
  const handleFullscreen = () => {
    setFullscreen(!fullscreen);
  };
  
  // Simulates a loading state for the visualization
  useEffect(() => {
    setLoading(true);
    const timer = setTimeout(() => {
      setLoading(false);
    }, 1500);
    return () => clearTimeout(timer);
  }, []);
  
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
                perspective: '800px'
              }}
            >
              {/* Brain outline representation */}
              <Box
                sx={{
                  width: '250px',
                  height: '180px',
                  borderRadius: '50% 50% 50% 50% / 60% 60% 40% 40%',
                  border: '2px solid rgba(128, 128, 128, 0.5)',
                  position: 'absolute',
                  transform: `rotateY(${viewAngle}deg)`,
                  transition: 'transform 0.1s',
                  background: 'linear-gradient(135deg, rgba(235,235,235,0.4) 0%, rgba(200,200,200,0.1) 100%)',
                }}
              />
              
              {/* Render metastases as circles */}
              {metastases.map((met, index) => {
                const size = Math.max(10, Math.min(50, met.volume / 10)); // Scale size based on volume
                const color = `hsl(${(index * 30) % 360}, 80%, 50%)`; // Different color for each
                const depth = (index % 3) - 1; // -1, 0, 1 for z-position
                
                // Calculate position based on the rotated view
                const angle = ((index * 60) + viewAngle) % 360;
                const radians = (angle * Math.PI) / 180;
                const radius = 70 + (index % 20); // Vary radius slightly
                
                const x = Math.cos(radians) * radius;
                const y = Math.sin(radians) * radius * 0.7; // Adjust for brain shape
                
                return (
                  <Box
                    key={index}
                    sx={{
                      width: `${size}px`,
                      height: `${size}px`,
                      borderRadius: '50%',
                      position: 'absolute',
                      backgroundColor: color,
                      opacity: 0.8,
                      transform: `translate(${x}px, ${y}px) scale(${1 + depth * 0.2})`,
                      transition: 'transform 0.1s',
                      boxShadow: '0 0 8px rgba(0,0,0,0.3)',
                      zIndex: depth + 2,
                    }}
                  />
                );
              })}
            </Box>
            
            <Box sx={{ 
              width: '100%', 
              mt: 2, 
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center'
            }}>
              <Slider
                value={viewAngle}
                onChange={handleViewAngleChange}
                min={0}
                max={359}
                sx={{ width: '80%', mb: 2 }}
                disabled={rotationActive}
              />
              
              <Button
                variant="contained"
                color={rotationActive ? "secondary" : "primary"}
                onClick={toggleRotation}
                size="small"
              >
                {rotationActive ? "Stop Rotation" : "Auto-Rotate"}
              </Button>
              
              <Typography variant="body2" color="text.secondary" sx={{ mt: 2, maxWidth: 400, textAlign: 'center' }}>
                This scan contains {metastases.length} metastases with a combined volume of {metastases.reduce((sum, m) => sum + m.volume, 0).toFixed(2)} mm³.
              </Typography>
            </Box>
          </>
        ) : (
          <>
            <Info sx={{ fontSize: 60, color: 'primary.main', mb: 2, opacity: 0.7 }} />
            <Typography variant="h6" gutterBottom>
              No Metastases Detected
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ maxWidth: 400, mb: 2 }}>
              No metastases were found in this scan, or metastasis data is unavailable.
            </Typography>
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