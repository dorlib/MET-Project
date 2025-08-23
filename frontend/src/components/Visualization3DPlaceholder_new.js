import React, { useRef, useEffect, useState } from 'react';
import { 
  Box, 
  Typography, 
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  Paper,
  Grid,
  LinearProgress,
  Alert
} from '@mui/material';
import apiService from '../services/api';

const Visualization3DPlaceholder = ({ jobId, result }) => {
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  const [viewType, setViewType] = useState('surface');
  const [opacity, setOpacity] = useState(0.7);
  const [rotation, setRotation] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1.0);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [brainData, setBrainData] = useState(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);

  // Load brain volume data
  useEffect(() => {
    if (!jobId) return;

    const loadBrainData = async () => {
      try {
        setIsLoading(true);
        setError(null);
        
        const data = await apiService.getBrainVolume3D(jobId);
        console.log('Brain volume data loaded:', data);
        setBrainData(data);
      } catch (error) {
        console.error('Error loading brain volume data:', error);
        setError('Failed to load 3D brain data');
      } finally {
        setIsLoading(false);
      }
    };

    loadBrainData();
  }, [jobId]);

  // Convert 3D coordinates to screen coordinates
  const project3DToScreen = (x, y, z, canvasWidth, canvasHeight) => {
    // Apply rotation
    const rotX = rotation.x * Math.PI / 180;
    const rotY = rotation.y * Math.PI / 180;
    
    // Rotate around X axis
    let newY = y * Math.cos(rotX) - z * Math.sin(rotX);
    let newZ = y * Math.sin(rotX) + z * Math.cos(rotX);
    
    // Rotate around Y axis
    let newX = x * Math.cos(rotY) + newZ * Math.sin(rotY);
    newZ = -x * Math.sin(rotY) + newZ * Math.cos(rotY);
    
    // Apply zoom
    newX *= zoom;
    newY *= zoom;
    newZ *= zoom;
    
    // Simple perspective projection
    const perspective = 600;
    const scale = perspective / (perspective + newZ * 200);
    
    const screenX = (newX * scale * 150) + canvasWidth / 2;
    const screenY = (newY * scale * 150) + canvasHeight / 2;
    const depth = newZ;
    
    return { x: screenX, y: screenY, z: depth, scale };
  };

  // Render brain surface as a point cloud with depth
  const renderBrainSurface = (ctx, brainPoints, canvasWidth, canvasHeight) => {
    if (!brainPoints || brainPoints.length === 0) return;

    // Project all points and sort by depth
    const projectedPoints = brainPoints.map(point => {
      const projected = project3DToScreen(
        point.position[0], 
        point.position[1], 
        point.position[2], 
        canvasWidth, 
        canvasHeight
      );
      return {
        ...projected,
        intensity: point.intensity || 0.3
      };
    }).sort((a, b) => a.z - b.z); // Back to front for proper depth

    // Render points as brain tissue
    projectedPoints.forEach(point => {
      if (point.x < 0 || point.x > canvasWidth || point.y < 0 || point.y > canvasHeight) return;
      
      // Depth-based transparency and size
      const depthAlpha = Math.max(0.1, 1 - Math.abs(point.z) * 0.5);
      const pointSize = Math.max(0.5, point.scale * 2);
      
      // Brain tissue color - gray matter
      const gray = Math.floor(100 + point.intensity * 100);
      ctx.fillStyle = `rgba(${gray}, ${gray}, ${gray + 20}, ${depthAlpha * opacity})`;
      
      ctx.beginPath();
      ctx.arc(point.x, point.y, pointSize, 0, 2 * Math.PI);
      ctx.fill();
    });
  };

  // Render metastases as bright red spheres
  const renderMetastases = (ctx, metastases, canvasWidth, canvasHeight) => {
    if (!metastases || metastases.length === 0) return;

    // Project and sort metastases by depth
    const projectedMets = metastases.map(met => {
      const projected = project3DToScreen(
        met.position[0], 
        met.position[1], 
        met.position[2], 
        canvasWidth, 
        canvasHeight
      );
      return {
        ...projected,
        volume: met.volume,
        id: met.id
      };
    }).sort((a, b) => a.z - b.z);

    // Render metastases on top
    projectedMets.forEach(met => {
      if (met.x < 0 || met.x > canvasWidth || met.y < 0 || met.y > canvasHeight) return;
      
      // Size based on volume (logarithmic scale for better visibility)
      const baseSize = 3;
      const volumeSize = met.volume ? Math.log10(Math.max(1, met.volume)) * 2 : baseSize;
      const size = Math.max(baseSize, volumeSize * met.scale);
      
      // Bright red with glow effect
      const glowSize = size * 2;
      
      // Outer glow
      const gradient = ctx.createRadialGradient(met.x, met.y, 0, met.x, met.y, glowSize);
      gradient.addColorStop(0, 'rgba(255, 50, 50, 0.8)');
      gradient.addColorStop(0.5, 'rgba(255, 80, 80, 0.4)');
      gradient.addColorStop(1, 'rgba(255, 100, 100, 0.1)');
      
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(met.x, met.y, glowSize, 0, 2 * Math.PI);
      ctx.fill();
      
      // Inner core
      ctx.fillStyle = 'rgba(255, 30, 30, 0.9)';
      ctx.beginPath();
      ctx.arc(met.x, met.y, size, 0, 2 * Math.PI);
      ctx.fill();
      
      // Highlight
      ctx.fillStyle = 'rgba(255, 200, 200, 0.6)';
      ctx.beginPath();
      ctx.arc(met.x - size/3, met.y - size/3, size/3, 0, 2 * Math.PI);
      ctx.fill();
    });
  };

  // Main render function
  const render = () => {
    const canvas = canvasRef.current;
    if (!canvas || !brainData) return;

    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    const canvasWidth = rect.width;
    const canvasHeight = rect.height;
    
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;

    // Clear with medical dark background
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    // Add subtle grid for depth reference
    ctx.strokeStyle = 'rgba(40, 40, 60, 0.3)';
    ctx.lineWidth = 1;
    for (let i = 0; i < canvasWidth; i += 50) {
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i, canvasHeight);
      ctx.stroke();
    }
    for (let j = 0; j < canvasHeight; j += 50) {
      ctx.beginPath();
      ctx.moveTo(0, j);
      ctx.lineTo(canvasWidth, j);
      ctx.stroke();
    }

    // Render brain surface
    if (brainData.brain_surface_points && viewType !== 'metastases-only') {
      renderBrainSurface(ctx, brainData.brain_surface_points, canvasWidth, canvasHeight);
    }

    // Render metastases
    if (brainData.metastases) {
      renderMetastases(ctx, brainData.metastases, canvasWidth, canvasHeight);
    }

    // Add coordinate system reference
    ctx.strokeStyle = 'rgba(100, 100, 100, 0.5)';
    ctx.lineWidth = 2;
    const centerX = canvasWidth / 2;
    const centerY = canvasHeight / 2;
    
    // X axis (red)
    ctx.strokeStyle = 'rgba(255, 100, 100, 0.7)';
    ctx.beginPath();
    ctx.moveTo(centerX - 30, centerY);
    ctx.lineTo(centerX + 30, centerY);
    ctx.stroke();
    
    // Y axis (green)
    ctx.strokeStyle = 'rgba(100, 255, 100, 0.7)';
    ctx.beginPath();
    ctx.moveTo(centerX, centerY - 30);
    ctx.lineTo(centerX, centerY + 30);
    ctx.stroke();
  };

  // Animation loop
  useEffect(() => {
    const animate = () => {
      render();
      animationRef.current = requestAnimationFrame(animate);
    };
    
    if (brainData) {
      animate();
    }
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [brainData, viewType, opacity, rotation, zoom]);

  // Auto-rotation effect
  useEffect(() => {
    const autoRotate = () => {
      if (!isDragging) {
        setRotation(prev => ({
          x: prev.x,
          y: (prev.y + 0.5) % 360
        }));
      }
    };

    const interval = setInterval(autoRotate, 50);
    return () => clearInterval(interval);
  }, [isDragging]);

  // Mouse interaction handlers
  const handleMouseDown = (e) => {
    setIsDragging(true);
    setMousePos({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = (e) => {
    if (!isDragging) return;
    
    const deltaX = e.clientX - mousePos.x;
    const deltaY = e.clientY - mousePos.y;
    
    setRotation(prev => ({
      x: Math.max(-90, Math.min(90, prev.x + deltaY * 0.5)),
      y: (prev.y + deltaX * 0.5) % 360
    }));
    
    setMousePos({ x: e.clientX, y: e.clientY });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleWheel = (e) => {
    e.preventDefault();
    const delta = e.deltaY * -0.001;
    setZoom(prev => Math.max(0.5, Math.min(3.0, prev + delta)));
  };

  if (isLoading) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          3D Brain Visualization
        </Typography>
        <LinearProgress />
        <Typography variant="body2" sx={{ mt: 2 }}>
          Loading brain volume data from all slices...
        </Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          3D Brain Visualization
        </Typography>
        <Alert severity="error">{error}</Alert>
      </Box>
    );
  }

  return (
    <Paper sx={{ p: 2, bgcolor: '#f5f5f5' }}>
      <Typography variant="h6" gutterBottom>
        3D Brain Visualization - Real Volume Data
      </Typography>
      
      <Grid container spacing={2} sx={{ mb: 2 }}>
        <Grid item xs={12} sm={4}>
          <FormControl fullWidth size="small">
            <InputLabel>View Type</InputLabel>
            <Select
              value={viewType}
              label="View Type"
              onChange={(e) => setViewType(e.target.value)}
            >
              <MenuItem value="surface">Brain Surface + Metastases</MenuItem>
              <MenuItem value="metastases-only">Metastases Only</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} sm={4}>
          <Typography variant="body2" gutterBottom>
            Opacity: {Math.round(opacity * 100)}%
          </Typography>
          <Slider
            value={opacity}
            onChange={(e, newValue) => setOpacity(newValue)}
            min={0.1}
            max={1.0}
            step={0.1}
            size="small"
          />
        </Grid>
        <Grid item xs={12} sm={4}>
          <Typography variant="body2" gutterBottom>
            Zoom: {zoom.toFixed(1)}x
          </Typography>
          <Slider
            value={zoom}
            onChange={(e, newValue) => setZoom(newValue)}
            min={0.5}
            max={3.0}
            step={0.1}
            size="small"
          />
        </Grid>
      </Grid>

      {brainData && (
        <Box sx={{ mb: 2 }}>
          <Typography variant="body2" color="text.secondary">
            Brain Points: {brainData.brain_points_count} | 
            Metastases: {brainData.metastases_count} | 
            Volume Shape: {brainData.segmentation_shape?.join('×')}
          </Typography>
        </Box>
      )}

      <Box
        sx={{
          width: '100%',
          height: '500px',
          bgcolor: '#0a0a0a',
          border: '2px solid #333',
          borderRadius: 2,
          position: 'relative',
          overflow: 'hidden',
          cursor: isDragging ? 'grabbing' : 'grab'
        }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
      >
        <canvas
          ref={canvasRef}
          style={{
            width: '100%',
            height: '100%',
            display: 'block'
          }}
        />
        
        <Box
          sx={{
            position: 'absolute',
            bottom: 10,
            left: 10,
            color: 'rgba(255, 255, 255, 0.7)',
            fontSize: '12px',
            fontFamily: 'monospace'
          }}
        >
          Click and drag to rotate | Scroll to zoom
        </Box>
      </Box>
    </Paper>
  );
};

export default Visualization3DPlaceholder;
