import React, { useRef, useEffect, useState, useCallback } from 'react';
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
  Alert,
  FormControlLabel,
  Checkbox,
  Switch
} from '@mui/material';
import apiService from '../services/api';

// Shared tissue colors from backend configuration - matching matplotlib jet colormap
const TISSUE_COLORS = {
  0: { name: "Background", color: [0, 0, 127, 0] },        // Dark blue, transparent
  1: { name: "Metastasis", color: [0, 212, 255, 128] },    // Cyan-blue
  2: { name: "Edema", color: [255, 229, 0, 128] },         // Yellow 
  3: { name: "Enhancing Tumor", color: [127, 0, 0, 128] }  // Dark red
};

const VolumetricVisualization3D = ({ jobId, result }) => {
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  const [viewType, setViewType] = useState('volume');
  const [opacity, setOpacity] = useState(0.6);
  const [rotation, setRotation] = useState({ x: 15, y: 0 });
  const [zoom, setZoom] = useState(1.2);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [volumeData, setVolumeData] = useState(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [autoRotate, setAutoRotate] = useState(true);
  const [showSegmentation, setShowSegmentation] = useState(true);
  const [brainIntensity, setBrainIntensity] = useState(0.8);

  // Load real volumetric data from brain scan
  useEffect(() => {
    if (!jobId) return;

    const loadVolumeData = async () => {
      try {
        setIsLoading(true);
        setError(null);
        
        console.log('Loading volumetric 3D data for job:', jobId);
        const response = await apiService.getVolumetric3D(jobId);
        console.log('Volumetric 3D data loaded:', response.data);
        
        setVolumeData(response.data);
      } catch (error) {
        console.error('Error loading volumetric 3D data:', error);
        setError('Failed to load volumetric brain data');
      } finally {
        setIsLoading(false);
      }
    };

    loadVolumeData();
  }, [jobId]);

  // Convert 3D coordinates to screen coordinates with perspective
  const project3DToScreen = useCallback((x, y, z, canvasWidth, canvasHeight) => {
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
    
    // Perspective projection
    const perspective = 800;
    const scale = perspective / (perspective + newZ * 300);
    
    const screenX = (newX * scale * 180) + canvasWidth / 2;
    const screenY = (newY * scale * 180) + canvasHeight / 2;
    
    return { x: screenX, y: screenY, z: newZ, scale };
  }, [rotation.x, rotation.y, zoom]);

  // Render brain volume slices as 3D planes
  const renderVolumeSlices = useCallback((ctx, volumeSlices, canvasWidth, canvasHeight) => {
    if (!volumeSlices || volumeSlices.length === 0) return;

    // Sort slices by Z depth for proper rendering order
    const sortedSlices = [...volumeSlices].sort((a, b) => {
      const aZ = a.z_position - 0.5; // Center around 0
      const bZ = b.z_position - 0.5;
      // After rotation, determine which is further back
      const rotX = rotation.x * Math.PI / 180;
      const rotY = rotation.y * Math.PI / 180;
      
      // Simple depth approximation for sorting
      const aDepth = aZ * Math.cos(rotX) * Math.cos(rotY);
      const bDepth = bZ * Math.cos(rotX) * Math.cos(rotY);
      
      return aDepth - bDepth; // Back to front
    });

    sortedSlices.forEach((slice, sliceIndex) => {
      if (!slice.brain_data && !slice.segmentation_data) return;
      
      const sliceZ = (slice.z_position - 0.5) * 2; // Normalize to [-1, 1]
      const sliceShape = slice.shape;
      
      if (!sliceShape || sliceShape.length !== 2) return;
      
      const [height, width] = sliceShape;
      
      // Create slice canvas with improved resolution for better anatomy
      const upscale = 2; // Render at higher resolution
      const sliceCanvas = document.createElement('canvas');
      sliceCanvas.width = width * upscale;
      sliceCanvas.height = height * upscale;
      const sliceCtx = sliceCanvas.getContext('2d');
      sliceCtx.imageSmoothingEnabled = true;
      sliceCtx.imageSmoothingQuality = 'high';
      
      // Create ImageData for the slice at higher resolution
      const imageData = sliceCtx.createImageData(width * upscale, height * upscale);
      const data = imageData.data;
      
      // Render brain data with enhanced anatomy preservation
      if (slice.brain_data && brainIntensity > 0) {
        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            const originalValue = slice.brain_data[y][x];
            
            if (originalValue > 0.02) { // Lower threshold for better anatomy
              // Apply gamma correction for better brain tissue contrast
              const enhancedValue = Math.pow(originalValue, 0.8);
              const brainValue = enhancedValue * brainIntensity * 255;
              
              // Render at higher resolution
              for (let dy = 0; dy < upscale; dy++) {
                for (let dx = 0; dx < upscale; dx++) {
                  const upY = y * upscale + dy;
                  const upX = x * upscale + dx;
                  const pixelIdx = (upY * width * upscale + upX) * 4;
                  
                  data[pixelIdx] = brainValue;     // R
                  data[pixelIdx + 1] = brainValue; // G  
                  data[pixelIdx + 2] = brainValue; // B
                  data[pixelIdx + 3] = 255 * opacity; // A
                }
              }
            }
          }
        }
      }
      
      // Overlay segmentation data with categorical colors at higher resolution
      if (slice.segmentation_data && showSegmentation) {
        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            const segValue = slice.segmentation_data[y][x];
            
            if (segValue > 0 && TISSUE_COLORS[segValue]) {
              const color = TISSUE_COLORS[segValue].color;
              const alpha = color[3] / 255;
              const invAlpha = 1 - alpha;
              
              // Render at higher resolution
              for (let dy = 0; dy < upscale; dy++) {
                for (let dx = 0; dx < upscale; dx++) {
                  const upY = y * upscale + dy;
                  const upX = x * upscale + dx;
                  const pixelIdx = (upY * width * upscale + upX) * 4;
                  
                  // Blend with existing brain data
                  data[pixelIdx] = data[pixelIdx] * invAlpha + color[0] * alpha;     // R
                  data[pixelIdx + 1] = data[pixelIdx + 1] * invAlpha + color[1] * alpha; // G
                  data[pixelIdx + 2] = data[pixelIdx + 2] * invAlpha + color[2] * alpha; // B
                  data[pixelIdx + 3] = Math.max(data[pixelIdx + 3], color[3] * opacity); // A
                }
              }
            }
          }
        }
      }
      
      sliceCtx.putImageData(imageData, 0, 0);
      
      // Project the slice corners to screen space
      const corners = [
        { x: -0.8, y: -0.8, z: sliceZ },
        { x: 0.8, y: -0.8, z: sliceZ },
        { x: 0.8, y: 0.8, z: sliceZ },
        { x: -0.8, y: 0.8, z: sliceZ }
      ];
      
      const projectedCorners = corners.map(corner => 
        project3DToScreen(corner.x, corner.y, corner.z, canvasWidth, canvasHeight)
      );
      
      // Check if slice is visible (not completely behind camera)
      const avgZ = projectedCorners.reduce((sum, corner) => sum + corner.z, 0) / 4;
      if (avgZ < -1000) return; // Skip slices too far behind
      
      // Calculate depth-based alpha
      const depthAlpha = Math.max(0.1, Math.min(1.0, 1 - Math.abs(avgZ) * 0.001));
      
      // Draw the textured quad (slice)
      ctx.save();
      ctx.globalAlpha = depthAlpha * opacity;
      
      try {
        // Create transformation matrix for the quad
        const p0 = projectedCorners[0];
        const p1 = projectedCorners[1];
        const p2 = projectedCorners[3];
        
        // Simple perspective transform (not perfect but good enough)
        const scaleX = (p1.x - p0.x) / (width * upscale);
        const scaleY = (p2.y - p0.y) / (height * upscale);
        
        ctx.setTransform(scaleX, 0, 0, scaleY, p0.x, p0.y);
        ctx.drawImage(sliceCanvas, 0, 0);
      } catch (e) {
        // Fallback: just draw a line representation
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.strokeStyle = `rgba(100, 100, 100, ${depthAlpha})`;
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 0; i < projectedCorners.length; i++) {
          const corner = projectedCorners[i];
          if (i === 0) {
            ctx.moveTo(corner.x, corner.y);
          } else {
            ctx.lineTo(corner.x, corner.y);
          }
        }
        ctx.closePath();
        ctx.stroke();
      }
      
      ctx.restore();
    });
  }, [project3DToScreen, rotation, opacity, showSegmentation, brainIntensity]);

  // Render metastases as glowing spheres
  const renderMetastases = useCallback((ctx, metastases, canvasWidth, canvasHeight) => {
    if (!metastases || metastases.length === 0) return;

    // Project and sort metastases by depth
    const projectedMets = metastases.map(met => {
      // Convert from [0,1] to [-1,1] coordinates
      const x = (met.position[0] - 0.5) * 2;
      const y = (met.position[1] - 0.5) * 2;
      const z = (met.position[2] - 0.5) * 2;
      
      const projected = project3DToScreen(x, y, z, canvasWidth, canvasHeight);
      return {
        ...projected,
        volume: met.volume,
        id: met.id
      };
    }).sort((a, b) => a.z - b.z); // Back to front

    // Render metastases
    projectedMets.forEach(met => {
      if (met.x < 0 || met.x > canvasWidth || met.y < 0 || met.y > canvasHeight) return;
      
      // Size based on volume with perspective scaling
      const baseSize = Math.max(3, Math.min(20, Math.cbrt(met.volume || 10) * 1.5));
      const size = baseSize * met.scale;
      
      // Depth-based intensity
      const depthAlpha = Math.max(0.3, Math.min(1.0, 1 - Math.abs(met.z) * 0.001));
      
      // Glowing red sphere
      const gradient = ctx.createRadialGradient(met.x, met.y, 0, met.x, met.y, size * 2);
      gradient.addColorStop(0, `rgba(255, 60, 60, ${depthAlpha * 0.9})`);
      gradient.addColorStop(0.5, `rgba(255, 100, 100, ${depthAlpha * 0.6})`);
      gradient.addColorStop(1, `rgba(255, 150, 150, ${depthAlpha * 0.1})`);
      
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(met.x, met.y, size * 2, 0, 2 * Math.PI);
      ctx.fill();
      
      // Core
      ctx.fillStyle = `rgba(255, 30, 30, ${depthAlpha})`;
      ctx.beginPath();
      ctx.arc(met.x, met.y, size, 0, 2 * Math.PI);
      ctx.fill();
      
      // Highlight
      ctx.fillStyle = `rgba(255, 200, 200, ${depthAlpha * 0.7})`;
      ctx.beginPath();
      ctx.arc(met.x - size/3, met.y - size/3, size/3, 0, 2 * Math.PI);
      ctx.fill();
    });
  }, [project3DToScreen]);

  // Main render function
  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !volumeData) return;

    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    const canvasWidth = rect.width;
    const canvasHeight = rect.height;
    
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;

    // Clear with dark medical background
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    // Render volumetric brain slices
    if (volumeData.volume_slices && viewType !== 'metastases-only') {
      renderVolumeSlices(ctx, volumeData.volume_slices, canvasWidth, canvasHeight);
    }

    // Render metastases on top
    if (volumeData.metastases) {
      renderMetastases(ctx, volumeData.metastases, canvasWidth, canvasHeight);
    }

    // Add coordinate system reference
    const centerX = canvasWidth / 2;
    const centerY = canvasHeight / 2;
    
    ctx.strokeStyle = 'rgba(255, 100, 100, 0.5)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(centerX - 30, centerY);
    ctx.lineTo(centerX + 30, centerY);
    ctx.stroke();
    
    ctx.strokeStyle = 'rgba(100, 255, 100, 0.5)';
    ctx.beginPath();
    ctx.moveTo(centerX, centerY - 30);
    ctx.lineTo(centerX, centerY + 30);
    ctx.stroke();
  }, [volumeData, viewType, renderVolumeSlices, renderMetastases]);

  // Animation loop
  useEffect(() => {
    const animate = () => {
      render();
      animationRef.current = requestAnimationFrame(animate);
    };
    
    if (volumeData) {
      animate();
    }
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [volumeData, render]);

  // Auto-rotation effect
  useEffect(() => {
    const autoRotateFunction = () => {
      if (!isDragging && autoRotate) {
        setRotation(prev => ({
          x: prev.x,
          y: (prev.y + 0.3) % 360
        }));
      }
    };

    const interval = setInterval(autoRotateFunction, 50);
    return () => clearInterval(interval);
  }, [isDragging, autoRotate]);

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

  const handleWheel = useCallback((e) => {
    e.preventDefault();
    const delta = e.deltaY * -0.001;
    setZoom(prev => Math.max(0.3, Math.min(3.0, prev + delta)));
  }, []);

  // Setup wheel event listener with proper passive handling
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Add wheel event listener with passive: false to allow preventDefault
    canvas.addEventListener('wheel', handleWheel, { passive: false });
    
    return () => {
      canvas.removeEventListener('wheel', handleWheel);
    };
  }, [handleWheel]);

  if (isLoading) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          3D Brain Visualization
        </Typography>
        <LinearProgress />
        <Typography variant="body2" sx={{ mt: 2 }}>
          Loading real brain scan slices for 3D visualization...
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
        3D Brain Visualization - Real Brain Scan
      </Typography>
      
      <Grid container spacing={2} sx={{ mb: 2 }}>
        <Grid item xs={12} sm={3}>
          <FormControl fullWidth size="small">
            <InputLabel>View Type</InputLabel>
            <Select
              value={viewType}
              label="View Type"
              onChange={(e) => setViewType(e.target.value)}
            >
              <MenuItem value="volume">Brain Volume + Segmentation</MenuItem>
              <MenuItem value="metastases-only">Metastases Only</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} sm={3}>
          <Typography variant="body2" gutterBottom>
            Volume Opacity: {Math.round(opacity * 100)}%
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
        <Grid item xs={12} sm={3}>
          <Typography variant="body2" gutterBottom>
            Brain Intensity: {Math.round(brainIntensity * 100)}%
          </Typography>
          <Slider
            value={brainIntensity}
            onChange={(e, newValue) => setBrainIntensity(newValue)}
            min={0.0}
            max={1.0}
            step={0.1}
            size="small"
          />
        </Grid>
        <Grid item xs={12} sm={3}>
          <Typography variant="body2" gutterBottom>
            Zoom: {zoom.toFixed(1)}x
          </Typography>
          <Slider
            value={zoom}
            onChange={(e, newValue) => setZoom(newValue)}
            min={0.3}
            max={3.0}
            step={0.1}
            size="small"
          />
        </Grid>
      </Grid>

      <Grid container spacing={2} sx={{ mb: 2 }}>
        <Grid item xs={12} sm={4}>
          <FormControlLabel
            control={
              <Switch
                checked={autoRotate}
                onChange={(e) => setAutoRotate(e.target.checked)}
                size="small"
              />
            }
            label="Auto-rotate"
          />
        </Grid>
        <Grid item xs={12} sm={4}>
          <FormControlLabel
            control={
              <Switch
                checked={showSegmentation}
                onChange={(e) => setShowSegmentation(e.target.checked)}
                size="small"
              />
            }
            label="Show Segmentation"
          />
        </Grid>
        <Grid item xs={12} sm={4}>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            Drag to rotate, scroll to zoom
          </Typography>
        </Grid>
      </Grid>

      {volumeData && (
        <Box sx={{ mb: 2 }}>
          <Typography variant="body2" color="text.secondary">
            Original Shape: {volumeData.original_shape?.join('×')} | 
            Slices: {volumeData.slice_count} | 
            Metastases: {volumeData.metastases_count} | 
            Has Brain Data: {volumeData.has_brain_data ? 'Yes' : 'No'}
          </Typography>
        </Box>
      )}

      <Box
        sx={{
          width: '100%',
          height: '600px',
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
        
        {/* Legend */}
        <Box
          sx={{
            position: 'absolute',
            top: 10,
            right: 10,
            bgcolor: 'rgba(0, 0, 0, 0.7)',
            color: 'white',
            p: 1,
            borderRadius: 1,
            fontSize: '12px'
          }}
        >
          {Object.entries(TISSUE_COLORS).map(([classId, info]) => (
            classId !== '0' && (
              <Box key={classId} sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
                <Box 
                  sx={{ 
                    width: 12, 
                    height: 12, 
                    bgcolor: `rgba(${info.color.join(',')})`,
                    mr: 1,
                    border: '1px solid #555'
                  }} 
                />
                {info.name}
              </Box>
            )
          ))}
        </Box>
        
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
          Real brain scan slices with segmentation overlay
        </Box>
      </Box>
    </Paper>
  );
};

export default VolumetricVisualization3D;
