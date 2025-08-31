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
  Switch,
  Chip,
  Divider,
  Card,
  CardContent
} from '@mui/material';
import { Visibility, VisibilityOff, Analytics } from '@mui/icons-material';
import apiService from '../services/api';

const Enhanced3DVisualization = ({ jobId, result }) => {
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  
  // State management
  const [volumeData, setVolumeData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Visualization controls
  const [rotation, setRotation] = useState({ x: 15, y: 0 });
  const [zoom, setZoom] = useState(1.0);
  const [autoRotate, setAutoRotate] = useState(true);
  const [showBrainData, setShowBrainData] = useState(true);
  const [brainOpacity, setBrainOpacity] = useState(0.5);
  
  // Tissue layer visibility controls
  const [layerVisibility, setLayerVisibility] = useState({});
  const [layerOpacity, setLayerOpacity] = useState({});
  
  // Mouse interaction
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  
  // Load enhanced volumetric data
  useEffect(() => {
    if (!jobId) return;

    const loadVolumeData = async () => {
      try {
        setIsLoading(true);
        setError(null);
        
        console.log('Loading enhanced 3D data for job:', jobId);
        const response = await apiService.getVolumetric3D(jobId);
        console.log('Enhanced 3D data loaded:', response.data);
        
        const data = response.data;
        setVolumeData(data);
        
        // Initialize layer visibility and opacity
        const visibility = {};
        const opacity = {};
        data.tissue_layers?.forEach(layer => {
          visibility[layer.class_id] = true;
          opacity[layer.class_id] = layer.color[3] || 0.6; // Use alpha from color
        });
        setLayerVisibility(visibility);
        setLayerOpacity(opacity);
        
      } catch (error) {
        console.error('Error loading enhanced 3D data:', error);
        setError('Failed to load enhanced volumetric brain data');
      } finally {
        setIsLoading(false);
      }
    };

    loadVolumeData();
  }, [jobId]);

  // 3D to screen projection
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
    
    const screenX = (newX * scale * 200) + canvasWidth / 2;
    const screenY = (newY * scale * 200) + canvasHeight / 2;
    
    return { x: screenX, y: screenY, z: newZ, scale };
  }, [rotation.x, rotation.y, zoom]);

  // Simple brain volume rendering like napari - grayscale background at 0.5 opacity
  const renderBrainVolume = useCallback((ctx, brainData, canvasWidth, canvasHeight) => {
    if (!brainData || !showBrainData || brainOpacity === 0) return;

    const { shape, data } = brainData;
    if (!shape || !data) return;

    const [depth, height, width] = shape;
    
    // Simple approach: render every slice as napari would
    for (let z = 0; z < depth; z++) {
      if (!data[z]) continue;
      
      const normalizedZ = (z / depth - 0.5) * 2; // [-1, 1]
      const sliceData = data[z];
      
      // Check if slice has brain data
      let hasData = false;
      for (let y = 0; y < height && !hasData; y++) {
        for (let x = 0; x < width && !hasData; x++) {
          if (sliceData[y][x] > 0.1) hasData = true;
        }
      }
      
      if (!hasData) continue;
      
      // Create simple brain slice
      const brainCanvas = document.createElement('canvas');
      brainCanvas.width = width;
      brainCanvas.height = height;
      const brainCtx = brainCanvas.getContext('2d');
      const imageData = brainCtx.createImageData(width, height);
      const imgData = imageData.data;
      
      // Simple grayscale rendering like napari "Raw Image" layer
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const idx = (y * width + x) * 4;
          const brainValue = Math.floor(sliceData[y][x] * 255);
          
          if (brainValue > 25) { // Only render meaningful brain tissue
            // Simple grayscale like napari
            imgData[idx] = brainValue;     // R
            imgData[idx + 1] = brainValue; // G  
            imgData[idx + 2] = brainValue; // B
            imgData[idx + 3] = Math.floor(brainOpacity * 255); // A - use full opacity setting
          }
        }
      }
      
      brainCtx.putImageData(imageData, 0, 0);
      
      // Simple 3D projection
      const corners = [
        { x: -0.8, y: -0.8, z: normalizedZ },
        { x: 0.8, y: -0.8, z: normalizedZ },
        { x: 0.8, y: 0.8, z: normalizedZ },
        { x: -0.8, y: 0.8, z: normalizedZ }
      ];
      
      const projectedCorners = corners.map(corner => 
        project3DToScreen(corner.x, corner.y, corner.z, canvasWidth, canvasHeight)
      );
      
      const avgZ = projectedCorners.reduce((sum, corner) => sum + corner.z, 0) / 4;
      if (avgZ < -1000) continue;
      
      ctx.save();
      ctx.globalAlpha = brainOpacity; // Match napari opacity approach
      
      try {
        const p0 = projectedCorners[0];
        const p1 = projectedCorners[1];
        const p2 = projectedCorners[3];
        
        const scaleX = (p1.x - p0.x) / width;
        const scaleY = (p2.y - p0.y) / height;
        
        ctx.setTransform(scaleX, 0, 0, scaleY, p0.x, p0.y);
        ctx.drawImage(brainCanvas, 0, 0);
      } catch (e) {
        // Simple fallback
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.fillStyle = `rgba(128, 128, 128, ${brainOpacity})`;
        ctx.beginPath();
        projectedCorners.forEach((corner, i) => {
          if (i === 0) ctx.moveTo(corner.x, corner.y);
          else ctx.lineTo(corner.x, corner.y);
        });
        ctx.closePath();
        ctx.fill();
      }
      
      ctx.restore();
    }
  }, [project3DToScreen, showBrainData, brainOpacity]);

  // Render tissue layers as unified 3D volumes with proper depth blending
  const renderTissueLayers = useCallback((ctx, tissueLayers, canvasWidth, canvasHeight) => {
    if (!tissueLayers) return;

    // Pre-compute all visible slices with their depth information for proper z-ordering
    const allSlices = [];
    
    tissueLayers.forEach(layer => {
      const { class_id, class_name, color, mask_data, mask_shape } = layer;
      
      if (!layerVisibility[class_id] || layerOpacity[class_id] === 0) return;
      if (!mask_data || !mask_shape) return;

      const [depth, height, width] = mask_shape;
      const currentOpacity = layerOpacity[class_id];
      
      // Collect all slices with tissue data
      for (let z = 0; z < depth; z++) {
        if (!mask_data[z]) continue;
        
        const normalizedZ = (z / depth - 0.5) * 2; // [-1, 1]
        
        // Check if slice has any tissue data
        let hasData = false;
        const slicePixels = [];
        
        for (let y = 0; y < height; y++) {
          const row = [];
          for (let x = 0; x < width; x++) {
            const maskValue = mask_data[z][y][x];
            row.push(maskValue);
            if (maskValue > 0) hasData = true;
          }
          slicePixels.push(row);
        }
        
        if (!hasData) continue;
        
        // Project slice corners to get average Z depth
        const corners = [
          { x: -0.8, y: -0.8, z: normalizedZ },
          { x: 0.8, y: -0.8, z: normalizedZ },
          { x: 0.8, y: 0.8, z: normalizedZ },
          { x: -0.8, y: 0.8, z: normalizedZ }
        ];
        
        const projectedCorners = corners.map(corner => 
          project3DToScreen(corner.x, corner.y, corner.z, canvasWidth, canvasHeight)
        );
        
        const avgZ = projectedCorners.reduce((sum, corner) => sum + corner.z, 0) / 4;
        
        // Store slice data for depth-sorted rendering
        allSlices.push({
          layer,
          z,
          normalizedZ,
          avgZ,
          projectedCorners,
          slicePixels,
          width,
          height,
          color,
          currentOpacity,
          class_id
        });
      }
    });
    
    // Sort all slices by depth (back to front) for proper alpha blending
    allSlices.sort((a, b) => b.avgZ - a.avgZ);
    
    // Render all slices in proper depth order
    allSlices.forEach(slice => {
      const { 
        projectedCorners, slicePixels, width, height, 
        color, currentOpacity, avgZ, class_id 
      } = slice;
      
      if (avgZ < -1000) return; // Skip slices too far behind
      
      // Create slice canvas with anti-aliasing
      const sliceCanvas = document.createElement('canvas');
      sliceCanvas.width = width * 2; // Higher resolution for better quality
      sliceCanvas.height = height * 2;
      const sliceCtx = sliceCanvas.getContext('2d');
      sliceCtx.imageSmoothingEnabled = true;
      sliceCtx.imageSmoothingQuality = 'high';
      
      const imageData = sliceCtx.createImageData(width * 2, height * 2);
      const imgData = imageData.data;
      
      // Render with edge smoothing and proper alpha blending
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const maskValue = slicePixels[y][x];
          
          if (maskValue > 0) {
            // Render at 2x resolution for smoothing
            for (let dy = 0; dy < 2; dy++) {
              for (let dx = 0; dx < 2; dx++) {
                const pixelY = y * 2 + dy;
                const pixelX = x * 2 + dx;
                const pixelIdx = (pixelY * width * 2 + pixelX) * 4;
                
                // Edge detection for smoother borders
                const neighbors = [
                  (y > 0 && slicePixels[y-1][x]) ? 1 : 0,
                  (y < height-1 && slicePixels[y+1][x]) ? 1 : 0,
                  (x > 0 && slicePixels[y][x-1]) ? 1 : 0,
                  (x < width-1 && slicePixels[y][x+1]) ? 1 : 0
                ];
                const neighborCount = neighbors.reduce((a, b) => a + b, 0);
                
                // Softer edges at boundaries
                const edgeFactor = Math.min(1.0, neighborCount / 4 * 1.2);
                
                imgData[pixelIdx] = Math.floor(color[0] * 255);     // R
                imgData[pixelIdx + 1] = Math.floor(color[1] * 255); // G
                imgData[pixelIdx + 2] = Math.floor(color[2] * 255); // B
                imgData[pixelIdx + 3] = Math.floor(currentOpacity * edgeFactor * 255); // A
              }
            }
          }
        }
      }
      
      sliceCtx.putImageData(imageData, 0, 0);
      
      // Calculate depth-based alpha and scale
      const depthScale = Math.max(0.7, Math.min(1.2, 1 + avgZ * 0.0001));
      const depthAlpha = Math.max(0.2, Math.min(1.0, 1 - Math.abs(avgZ) * 0.0005));
      
      ctx.save();
      ctx.globalAlpha = depthAlpha * currentOpacity;
      ctx.globalCompositeOperation = 'source-over'; // Proper alpha blending
      
      try {
        const p0 = projectedCorners[0];
        const p1 = projectedCorners[1];
        const p2 = projectedCorners[3];
        
        const scaleX = (p1.x - p0.x) / (width * 2);
        const scaleY = (p2.y - p0.y) / (height * 2);
        
        ctx.setTransform(scaleX * depthScale, 0, 0, scaleY * depthScale, p0.x, p0.y);
        ctx.drawImage(sliceCanvas, 0, 0);
      } catch (e) {
        // Fallback with improved polygon rendering
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        
        // Create gradient for better volume appearance
        const centerX = projectedCorners.reduce((sum, p) => sum + p.x, 0) / 4;
        const centerY = projectedCorners.reduce((sum, p) => sum + p.y, 0) / 4;
        const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, 50);
        gradient.addColorStop(0, `rgba(${Math.floor(color[0]*255)}, ${Math.floor(color[1]*255)}, ${Math.floor(color[2]*255)}, ${depthAlpha * currentOpacity})`);
        gradient.addColorStop(1, `rgba(${Math.floor(color[0]*255)}, ${Math.floor(color[1]*255)}, ${Math.floor(color[2]*255)}, ${depthAlpha * currentOpacity * 0.3})`);
        
        ctx.fillStyle = gradient;
        ctx.beginPath();
        projectedCorners.forEach((corner, i) => {
          if (i === 0) ctx.moveTo(corner.x, corner.y);
          else ctx.lineTo(corner.x, corner.y);
        });
        ctx.closePath();
          ctx.fill();
      }
      
      ctx.restore();
    });
  }, [project3DToScreen, layerVisibility, layerOpacity]);

  // Render connected component markers
  const renderComponentMarkers = useCallback((ctx, tissueLayers, canvasWidth, canvasHeight) => {
    if (!tissueLayers) return;

    tissueLayers.forEach(layer => {
      const { class_id, class_name, color, instances } = layer;
      
      if (!layerVisibility[class_id] || !instances) return;

      instances.forEach(instance => {
        const { centroid_normalized, volume_mm3 } = instance;
        const [z_norm, y_norm, x_norm] = centroid_normalized;
        
        // Convert to [-1, 1] coordinates
        const x = (x_norm - 0.5) * 2;
        const y = (y_norm - 0.5) * 2;
        const z = (z_norm - 0.5) * 2;
        
        const projected = project3DToScreen(x, y, z, canvasWidth, canvasHeight);
        
        if (projected.x < 0 || projected.x > canvasWidth || 
            projected.y < 0 || projected.y > canvasHeight) return;
        
        // Size based on volume
        const baseSize = Math.max(2, Math.min(15, Math.cbrt(volume_mm3 || 1) * 0.8));
        const size = baseSize * projected.scale;
        
        const depthAlpha = Math.max(0.3, Math.min(1.0, 1 - Math.abs(projected.z) * 0.001));
        
        // Draw glowing marker
        const gradient = ctx.createRadialGradient(
          projected.x, projected.y, 0, 
          projected.x, projected.y, size * 2
        );
        gradient.addColorStop(0, `rgba(${Math.floor(color[0]*255)}, ${Math.floor(color[1]*255)}, ${Math.floor(color[2]*255)}, ${depthAlpha * 0.9})`);
        gradient.addColorStop(0.5, `rgba(${Math.floor(color[0]*255)}, ${Math.floor(color[1]*255)}, ${Math.floor(color[2]*255)}, ${depthAlpha * 0.6})`);
        gradient.addColorStop(1, `rgba(${Math.floor(color[0]*255)}, ${Math.floor(color[1]*255)}, ${Math.floor(color[2]*255)}, ${depthAlpha * 0.1})`);
        
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(projected.x, projected.y, size * 2, 0, 2 * Math.PI);
        ctx.fill();
        
        // Core marker
        ctx.fillStyle = `rgba(${Math.floor(color[0]*255)}, ${Math.floor(color[1]*255)}, ${Math.floor(color[2]*255)}, ${depthAlpha})`;
        ctx.beginPath();
        ctx.arc(projected.x, projected.y, size, 0, 2 * Math.PI);
        ctx.fill();
      });
    });
  }, [project3DToScreen, layerVisibility]);

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

    // Clear with dark background
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    // Render brain volume exactly like napari: simple, visible background at 0.5 opacity
    if (volumeData.brain_volume) {
      renderBrainVolume(ctx, volumeData.brain_volume, canvasWidth, canvasHeight);
    }

    // Render tissue layers (segmentation) - these should be the main focus
    if (volumeData.tissue_layers) {
      renderTissueLayers(ctx, volumeData.tissue_layers, canvasWidth, canvasHeight);
    }

    // Coordinate system reference
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
  }, [volumeData, renderBrainVolume, renderTissueLayers, renderComponentMarkers]);

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

  // Auto-rotation
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

  // Setup wheel event listener
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.addEventListener('wheel', handleWheel, { passive: false });
    
    return () => {
      canvas.removeEventListener('wheel', handleWheel);
    };
  }, [handleWheel]);

  // Toggle layer visibility
  const toggleLayerVisibility = (classId) => {
    setLayerVisibility(prev => ({
      ...prev,
      [classId]: !prev[classId]
    }));
  };

  // Update layer opacity
  const updateLayerOpacity = (classId, opacity) => {
    setLayerOpacity(prev => ({
      ...prev,
      [classId]: opacity
    }));
  };

  if (isLoading) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Enhanced 3D Brain Visualization
        </Typography>
        <LinearProgress />
        <Typography variant="body2" sx={{ mt: 2 }}>
          Loading comprehensive brain volume data with connected component analysis...
        </Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Enhanced 3D Brain Visualization
        </Typography>
        <Alert severity="error">{error}</Alert>
      </Box>
    );
  }

  return (
    <Paper sx={{ p: 2, bgcolor: '#f5f5f5' }}>
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
        <Analytics sx={{ mr: 1, color: 'primary.main' }} />
        Enhanced 3D Segmentation Visualization
      </Typography>
      
      {/* Global Controls */}
      <Grid container spacing={2} sx={{ mb: 2 }}>
        <Grid item xs={12} sm={3}>
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
        <Grid item xs={12} sm={3}>
          <FormControlLabel
            control={
              <Switch
                checked={showBrainData}
                onChange={(e) => setShowBrainData(e.target.checked)}
                size="small"
              />
            }
            label="Show Brain Volume"
          />
        </Grid>
        <Grid item xs={12} sm={3}>
          <Typography variant="body2" gutterBottom>
            Brain Opacity: {Math.round(brainOpacity * 100)}%
          </Typography>
          <Slider
            value={brainOpacity}
            onChange={(e, newValue) => setBrainOpacity(newValue)}
            min={0.0}
            max={1.0}
            step={0.1}
            size="small"
            disabled={!showBrainData}
          />
        </Grid>
      </Grid>

      <Divider sx={{ mb: 2 }} />

      {/* Tissue Layer Controls */}
      {volumeData?.tissue_layers && (
        <Card sx={{ mb: 2, bgcolor: '#fafafa' }}>
          <CardContent>
            <Typography variant="subtitle2" gutterBottom>
              Tissue Layer Controls
            </Typography>
            <Grid container spacing={2}>
              {volumeData.tissue_layers.map(layer => (
                <Grid item xs={12} sm={6} md={4} key={layer.class_id}>
                  <Box sx={{ p: 1, border: '1px solid #ddd', borderRadius: 1 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                      <Box 
                        sx={{ 
                          width: 16, 
                          height: 16, 
                          bgcolor: `rgba(${Math.floor(layer.color[0]*255)}, ${Math.floor(layer.color[1]*255)}, ${Math.floor(layer.color[2]*255)}, 1)`,
                          mr: 1,
                          border: '1px solid #333'
                        }} 
                      />
                      <Typography variant="body2" sx={{ flex: 1 }}>
                        {layer.class_name}
                      </Typography>
                      <Switch
                        size="small"
                        checked={layerVisibility[layer.class_id] || false}
                        onChange={() => toggleLayerVisibility(layer.class_id)}
                      />
                    </Box>
                    <Typography variant="caption" color="text.secondary">
                      {layer.total_instances} components
                    </Typography>
                    <Typography variant="body2" gutterBottom>
                      Opacity: {Math.round((layerOpacity[layer.class_id] || 0) * 100)}%
                    </Typography>
                    <Slider
                      value={layerOpacity[layer.class_id] || 0}
                      onChange={(e, newValue) => updateLayerOpacity(layer.class_id, newValue)}
                      min={0.0}
                      max={1.0}
                      step={0.1}
                      size="small"
                      disabled={!layerVisibility[layer.class_id]}
                    />
                  </Box>
                </Grid>
              ))}
            </Grid>
          </CardContent>
        </Card>
      )}

      {/* Volume Info */}
      {volumeData && (
        <Box sx={{ mb: 2 }}>
          <Typography variant="body2" color="text.secondary">
            Original Shape: {volumeData.original_shape?.join('×')} | 
            Tissue Classes: {volumeData.total_classes} | 
            Has Brain Data: {volumeData.has_brain_data ? 'Yes' : 'No'} |
            {volumeData.downsample_info && ` ${volumeData.downsample_info.description}`}
          </Typography>
        </Box>
      )}

      {/* 3D Visualization Canvas */}
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
      >
        <canvas
          ref={canvasRef}
          style={{
            width: '100%',
            height: '100%',
            display: 'block'
          }}
        />
        
        {/* Instructions */}
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
          Drag to rotate • Scroll to zoom • Toggle layers above
        </Box>
      </Box>

      {/* Connected Components Summary */}
      {volumeData?.connected_components && (
        <Card sx={{ mt: 2 }}>
          <CardContent>
            <Typography variant="subtitle2" gutterBottom>
              Connected Components Analysis
            </Typography>
            <Grid container spacing={1}>
              {Object.entries(volumeData.connected_components).map(([classId, instances]) => {
                const className = volumeData.class_names[classId] || `Class ${classId}`;
                const totalVolume = instances.reduce((sum, inst) => sum + inst.volume_mm3, 0);
                
                return (
                  <Grid item key={classId}>
                    <Chip
                      size="small"
                      label={`${className}: ${instances.length} (${totalVolume.toFixed(1)} mm³)`}
                      sx={{ 
                        bgcolor: `rgba(${Math.floor(volumeData.color_mapping[classId][0]*255)}, ${Math.floor(volumeData.color_mapping[classId][1]*255)}, ${Math.floor(volumeData.color_mapping[classId][2]*255)}, 0.8)`,
                        color: 'white',
                        fontSize: '0.75rem'
                      }}
                    />
                  </Grid>
                );
              })}
            </Grid>
          </CardContent>
        </Card>
      )}
    </Paper>
  );
};

export default Enhanced3DVisualization;
