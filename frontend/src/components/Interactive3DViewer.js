import React, { useRef, useEffect, useState } from 'react';
import { 
  Box, 
  Typography, 
  Grid, 
  FormControlLabel, 
  Switch, 
  Slider,
  Paper,
  Alert,
  CircularProgress
} from '@mui/material';
import * as THREE from 'three';
import api from '../services/api';

/**
 * Interactive 3D Brain Viewer using Three.js
 * Shows actual segmentation results from the model
 */
const Interactive3DViewer = ({ jobId, status, results, enhanceContrast, setEnhanceContrast, enhanceEdges, setEnhanceEdges }) => {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const animationRef = useRef(null);
  
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [opacity, setOpacity] = useState(0.8);
  const [showBrain, setShowBrain] = useState(true);
  const [showMetastases, setShowMetastases] = useState(true);
  const [segmentationData, setSegmentationData] = useState(null);
  const [isLoadingSegmentation, setIsLoadingSegmentation] = useState(false);
  const [segmentationError, setSegmentationError] = useState(null);

  // Define loadSegmentationData with useCallback for proper dependency management
  const loadSegmentationData = React.useCallback(async () => {
    if (!jobId || status !== 'completed') return;
    
    setIsLoadingSegmentation(true);
    setSegmentationError(null);
    
    try {
      console.log('Loading segmentation data for job:', jobId);
      const response = await api.getSegmentationData(jobId, {
        downsample: 4,  // Downsample by factor of 4 for performance
        max_voxels: 5000  // Limit to 5000 voxels for performance
      });
      
      console.log('Segmentation data loaded:', response.data);
      setSegmentationData(response.data); // Use response.data, not the full response object
    } catch (error) {
      console.error('Error loading segmentation data:', error);
      setSegmentationError(`Failed to load segmentation data: ${error.message}`);
    } finally {
      setIsLoadingSegmentation(false);
    }
  }, [jobId, status]);

  // Load segmentation data when component mounts or dependencies change
  useEffect(() => {
    loadSegmentationData();
  }, [loadSegmentationData]);

  useEffect(() => {
    if (!mountRef.current || status !== 'completed') return;

    // Initialize Three.js scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    sceneRef.current = scene;

    // Camera setup
    const camera = new THREE.PerspectiveCamera(
      75,
      mountRef.current.clientWidth / mountRef.current.clientHeight,
      0.1,
      1000
    );
    camera.position.set(0, 0, 5);
    cameraRef.current = camera;

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Lighting setup
    const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, 5, 5);
    directionalLight.castShadow = true;
    scene.add(directionalLight);

    // Create brain volume representation
    createBrainVolume(scene);

    // Create metastases from segmentation data if available
    if (segmentationData && segmentationData.voxels && segmentationData.voxels.length > 0) {
      createSegmentationMetastases(scene, segmentationData);
    } else if (results?.metastasis_volumes && results.metastasis_volumes.length > 0) {
      // Fallback to volume-based if segmentation data not available
      createVolumeBasedMetastases(scene, results);
    }

    // Animation loop
    const animate = () => {
      animationRef.current = requestAnimationFrame(animate);
      renderer.render(scene, camera);
    };
    animate();

    setLoading(false);

    // Handle window resize
    const handleResize = () => {
      if (mountRef.current && renderer && camera) {
        const width = mountRef.current.clientWidth;
        const height = mountRef.current.clientHeight;
        
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height);
      }
    };

    window.addEventListener('resize', handleResize);

    // Mouse controls for interaction
    let mouseDown = false;
    let mouseX = 0;
    let mouseY = 0;

    const handleMouseDown = (event) => {
      mouseDown = true;
      mouseX = event.clientX;
      mouseY = event.clientY;
    };

    const handleMouseMove = (event) => {
      if (!mouseDown) return;
      
      const deltaX = event.clientX - mouseX;
      const deltaY = event.clientY - mouseY;
      
      scene.rotation.y += deltaX * 0.01;
      scene.rotation.x += deltaY * 0.01;
      
      mouseX = event.clientX;
      mouseY = event.clientY;
    };

    const handleMouseUp = () => {
      mouseDown = false;
    };

    const handleWheel = (event) => {
      camera.position.z += event.deltaY * 0.01;
      camera.position.z = Math.max(2, Math.min(10, camera.position.z));
    };

    renderer.domElement.addEventListener('mousedown', handleMouseDown);
    renderer.domElement.addEventListener('mousemove', handleMouseMove);
    renderer.domElement.addEventListener('mouseup', handleMouseUp);
    renderer.domElement.addEventListener('wheel', handleWheel);

    // Cleanup function
    return () => {
      window.removeEventListener('resize', handleResize);
      
      if (renderer.domElement) {
        renderer.domElement.removeEventListener('mousedown', handleMouseDown);
        renderer.domElement.removeEventListener('mousemove', handleMouseMove);
        renderer.domElement.removeEventListener('mouseup', handleMouseUp);
        renderer.domElement.removeEventListener('wheel', handleWheel);
      }
      
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }
      
      renderer.dispose();
    };
  }, [jobId, status, results, segmentationData]);

  // Update opacity when slider changes
  useEffect(() => {
    if (sceneRef.current) {
      sceneRef.current.traverse((child) => {
        if (child.material) {
          if (child.userData.type === 'brain') {
            child.material.opacity = showBrain ? opacity * 0.3 : 0;
          } else if (child.userData.type === 'metastasis') {
            child.material.opacity = showMetastases ? opacity : 0;
          }
          child.material.transparent = true;
          child.visible = child.material.opacity > 0;
        }
      });
    }
  }, [opacity, showBrain, showMetastases]);

  const createBrainVolume = (scene) => {
    // Create a brain-like shape using multiple spheres and ellipsoids
    const brainGroup = new THREE.Group();
    brainGroup.userData.type = 'brain';

    // Main brain volume (ellipsoid)
    const brainGeometry = new THREE.SphereGeometry(1.5, 32, 32);
    brainGeometry.scale(1.2, 1.0, 1.4); // Make it more brain-like
    
    const brainMaterial = new THREE.MeshPhongMaterial({
      color: 0x888888,
      transparent: true,
      opacity: 0.2,
      wireframe: false
    });
    
    const brainMesh = new THREE.Mesh(brainGeometry, brainMaterial);
    brainMesh.userData.type = 'brain';
    brainGroup.add(brainMesh);

    // Add some texture with smaller spheres to simulate brain structure
    for (let i = 0; i < 20; i++) {
      const smallSphere = new THREE.SphereGeometry(0.1, 8, 8);
      const smallMaterial = new THREE.MeshPhongMaterial({
        color: 0xaaaaaa,
        transparent: true,
        opacity: 0.1
      });
      const smallMesh = new THREE.Mesh(smallSphere, smallMaterial);
      smallMesh.userData.type = 'brain';
      
      // Random positions within brain volume
      smallMesh.position.set(
        (Math.random() - 0.5) * 2.5,
        (Math.random() - 0.5) * 2.0,
        (Math.random() - 0.5) * 2.8
      );
      
      brainGroup.add(smallMesh);
    }

    scene.add(brainGroup);
  };

  const createSegmentationMetastases = (scene, segmentationData) => {
    // console.log('Creating segmentation-based metastases:', segmentationData);
    
    try {
      const metastasesGroup = new THREE.Group();
      metastasesGroup.userData.type = 'metastases';

      // Group voxels by class
      const voxelsByClass = {};
      
      segmentationData.voxels.forEach(voxel => {
        const classId = String(voxel.class); // Ensure classId is string
        if (!voxelsByClass[classId]) {
          voxelsByClass[classId] = [];
        }
        voxelsByClass[classId].push(voxel);
      });

      // Create instances for each class
      Object.keys(voxelsByClass).forEach(classId => {
        const classVoxels = voxelsByClass[classId];
        const classInfo = segmentationData.classes[classId];
        
        if (!classInfo || classId == 0 || classId === '0') return; // Skip background class
        
        console.log(`Creating ${classVoxels.length} voxels for class ${classId} (${classInfo.name})`);
        
        // Create instanced geometry for performance
        const geometry = new THREE.BoxGeometry(0.05, 0.05, 0.05); // Small voxel size
        const material = new THREE.MeshLambertMaterial({
          color: new THREE.Color(classInfo.color),
          transparent: true,
          opacity: 0.8
        });

        const instancedMesh = new THREE.InstancedMesh(geometry, material, classVoxels.length);
        instancedMesh.userData.type = 'metastasis';
        instancedMesh.userData.class = classId;
        instancedMesh.userData.className = classInfo.name;

        // Position each instance
        const dummy = new THREE.Object3D();
        classVoxels.forEach((voxel, index) => {
          // Scale coordinates to fit in our scene (-2 to 2 range)
          const scaleX = 4.0 / (segmentationData.metadata?.shape?.[0] || 32);
          const scaleY = 4.0 / (segmentationData.metadata?.shape?.[1] || 32); 
          const scaleZ = 4.0 / (segmentationData.metadata?.shape?.[2] || 32);
          
          dummy.position.set(
            (voxel.x - (segmentationData.metadata?.shape?.[0] || 32) / 2) * scaleX,
            (voxel.y - (segmentationData.metadata?.shape?.[1] || 32) / 2) * scaleY,
            (voxel.z - (segmentationData.metadata?.shape?.[2] || 32) / 2) * scaleZ
          );
          
          dummy.updateMatrix();
          instancedMesh.setMatrixAt(index, dummy.matrix);
        });
        
        instancedMesh.instanceMatrix.needsUpdate = true;
        metastasesGroup.add(instancedMesh);
      });

      scene.add(metastasesGroup);
    } catch (error) {
      console.error('Error creating segmentation metastases:', error);
      // Fall back to volume-based rendering if segmentation fails
      if (results?.metastasis_volumes && results.metastasis_volumes.length > 0) {
        createVolumeBasedMetastases(scene, results);
      }
    }
  };

  const createVolumeBasedMetastases = (scene, results) => {
    console.log('Creating volume-based metastases (fallback)');
    
    const metastasesGroup = new THREE.Group();
    metastasesGroup.userData.type = 'metastases';

    const colors = [
      0xff4444, // Red
      0x44ff44, // Green
      0x4444ff, // Blue
      0xffff44, // Yellow
      0xff44ff, // Magenta
      0x44ffff, // Cyan
      0xff8844, // Orange
      0x8844ff, // Purple
      0x44ff88, // Light Green
      0xff4488  // Pink
    ];

    results.metastasis_volumes.forEach((volume, index) => {
      // Size based on volume (scale appropriately for constant display)
      const radius = Math.max(0.05, Math.min(0.3, volume * 0.1));
      
      // Create segmentation-style geometry (more angular/voxel-like)
      const geometry = new THREE.BoxGeometry(radius * 2, radius * 2, radius * 2);
      const material = new THREE.MeshLambertMaterial({
        color: colors[index % colors.length],
        transparent: true,
        opacity: 0.9,
      });
      
      const metastasis = new THREE.Mesh(geometry, material);
      metastasis.userData.type = 'metastasis';
      metastasis.userData.volume = volume;
      metastasis.userData.index = index;

      // Random but realistic positions within brain volume
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.random() * Math.PI;
      const r = 0.5 + Math.random() * 1.0; // Within brain radius
      
      metastasis.position.set(
        r * Math.sin(phi) * Math.cos(theta),
        r * Math.cos(phi) * 0.8, // Slightly flattened
        r * Math.sin(phi) * Math.sin(theta)
      );

      metastasesGroup.add(metastasis);
    });

    scene.add(metastasesGroup);
  };

  if (status !== 'completed') {
    return (
      <Box sx={{ textAlign: 'center', py: 4 }}>
        <Typography color="text.secondary">
          3D visualization will be available when processing is complete
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Interactive 3D Brain Visualization
      </Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        Interactive 3D view of the brain with actual segmentation results from the model. 
        Drag to rotate, scroll to zoom.
      </Typography>

      {/* Loading/Error States for Segmentation */}
      {isLoadingSegmentation && (
        <Alert severity="info" sx={{ mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <CircularProgress size={20} />
            Loading segmentation data...
          </Box>
        </Alert>
      )}

      {segmentationError && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          {segmentationError}. Showing volume-based approximation instead.
        </Alert>
      )}

      {segmentationData && (
        <Alert severity="success" sx={{ mb: 2 }}>
          Showing actual segmentation with {segmentationData.total_voxels || 0} voxels
          {segmentationData.metadata?.downsample_factor && ` (downsampled by ${segmentationData.metadata.downsample_factor}x)`}
        </Alert>
      )}

      {/* Controls */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6}>
            <FormControlLabel
              control={
                <Switch
                  checked={showBrain}
                  onChange={(e) => setShowBrain(e.target.checked)}
                />
              }
              label="Show Brain Volume"
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <FormControlLabel
              control={
                <Switch
                  checked={showMetastases}
                  onChange={(e) => setShowMetastases(e.target.checked)}
                />
              }
              label="Show Metastases"
            />
          </Grid>
          <Grid item xs={12}>
            <Typography gutterBottom>Opacity: {(opacity * 100).toFixed(0)}%</Typography>
            <Slider
              value={opacity}
              onChange={(e, newValue) => setOpacity(newValue)}
              min={0.1}
              max={1.0}
              step={0.1}
              marks={[
                { value: 0.1, label: '10%' },
                { value: 0.5, label: '50%' },
                { value: 1.0, label: '100%' }
              ]}
            />
          </Grid>
        </Grid>
      </Paper>

      {/* Segmentation Classes Legend */}
      {segmentationData && segmentationData.classes && (
        <Paper sx={{ p: 2, mb: 3 }}>
          <Typography variant="subtitle1" gutterBottom>
            Segmentation Classes
          </Typography>
          <Grid container spacing={1}>
            {Object.entries(segmentationData.classes).map(([classId, classInfo]) => {
              if (classId == 0) return null; // Skip background
              return (
                <Grid item key={classId}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Box
                      sx={{
                        width: 16,
                        height: 16,
                        backgroundColor: classInfo.color,
                        border: '1px solid rgba(0,0,0,0.2)'
                      }}
                    />
                    <Typography variant="caption">
                      {classInfo.name || 'Unknown'} ({classInfo.voxel_count || 0} voxels)
                    </Typography>
                  </Box>
                </Grid>
              );
            })}
          </Grid>
        </Paper>
      )}

      {/* Fallback Volume Legend */}
      {!segmentationData && results?.metastasis_volumes && results.metastasis_volumes.length > 0 && (
        <Paper sx={{ p: 2, mb: 3 }}>
          <Typography variant="subtitle1" gutterBottom>
            Metastases Legend (Volume-based)
          </Typography>
          <Grid container spacing={1}>
            {results.metastasis_volumes.map((volume, index) => (
              <Grid item key={index}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Box
                    sx={{
                      width: 16,
                      height: 16,
                      borderRadius: '50%',
                      backgroundColor: `hsl(${(index * 36) % 360}, 70%, 50%)`,
                      border: '1px solid rgba(0,0,0,0.2)'
                    }}
                  />
                  <Typography variant="caption">
                    Met {index + 1}: {volume.toFixed(2)} cm³
                  </Typography>
                </Box>
              </Grid>
            ))}
          </Grid>
        </Paper>
      )}

      {/* 3D Viewer Container */}
      <Paper sx={{ position: 'relative', height: 500, overflow: 'hidden' }}>
        {loading && (
          <Box
            sx={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              zIndex: 2
            }}
          >
            <CircularProgress />
            <Typography variant="caption" display="block" sx={{ mt: 1 }}>
              Loading 3D visualization...
            </Typography>
          </Box>
        )}
        
        {error && (
          <Alert severity="error" sx={{ m: 2 }}>
            {error}
          </Alert>
        )}

        <Box
          ref={mountRef}
          sx={{
            width: '100%',
            height: '100%',
            cursor: 'grab',
            '&:active': {
              cursor: 'grabbing'
            }
          }}
        />

        {/* Instructions */}
        <Typography
          variant="caption"
          sx={{
            position: 'absolute',
            bottom: 10,
            left: 10,
            backgroundColor: 'rgba(0,0,0,0.7)',
            color: 'white',
            padding: '4px 8px',
            borderRadius: 1,
            userSelect: 'none'
          }}
        >
          Drag to rotate • Scroll to zoom
        </Typography>

        {/* Data source indicator */}
        <Typography
          variant="caption"
          sx={{
            position: 'absolute',
            bottom: 10,
            right: 10,
            backgroundColor: 'rgba(0,0,0,0.7)',
            color: 'white',
            padding: '4px 8px',
            borderRadius: 1,
            userSelect: 'none'
          }}
        >
          {segmentationData ? 'Actual Segmentation' : 'Volume Approximation'} • {results?.metastasis_count || 0} detected
        </Typography>
      </Paper>

      {/* Instructions */}
      <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
        Use the controls above to adjust visibility. Click and drag to manually rotate the view.
        {segmentationData ? ' Showing actual voxel-level segmentation from the model.' : ' Segmentation data loading or unavailable - showing volume approximation.'}
      </Typography>
    </Box>
  );
};

export default Interactive3DViewer;
