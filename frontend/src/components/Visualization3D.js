import React, { useRef, useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Html } from '@react-three/drei';
import * as THREE from 'three';
import { Box, Typography, Slider, FormControlLabel, Switch, Paper } from '@mui/material';

/**
 * 3D Brain and Metastasis Visualization component using Three.js
 * 
 * @param {Object} props
 * @param {Object} props.segmentation - Segmentation data
 * @param {Object} props.originalImage - Original MRI data
 * @param {Array} props.metastases - Array of metastasis objects with position and volume
 */
const Visualization3D = ({ segmentation, originalImage, metastases = [] }) => {
  const [opacity, setOpacity] = useState(0.7);
  const [showOriginal, setShowOriginal] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [rotationSpeed, setRotationSpeed] = useState(0);
  
  // Create a simple brain mesh for demonstration
  // In a real implementation, this would use actual segmentation data
  const BrainModel = () => {
    const mesh = useRef();
    
    // Rotate the brain based on rotationSpeed
    useEffect(() => {
      if (rotationSpeed === 0) return;
      
      const interval = setInterval(() => {
        if (mesh.current) {
          mesh.current.rotation.y += rotationSpeed * 0.01;
        }
      }, 30);
      
      return () => clearInterval(interval);
    }, [rotationSpeed]);
    
    return (
      <mesh
        ref={mesh}
        scale={1.5}
      >
        <sphereGeometry args={[2, 16, 16]} />
        <meshStandardMaterial 
          color="#d1cfcf" 
          transparent={true} 
          opacity={showOriginal ? opacity : 0}
        />
      </mesh>
    );
  };
  
  // Create metastasis representations
  const Metastasis = ({ position, volume, index }) => {
    // Scale based on volume with minimum size for visibility
    const radius = Math.max(0.2, Math.cbrt(volume) * 0.2);
    
    return (
      <group position={position}>
        <mesh>
          <sphereGeometry args={[radius, 12, 12]} />
          <meshStandardMaterial 
            color="#ff4444" 
            transparent={true}
            opacity={0.8}
          />
        </mesh>
        
        {showLabels && (
          <Html distanceFactor={10}>
            <div style={{ 
              background: 'rgba(0,0,0,0.6)', 
              color: 'white',
              padding: '4px 8px',
              borderRadius: '4px',
              transform: 'translate3d(-50%, -50%, 0)'
            }}>
              {`#${index+1}: ${volume.toFixed(2)}ml`}
            </div>
          </Html>
        )}
      </group>
    );
  };

  // Demo metastasis positions - in production, these would come from actual data
  const demoMetastases = metastases.length > 0 ? metastases : [
    { position: [1.2, 0.5, 0], volume: 1.2 },
    { position: [-0.8, 1.2, 0], volume: 0.7 },
    { position: [0.3, -1.0, 0.6], volume: 0.4 },
  ];
  
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <Paper elevation={2} sx={{ p: 2, mb: 2 }}>
        <Typography variant="h6" gutterBottom>
          3D Visualization Controls
        </Typography>
        
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
          <Box sx={{ width: 200 }}>
            <Typography id="opacity-slider" gutterBottom>
              Brain Opacity
            </Typography>
            <Slider
              aria-labelledby="opacity-slider"
              value={opacity}
              min={0}
              max={1}
              step={0.05}
              onChange={(_, newValue) => setOpacity(newValue)}
              valueLabelDisplay="auto"
            />
          </Box>
          
          <Box sx={{ width: 200 }}>
            <Typography id="rotation-slider" gutterBottom>
              Rotation Speed
            </Typography>
            <Slider
              aria-labelledby="rotation-slider"
              value={rotationSpeed}
              min={0}
              max={5}
              step={0.5}
              onChange={(_, newValue) => setRotationSpeed(newValue)}
              valueLabelDisplay="auto"
            />
          </Box>
          
          <Box>
            <FormControlLabel
              control={
                <Switch
                  checked={showOriginal}
                  onChange={(e) => setShowOriginal(e.target.checked)}
                />
              }
              label="Show Brain"
            />
          </Box>
          
          <Box>
            <FormControlLabel
              control={
                <Switch
                  checked={showLabels}
                  onChange={(e) => setShowLabels(e.target.checked)}
                />
              }
              label="Show Labels"
            />
          </Box>
        </Box>
      </Paper>
      
      <Box sx={{ flexGrow: 1, position: 'relative', bgcolor: '#f5f5f5', borderRadius: 1 }}>
        <Canvas camera={{ position: [0, 0, 6], fov: 50 }}>
          <ambientLight intensity={0.5} />
          <pointLight position={[10, 10, 10]} intensity={1} />
          <pointLight position={[-10, -10, -10]} intensity={0.5} />
          
          <BrainModel />
          
          {/* Render metastases */}
          {demoMetastases.map((met, index) => (
            <Metastasis 
              key={index} 
              position={met.position}
              volume={met.volume}
              index={index}
            />
          ))}
          
          <OrbitControls 
            enableZoom={true}
            enablePan={true}
            enableRotate={true}
            autoRotate={rotationSpeed > 0}
            autoRotateSpeed={rotationSpeed * 2}
          />
        </Canvas>
        
        <Typography 
          variant="caption" 
          sx={{ 
            position: 'absolute', 
            bottom: 10, 
            right: 10, 
            color: 'text.secondary',
            backgroundColor: 'rgba(255,255,255,0.7)',
            px: 1,
            borderRadius: 1
          }}
        >
          Use mouse to rotate, scroll to zoom
        </Typography>
      </Box>
    </Box>
  );
};

export default Visualization3D;
