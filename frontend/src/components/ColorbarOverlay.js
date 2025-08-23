import React from 'react';
import { Box, Typography } from '@mui/material';

const ColorbarOverlay = ({ tissueClasses = [] }) => {
  // Default tissue classes with viridis-like colors if none provided
  const defaultClasses = [
    { id: 0, name: 'Background', color: '#440154', value: 0 },
    { id: 1, name: 'Class 1', color: '#31688e', value: 1 },
    { id: 2, name: 'Edema', color: '#35b779', value: 2 },
    { id: 3, name: 'Metastasis', color: '#fde725', value: 3 }
  ];

  const classes = tissueClasses.length > 0 ? tissueClasses : defaultClasses;

  return (
    <Box
      sx={{
        position: 'absolute',
        bottom: 16,
        right: 16,
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        borderRadius: 1,
        padding: 1.5,
        minWidth: 180,
        zIndex: 10
      }}
    >
      {/* Colorbar Title */}
      <Typography
        variant="caption"
        sx={{
          color: 'white',
          fontWeight: 'bold',
          fontSize: '14px',
          marginBottom: 1,
          display: 'block',
          textAlign: 'center'
        }}
      >
        Tissue Type
      </Typography>

      {/* Color Legend Items */}
      {classes.map((tissueClass) => (
        <Box
          key={tissueClass.id}
          sx={{
            display: 'flex',
            alignItems: 'center',
            marginBottom: 0.8,
            '&:last-child': {
              marginBottom: 0
            }
          }}
        >
          {/* Color Square */}
          <Box
            sx={{
              width: 24,
              height: 16,
              backgroundColor: tissueClass.color,
              marginRight: 1.5,
              borderRadius: 0.5,
              border: '1px solid rgba(255, 255, 255, 0.3)'
            }}
          />
          
          {/* Class Label */}
          <Typography
            variant="body2"
            sx={{
              color: 'white',
              fontSize: '13px',
              fontWeight: 500,
              minWidth: 80
            }}
          >
            {tissueClass.name}
          </Typography>
          
          {/* Value Indicator */}
          <Typography
            variant="caption"
            sx={{
              color: 'rgba(255, 255, 255, 0.7)',
              fontSize: '11px',
              marginLeft: 'auto'
            }}
          >
            {tissueClass.value}
          </Typography>
        </Box>
      ))}
    </Box>
  );
};

export default ColorbarOverlay;
