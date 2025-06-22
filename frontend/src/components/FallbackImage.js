import React from 'react';
import { Box, Typography, CircularProgress } from '@mui/material';

// Fallback component when a scan image cannot be loaded
const FallbackImage = ({ jobId, message, height = 400 }) => {
  return (
    <Box
      sx={{
        height: height,
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        bgcolor: 'background.paper',
        border: '1px solid',
        borderColor: 'divider',
        borderRadius: 1,
        p: 2
      }}
    >
      <Typography variant="h6" color="text.secondary" gutterBottom>
        Image not available
      </Typography>
      
      <Typography variant="body2" color="text.secondary" align="center">
        {message || "The image could not be loaded."}
      </Typography>
      
      {jobId && (
        <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
          Job ID: {jobId}
        </Typography>
      )}
    </Box>
  );
};

// Loading placeholder
export const LoadingImage = ({ height = 400 }) => {
  return (
    <Box
      sx={{
        height: height,
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        bgcolor: 'background.paper',
        border: '1px solid',
        borderColor: 'divider',
        borderRadius: 1
      }}
    >
      <CircularProgress size={40} />
      <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
        Loading visualization...
      </Typography>
    </Box>
  );
};

export default FallbackImage;
