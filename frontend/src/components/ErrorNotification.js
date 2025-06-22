import React from 'react';
import { Alert, Snackbar } from '@mui/material';

/**
 * Reusable error notification component
 * 
 * @param {Object} props - Component props
 * @param {string} props.message - Error message to display
 * @param {boolean} props.open - Whether the notification is visible
 * @param {Function} props.onClose - Function to call when closing the notification
 * @param {number} props.duration - Duration in milliseconds before auto-hiding
 */
const ErrorNotification = ({ 
  message, 
  open = false, 
  onClose, 
  duration = 6000, // 6 seconds by default
  severity = 'error'
}) => {
  return (
    <Snackbar
      open={open}
      autoHideDuration={duration}
      onClose={onClose}
      anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
    >
      <Alert 
        onClose={onClose} 
        severity={severity} 
        variant="filled" 
        sx={{ width: '100%' }}
      >
        {message}
      </Alert>
    </Snackbar>
  );
};

export default ErrorNotification;
