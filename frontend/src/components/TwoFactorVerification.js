import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Box,
  Typography,
  CircularProgress,
  Alert
} from '@mui/material';

/**
 * Component for verifying a 2FA code during login
 * 
 * @param {Object} props - Component props
 * @param {boolean} props.open - Whether the dialog is open
 * @param {Function} props.onClose - Function to call when closing the dialog
 * @param {string} props.email - The user's email for verification
 * @param {Function} props.onVerify - Function to call with the verification code
 */
const TwoFactorVerification = ({ open, onClose, email, onVerify }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [verificationCode, setVerificationCode] = useState('');
  
  const handleVerify = async () => {
    if (!verificationCode) {
      setError('Verification code is required');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      await onVerify(verificationCode);
      // The onVerify function will handle success responses
    } catch (err) {
      setError(err.response?.data?.error || 'Invalid verification code');
      setLoading(false);
    }
  };
  
  const handleCancel = () => {
    onClose();
    // Reset state
    setVerificationCode('');
    setError(null);
  };
  
  return (
    <Dialog open={open} onClose={handleCancel} maxWidth="xs" fullWidth>
      <DialogTitle>Two-Factor Authentication</DialogTitle>
      
      <DialogContent>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        
        <Box sx={{ my: 2 }}>
          <Typography variant="body1" sx={{ mb: 2 }}>
            Please enter the verification code from your authentication app for:
          </Typography>
          
          <Typography variant="subtitle1" sx={{ fontWeight: 'bold', mb: 2 }}>
            {email}
          </Typography>
          
          <TextField
            label="Verification Code"
            value={verificationCode}
            onChange={(e) => setVerificationCode(e.target.value)}
            fullWidth
            placeholder="Enter the 6-digit code"
            sx={{ mb: 2 }}
            autoFocus
            inputProps={{ maxLength: 6 }}
          />
        </Box>
      </DialogContent>
      
      <DialogActions>
        <Button onClick={handleCancel} color="primary">
          Cancel
        </Button>
        
        <Button 
          onClick={handleVerify} 
          color="primary" 
          variant="contained"
          disabled={loading || !verificationCode}
        >
          {loading ? <CircularProgress size={24} /> : 'Verify'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default TwoFactorVerification;
