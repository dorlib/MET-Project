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
import api from '../services/api';

/**
 * Component for setting up two-factor authentication
 * 
 * @param {Object} props - Component props
 * @param {boolean} props.open - Whether the dialog is open
 * @param {Function} props.onClose - Function to call when closing the dialog
 */
const TwoFactorSetup = ({ open, onClose }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [step, setStep] = useState(1); // 1: Initial setup, 2: Verification
  const [qrCode, setQrCode] = useState(null);
  const [secret, setSecret] = useState(null);
  const [verificationCode, setVerificationCode] = useState('');
  
  const handleSetup = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await api.setup2FA();
      setQrCode(response.data.qr_code);
      setSecret(response.data.secret);
      setStep(2);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to setup 2FA. Please try again.');
    } finally {
      setLoading(false);
    }
  };
  
  const handleVerify = async () => {
    if (!verificationCode) {
      setError('Verification code is required');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      await api.verify2FASetup(verificationCode);
      onClose(true); // Close with success indicator
    } catch (err) {
      setError(err.response?.data?.error || 'Invalid verification code');
    } finally {
      setLoading(false);
    }
  };
  
  const handleCancel = () => {
    onClose();
    
    // Reset state
    setStep(1);
    setQrCode(null);
    setSecret(null);
    setVerificationCode('');
    setError(null);
  };
  
  return (
    <Dialog open={open} onClose={handleCancel} maxWidth="sm" fullWidth>
      <DialogTitle>
        {step === 1 ? 'Setup Two-Factor Authentication' : 'Verify Setup'}
      </DialogTitle>
      
      <DialogContent>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        
        {step === 1 && (
          <Box sx={{ my: 2 }}>
            <Typography variant="body1" sx={{ mb: 2 }}>
              Two-factor authentication adds an extra layer of security to your account. Once enabled, 
              you'll need to provide a verification code from your authentication app in addition to 
              your password when logging in.
            </Typography>
            
            <Typography variant="body2" color="text.secondary">
              Click "Continue" to begin the setup process. You'll need to install an authentication app 
              like Google Authenticator, Authy, or Microsoft Authenticator.
            </Typography>
          </Box>
        )}
        
        {step === 2 && (
          <Box sx={{ my: 2, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <Typography variant="body1" sx={{ mb: 2 }}>
              Scan this QR code with your authentication app or enter the secret manually.
            </Typography>
            
            {qrCode && (
              <img 
                src={qrCode} 
                alt="QR Code for 2FA" 
                style={{ width: '200px', height: '200px', marginBottom: '1rem' }}
              />
            )}
            
            {secret && (
              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                  Manual entry code:
                </Typography>
                <Typography 
                  variant="body2"
                  sx={{ 
                    fontFamily: 'monospace', 
                    backgroundColor: '#f5f5f5', 
                    p: 1,
                    borderRadius: 1 
                  }}
                >
                  {secret}
                </Typography>
              </Box>
            )}
            
            <TextField
              label="Verification Code"
              value={verificationCode}
              onChange={(e) => setVerificationCode(e.target.value)}
              fullWidth
              placeholder="Enter the 6-digit code from your app"
              sx={{ mb: 2 }}
              inputProps={{ maxLength: 6 }}
            />
            
            <Typography variant="body2" color="text.secondary">
              Enter the 6-digit verification code displayed in your authentication app to complete the setup.
            </Typography>
          </Box>
        )}
      </DialogContent>
      
      <DialogActions>
        <Button onClick={handleCancel} color="primary">
          Cancel
        </Button>
        
        {step === 1 && (
          <Button 
            onClick={handleSetup} 
            color="primary" 
            variant="contained"
            disabled={loading}
          >
            {loading ? <CircularProgress size={24} /> : 'Continue'}
          </Button>
        )}
        
        {step === 2 && (
          <Button 
            onClick={handleVerify} 
            color="primary" 
            variant="contained"
            disabled={loading || !verificationCode}
          >
            {loading ? <CircularProgress size={24} /> : 'Verify'}
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
};

export default TwoFactorSetup;
