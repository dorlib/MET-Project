import React, { useState, useEffect } from 'react';
import {
  Box,
  LinearProgress,
  Typography,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  Alert
} from '@mui/material';
import {
  CheckCircle,
  Cancel,
  Warning
} from '@mui/icons-material';
import { green, red, orange, grey } from '@mui/material/colors';

const PasswordStrengthIndicator = ({ 
  password, 
  onValidationChange,
  showRequirements = true,
  compact = false 
}) => {
  const [validation, setValidation] = useState(null);
  const [requirements, setRequirements] = useState(null);
  const [loading, setLoading] = useState(false);

  // Fetch password requirements on component mount
  useEffect(() => {
    const fetchRequirements = async () => {
      try {
        const response = await fetch('/api/auth/password-requirements');
        if (response.ok) {
          const data = await response.json();
          setRequirements(data);
        }
      } catch (error) {
        console.error('Error fetching password requirements:', error);
      }
    };

    fetchRequirements();
  }, []);

  // Validate password whenever it changes
  useEffect(() => {
    if (!password || password.length === 0) {
      setValidation(null);
      if (onValidationChange) {
        onValidationChange({ is_valid: false, strength: 0 });
      }
      return;
    }

    const validatePassword = async () => {
      setLoading(true);
      try {
        const response = await fetch('/api/auth/validate-password', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ password }),
        });

        if (response.ok) {
          const data = await response.json();
          setValidation(data);
          if (onValidationChange) {
            onValidationChange(data);
          }
        }
      } catch (error) {
        console.error('Error validating password:', error);
      } finally {
        setLoading(false);
      }
    };

    // Debounce password validation
    const timeoutId = setTimeout(validatePassword, 300);
    return () => clearTimeout(timeoutId);
  }, [password, onValidationChange]);

  const getStrengthColor = (strength) => {
    if (strength >= 6) return green[500];
    if (strength >= 4) return orange[500];
    return red[500];
  };

  const getStrengthLabel = (strength) => {
    if (strength >= 6) return 'Strong';
    if (strength >= 4) return 'Medium';
    if (strength >= 2) return 'Weak';
    return 'Very Weak';
  };

  const getStrengthProgress = (strength) => {
    return (strength / 7) * 100; // Max strength is 7
  };

  const RequirementItem = ({ met, text, warning = false }) => {
    const icon = met ? (
      <CheckCircle sx={{ color: green[500] }} />
    ) : warning ? (
      <Warning sx={{ color: orange[500] }} />
    ) : (
      <Cancel sx={{ color: red[500] }} />
    );

    return (
      <ListItem dense sx={{ py: 0 }}>
        <ListItemIcon sx={{ minWidth: 32 }}>
          {icon}
        </ListItemIcon>
        <ListItemText
          primary={text}
          primaryTypographyProps={{
            variant: 'body2',
            color: met ? 'success.main' : warning ? 'warning.main' : 'error.main'
          }}
        />
      </ListItem>
    );
  };

  if (compact) {
    return (
      <Box sx={{ width: '100%', mt: 1 }}>
        {validation && (
          <>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
              <LinearProgress
                variant="determinate"
                value={getStrengthProgress(validation.strength_score)}
                sx={{
                  flexGrow: 1,
                  height: 6,
                  borderRadius: 3,
                  backgroundColor: grey[300],
                  '& .MuiLinearProgress-bar': {
                    backgroundColor: getStrengthColor(validation.strength_score),
                    borderRadius: 3,
                  },
                }}
              />
              <Chip
                label={getStrengthLabel(validation.strength_score)}
                size="small"
                sx={{
                  backgroundColor: getStrengthColor(validation.strength_score),
                  color: 'white',
                  fontWeight: 'bold',
                }}
              />
            </Box>
            {!validation.is_valid && (
              <Typography variant="caption" color="error" display="block">
                Password does not meet requirements
              </Typography>
            )}
          </>
        )}
      </Box>
    );
  }

  return (
    <Box sx={{ width: '100%', mt: 2 }}>
      {validation && (
        <Box sx={{ mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
            <Typography variant="subtitle2" color="textSecondary">
              Password Strength
            </Typography>
            <Chip
              label={getStrengthLabel(validation.strength_score)}
              size="small"
              sx={{
                backgroundColor: getStrengthColor(validation.strength_score),
                color: 'white',
                fontWeight: 'bold',
              }}
            />
          </Box>
          <LinearProgress
            variant="determinate"
            value={getStrengthProgress(validation.strength_score)}
            sx={{
              height: 8,
              borderRadius: 4,
              backgroundColor: grey[300],
              '& .MuiLinearProgress-bar': {
                backgroundColor: getStrengthColor(validation.strength_score),
                borderRadius: 4,
              },
            }}
          />
          <Typography variant="caption" color="textSecondary" sx={{ mt: 1, display: 'block' }}>
            Score: {validation.strength_score}/7
          </Typography>
        </Box>
      )}

      {showRequirements && requirements && (
        <Box>
          <Typography variant="subtitle2" gutterBottom>
            Password Requirements
          </Typography>
          <List dense sx={{ bgcolor: 'background.paper', borderRadius: 1, border: 1, borderColor: 'divider' }}>
            <RequirementItem
              met={validation?.checks?.min_length || false}
              text={`At least ${requirements.min_length} characters`}
            />
            <RequirementItem
              met={validation?.checks?.has_uppercase || false}
              text="Contains uppercase letters"
            />
            <RequirementItem
              met={validation?.checks?.has_lowercase || false}
              text="Contains lowercase letters"
            />
            <RequirementItem
              met={validation?.checks?.has_number || false}
              text="Contains numbers"
            />
            <RequirementItem
              met={validation?.checks?.has_special || false}
              text="Contains special characters"
            />
            <RequirementItem
              met={validation?.checks?.not_common || false}
              text="Not a common password"
              warning={validation?.checks?.not_common === false}
            />
            <RequirementItem
              met={validation?.checks?.no_sequences || false}
              text="No sequential characters (abc, 123)"
              warning={validation?.checks?.no_sequences === false}
            />
          </List>

          {validation && !validation.is_valid && validation.message && (
            <Alert severity="warning" sx={{ mt: 2 }}>
              {validation.message}
            </Alert>
          )}
        </Box>
      )}

      {loading && (
        <Box sx={{ mt: 1 }}>
          <LinearProgress size="sm" />
        </Box>
      )}
    </Box>
  );
};

export default PasswordStrengthIndicator;
