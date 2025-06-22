import React, { useState, useMemo } from 'react';
import { Container, CssBaseline, ThemeProvider, createTheme, Box, IconButton } from '@mui/material';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';
import Header from './components/Header';
import UploadForm from './components/UploadForm';
import ResultViewer from './components/ResultViewer';
import AuthPage from './components/AuthPage';
import UserProfile from './components/UserProfile';
import ErrorNotification from './components/ErrorNotification';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import api from './services/api';

// Create theme context for toggling between light and dark modes
export const ColorModeContext = React.createContext({ toggleColorMode: () => {} });

// Theme options
const getThemeOptions = (mode) => ({
  palette: {
    mode,
    ...(mode === 'light' 
      ? {
          // Light mode - bright and optimistic colors
          primary: {
            main: '#2196f3', // Bright blue
            light: '#64b5f6',
            dark: '#1976d2',
          },
          secondary: {
            main: '#ff9800', // Energetic orange
            light: '#ffb74d',
            dark: '#f57c00',
          },
          background: {
            default: '#f5f7fa', // Light gray-blue background
            paper: '#ffffff',
          },
          text: {
            primary: '#333333',
            secondary: '#5f6368',
          },
          success: {
            main: '#4caf50', // Cheerful green
          },
          info: {
            main: '#03a9f4', // Light blue
          },
          warning: {
            main: '#ff9800', // Warm orange
          },
          error: {
            main: '#f44336', // Soft red
          },
        } 
      : {
          // Dark mode - sophisticated dark theme
          primary: {
            main: '#90caf9', // Light blue
            light: '#c3fdff',
            dark: '#5d99c6',
          },
          secondary: {
            main: '#ffab40', // Amber
            light: '#ffdd71',
            dark: '#c97b00',
          },
          background: {
            default: '#121212', // Material dark theme recommended bg
            paper: '#1e1e1e',
          },
          text: {
            primary: '#ffffff',
            secondary: '#b3b3b3',
          },
        }),
  },
  typography: {
    fontFamily: "'Roboto', 'Helvetica', 'Arial', sans-serif",
    h1: {
      fontWeight: 500,
    },
    h2: {
      fontWeight: 500,
    },
  },
  shape: {
    borderRadius: 8,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none',
          fontWeight: 500,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: mode === 'light' 
            ? '0 4px 12px rgba(0,0,0,0.05)' 
            : '0 4px 12px rgba(0,0,0,0.25)',
        },
      },
    },
  },
});

// Main application content
const MainContent = () => {
  const [jobId, setJobId] = useState(null);
  const [processingStatus, setProcessingStatus] = useState(null);
  const [results, setResults] = useState(null);
  const [currentView, setCurrentView] = useState('upload'); // 'upload', 'profile', 'auth'
  const { currentUser } = useAuth();
  const { toggleColorMode, mode } = React.useContext(ColorModeContext);
  
  // Error notification state
  const [errorMessage, setErrorMessage] = useState('');
  const [showError, setShowError] = useState(false);
  const [errorSeverity, setErrorSeverity] = useState('error'); // Default to error severity

  // Check results status periodically if a job is in progress
  React.useEffect(() => {
    // Don't poll if we don't have a job ID or if job is already completed
    if (!jobId || processingStatus === 'completed') return;
    
    // Also stop polling for terminal states that won't change
    const terminalStates = ['not_found', 'failed', 'error'];
    if (terminalStates.includes(processingStatus)) {
      console.log(`Job ${jobId} is in terminal state: ${processingStatus}. Stopping polling.`);
      return;
    }
    
    console.log(`Setting up polling interval for job ${jobId} with status ${processingStatus}`);
    
    // Use an initial polling interval of 3 seconds for more responsive updates
    const baseInterval = 3000; // 3 seconds
    const extendedInterval = 8000; // 8 seconds
    const maxPollingTime = 5 * 60 * 1000; // 5 minutes maximum polling
    const startTime = Date.now();
    const timeThreshold = 30000; // 30 seconds
    
    // Track active API calls to prevent overlapping requests
    let isRequestActive = false;
    let currentInterval = baseInterval;
    let failedAttempts = 0;
    const maxFailedAttempts = 5; // Increased max attempts
    
    const interval = setInterval(async () => {
      // Check if we've been polling for too long (to prevent indefinite polling)
      if (Date.now() - startTime > maxPollingTime) {
        console.log(`Reached maximum polling time for job ${jobId}. Stopping.`);
        clearInterval(interval);
        setErrorMessage("Processing is taking longer than expected. Please check back later.");
        setShowError(true);
        return;
      }
      
      // Skip this interval if there's already a request in progress
      if (isRequestActive) {
        console.log("Skipping results check - previous request still active");
        return;
      }
      
      try {
        isRequestActive = true;
        
        // Determine if we should use the longer polling interval
        const currentTime = Date.now();
        if (currentTime - startTime > timeThreshold && currentInterval === baseInterval) {
          // If we've been polling for more than a minute, update the interval
          clearInterval(interval);
          console.log("Switching to longer polling interval");
          currentInterval = extendedInterval;
          
          // Create a new interval with the extended time
          setInterval(async () => {
            // This is the same code as in the main interval function
            if (isRequestActive) return;
            
            try {
              isRequestActive = true;
              console.log(`Checking status for job ${jobId} (extended interval)...`);
              const response = await api.getResults(jobId);
              const data = response.data;
              
              failedAttempts = 0;
              setProcessingStatus(data.status);
              
              if (data.status === 'completed' || terminalStates.includes(data.status)) {
                if (data.status === 'completed') {
                  setResults(data);
                } else if (data.status === 'not_found') {
                  setErrorMessage("The requested job could not be found. It may have been deleted or expired.");
                  setErrorSeverity('warning');
                  setShowError(true);
                } else {
                  setErrorMessage("Processing failed. Please try again or contact support if the issue persists.");
                  setErrorSeverity('error');
                  setShowError(true);
                }
                return; // No need to continue polling
              }
            } catch (error) {
              console.error("Failed to check job status in extended interval:", error);
              failedAttempts++;
            } finally {
              isRequestActive = false;
            }
          }, extendedInterval);
          
          // Return from this function to stop executing the original interval
          return;
        }
        
        console.log(`Checking status for job ${jobId}...`);
        const response = await api.getResults(jobId);
        const data = response.data;
        
        // Reset failed attempts counter on success
        failedAttempts = 0;
        
        // Update processing status
        setProcessingStatus(data.status);
        
        // Handle different statuses
        if (data.status === 'completed') {
          console.log("Processing completed, updating results");
          setResults(data);
          clearInterval(interval);
        } else if (terminalStates.includes(data.status)) {
          console.log(`Job ${jobId} reached terminal state: ${data.status}. Stopping polling.`);
          clearInterval(interval);
          
          // Show appropriate error messages
          if (data.status === 'not_found') {
            setErrorMessage("The requested job could not be found. It may have been deleted or expired.");
            setErrorSeverity('warning'); // Use warning for not found
          } else if (data.status === 'failed' || data.status === 'error') {
            setErrorMessage("Processing failed. Please try again or contact support if the issue persists.");
            setErrorSeverity('error'); // Use error for processing failure
          }
          setShowError(true);
        }
      } catch (error) {
        console.error("Failed to check job status:", error);
        
        // Increment failed attempts counter
        failedAttempts++;
        
        // If we've failed too many times in a row, stop polling
        if (failedAttempts >= maxFailedAttempts) {
          console.log(`Too many failed attempts (${failedAttempts}). Stopping polling.`);
          clearInterval(interval);
          setErrorMessage("Failed to check processing status. Please try again later.");
          setShowError(true);
        }
      } finally {
        isRequestActive = false;
      }
    }, baseInterval); // Start with checking every 5 seconds
    
    return () => clearInterval(interval);
  }, [jobId, processingStatus]);

  // Handle file upload success
  const handleUploadSuccess = (data) => {
    setJobId(data.job_id);
    setProcessingStatus(data.status);
    setResults(null);
    setCurrentView('upload'); // Switch to upload view to show results
    
    // Show a notification that processing has started
    setErrorMessage("Scan uploaded successfully. Processing has begun.");
    setShowError(true);
    setErrorSeverity('success'); // Use success severity for this notification
  };

  // Handle navigation to different views
  const navigateTo = (view) => {
    setCurrentView(view);
  };

  // Handle viewing a specific scan from history
  const handleViewScan = (scanJobId) => {
    setJobId(scanJobId);
    
    // Fetch the scan results
    api.getResults(scanJobId)
      .then(response => {
        setResults(response.data);
        setProcessingStatus(response.data.status);
        setCurrentView('upload'); // Switch to upload view to show results
      })
      .catch(error => {
        console.error("Failed to fetch scan results:", error);
        setErrorMessage("Failed to fetch scan results. Please try again later.");
        setShowError(true);
      });
  };
  
  // Handle error notification close
  const handleErrorClose = (event, reason) => {
    if (reason === 'clickaway') return;
    setShowError(false);
    // Reset error severity to default after notification is closed
    setTimeout(() => setErrorSeverity('error'), 300);
  };

  // Render the current view
  const renderCurrentView = () => {
    if (currentView === 'auth') {
      return <AuthPage />;
    }
    
    if (currentView === 'profile') {
      return <UserProfile onViewScan={handleViewScan} />;
    }
    
    // Default to upload/results view
    return (
      <>
        <UploadForm onUploadSuccess={handleUploadSuccess} />
        {jobId && (
          <ResultViewer 
            jobId={jobId} 
            status={processingStatus} 
            results={results} 
          />
        )}
      </>
    );
  };

  return (
    <Box>
      <Header 
        isAuthenticated={!!currentUser} 
        userName={currentUser?.name}
        onNavigate={navigateTo}
        currentView={currentView}
      />
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        {renderCurrentView()}
      </Container>
      
      {/* Global error notification */}
      <ErrorNotification
        message={errorMessage}
        open={showError}
        onClose={handleErrorClose}
        duration={6000}
        severity={errorSeverity}
      />
    </Box>
  );
};

function App() {
  // State for the color mode
  const [mode, setMode] = React.useState('light');

  // Color mode context to enable toggling from any component
  const colorMode = React.useMemo(
    () => ({
      toggleColorMode: () => {
        setMode((prevMode) => (prevMode === 'light' ? 'dark' : 'light'));
      },
      mode,
    }),
    [mode],
  );

  // Create theme based on current mode
  const theme = React.useMemo(
    () => createTheme(getThemeOptions(mode)),
    [mode],
  );

  return (
    <ColorModeContext.Provider value={colorMode}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <AuthProvider>
          <MainContent />
        </AuthProvider>
      </ThemeProvider>
    </ColorModeContext.Provider>
  );
}

export default App;
