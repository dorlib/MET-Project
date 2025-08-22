import React from 'react';
import { Alert, AlertTitle, Box, Button } from '@mui/material';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    // Update state so the next render will show the fallback UI
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    // Log the error to console
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    this.setState({ 
      error: error,
      errorInfo: errorInfo 
    });
  }

  handleReload = () => {
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      // You can render any custom fallback UI
      return (
        <Box sx={{ p: 4 }}>
          <Alert severity="error">
            <AlertTitle>Something went wrong</AlertTitle>
            An unexpected error occurred in the application. This might be due to:
            <ul>
              <li>A temporary network issue</li>
              <li>An outdated cached version of the app</li>
              <li>A JavaScript error in the code</li>
            </ul>
            
            <Box sx={{ mt: 2 }}>
              <Button 
                variant="contained" 
                onClick={this.handleReload}
                sx={{ mr: 2 }}
              >
                Reload Page
              </Button>
              <Button 
                variant="outlined" 
                onClick={() => this.setState({ hasError: false, error: null, errorInfo: null })}
              >
                Try Again
              </Button>
            </Box>
            
            {process.env.NODE_ENV === 'development' && this.state.error && (
              <details style={{ marginTop: '16px', whiteSpace: 'pre-wrap' }}>
                <summary>Error Details (Development)</summary>
                {this.state.error && this.state.error.toString()}
                <br />
                {this.state.errorInfo.componentStack}
              </details>
            )}
          </Alert>
        </Box>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
