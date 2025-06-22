import React, { createContext, useState, useEffect, useContext } from 'react';
import { CircularProgress } from '@mui/material';
import api from '../services/api';

// Create authentication context
const AuthContext = createContext(null);

// Hook to use the auth context
export const useAuth = () => useContext(AuthContext);

// Provider component that wraps the app and makes auth object available
export const AuthProvider = ({ children }) => {
  const [currentUser, setCurrentUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Check if a user is already logged in (from local storage)
  useEffect(() => {
    const initAuth = async () => {
      const token = localStorage.getItem('authToken');
      const userData = localStorage.getItem('userData');

      if (token && userData) {
        try {
          // Set default auth header for future requests
          api.setAuthHeader(token);

          // Get user profile to validate token
          const response = await api.getUserProfile();
          setCurrentUser({
            ...JSON.parse(userData),
            // Update with any new info from server
            ...response.data
          });
        } catch (err) {
          console.error('Token validation failed:', err);
          // Clear invalid session
          logout();
        }
      }
      setLoading(false);
    };

    initAuth();
  }, []);

  // Login function
  const login = async (email, password) => {
    setLoading(true);
    setError(null);

    try {
      const response = await api.login({ email, password });
      const { token, name, email: userEmail } = response.data;

      // Store auth data
      localStorage.setItem('authToken', token);
      localStorage.setItem('userData', JSON.stringify({ name, email: userEmail }));
      
      // Set default auth header for future requests
      api.setAuthHeader(token);

      // Update state
      setCurrentUser({ name, email: userEmail });
      return { success: true };
    } catch (err) {
      const errorMsg = err.response?.data?.error || 'Login failed';
      setError(errorMsg);
      return { success: false, error: errorMsg };
    } finally {
      setLoading(false);
    }
  };

  // Register function
  const register = async (name, email, password) => {
    setLoading(true);
    setError(null);

    try {
      const response = await api.register({ name, email, password });
      const { token, name: userName, email: userEmail } = response.data;

      // Store auth data
      localStorage.setItem('authToken', token);
      localStorage.setItem('userData', JSON.stringify({ name: userName, email: userEmail }));
      
      // Set default auth header for future requests
      api.setAuthHeader(token);

      // Update state
      setCurrentUser({ name: userName, email: userEmail });
      return { success: true };
    } catch (err) {
      const errorMsg = err.response?.data?.error || 'Registration failed';
      setError(errorMsg);
      return { success: false, error: errorMsg };
    } finally {
      setLoading(false);
    }
  };

  // Logout function
  const logout = async () => {
    setLoading(true);
    
    try {
      // Call the logout endpoint if user is authenticated
      if (currentUser) {
        await api.logout();
      }
    } catch (err) {
      console.error('Logout error:', err);
      // Continue with local logout even if the server request fails
    } finally {
      // Clear storage
      localStorage.removeItem('authToken');
      localStorage.removeItem('userData');
      
      // Clear auth header
      api.clearAuthHeader();
      
      // Update state
      setCurrentUser(null);
      setLoading(false);
    }
  };

  // Value object that will be provided to consumers of this context
  const value = {
    currentUser,
    loading,
    error,
    login,
    register,
    logout
  };

  return (
    <AuthContext.Provider value={value}>
      {!loading && children}
      {loading && (
        <div style={{ 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center', 
          height: '100vh' 
        }}>
          <CircularProgress />
        </div>
      )}
    </AuthContext.Provider>
  );
};

export default AuthContext;
