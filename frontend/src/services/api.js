import axios from 'axios';
import { throttle } from '../utils/throttle';

// Creating an axios instance with default configuration
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json'
  }
});

// API service with methods for all backend communication
const apiService = {
  // Auth token management
  setAuthHeader: (token) => {
    api.defaults.headers.common['Authorization'] = `Bearer ${token}`;
  },
  
  clearAuthHeader: () => {
    delete api.defaults.headers.common['Authorization'];
  },
  
  // Authentication endpoints
  login: (credentials) => {
    return api.post('/auth/login', credentials);
  },
  
  register: (userData) => {
    return api.post('/auth/register', userData);
  },
  
  refreshToken: () => {
    return api.post('/auth/refresh');
  },
  
  validateToken: () => {
    return api.get('/auth/validate');
  },
  
  // Scan upload and analysis with retry functionality
  uploadScan: async (formData, onProgress) => {
    // Configuration
    const maxRetries = 3;
    const retryDelay = 2000; // 2 seconds
    let retries = 0;
    
    const makeRequest = async () => {
      try {
        return await api.post('/upload', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          },
          timeout: 60000, // 60 seconds timeout
          onUploadProgress: onProgress ? (progressEvent) => {
            const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            onProgress(percentCompleted);
          } : undefined
        });
      } catch (error) {
        // Only retry on network errors or 5xx server errors
        if (retries < maxRetries && 
            (error.code === 'ECONNABORTED' || 
             !error.response || 
             error.response.status >= 500)) {
          
          console.log(`Upload attempt ${retries + 1} failed, retrying in ${retryDelay}ms...`);
          retries++;
          
          // Wait before retrying
          await new Promise(resolve => setTimeout(resolve, retryDelay));
          return makeRequest(); // Recursive retry
        }
        
        // If we've run out of retries or it's a 4xx error, throw the error
        throw error;
      }
    };
    
    return makeRequest();
  },
  
  getJobStatus: (jobId) => {
    return api.get(`/status/${jobId}`);
  },
  
  // Cache results to prevent redundant API calls
  _resultsCache: {},

  // Raw getResults implementation
  _getResultsImpl: (jobId) => {
    // If we have recent results for this job in cache, return a Promise with that data
    const cachedData = apiService._resultsCache[jobId];
    if (cachedData && (Date.now() - cachedData.timestamp) < 10000) { // 10 second cache
      console.log("Using cached results for job:", jobId);
      return Promise.resolve(cachedData.response);
    }
    
    console.log("Calling getResults API for job:", jobId);
    return api.get(`/results/${jobId}`, { timeout: 10000 }) // 10 second timeout
      .then(response => {
        // Check if response indicates job doesn't exist
        if (response.data && (
            response.data.status === 'not_found' || 
            response.data.error === 'Job not found' || 
            response.data.status === 'failed' || 
            response.data.status === 'error'
          )) {
          console.log(`Job ${jobId} ${response.data.status || 'not found'}`);
          // Cache this result with a longer expiry to prevent repeated failed calls
          apiService._resultsCache[jobId] = {
            response: {
              ...response,
              data: { 
                ...response.data, 
                status: response.data.status || 'not_found' 
              }
            },
            timestamp: Date.now()
          };
          return apiService._resultsCache[jobId].response;
        }
        
        // Special handling for processing status - short cache to ensure we check more frequently
        if (response.data && response.data.status === 'processing') {
          apiService._resultsCache[jobId] = {
            response: response,
            timestamp: Date.now()
          };
          return response;
        }
        
        // For completed status, cache longer
        if (response.data && response.data.status === 'completed') {
          apiService._resultsCache[jobId] = {
            response: response,
            timestamp: Date.now() + 3600000 // Cache completed results for 1 hour
          };
          return response;
        }
        
        // Regular response caching for other statuses
        apiService._resultsCache[jobId] = {
          response: response,
          timestamp: Date.now()
        };
        return response;
      })
      .catch(error => {
        // If we get a 404 or specific error codes, mark the job as not found
        if (error.response && (error.response.status === 404 || error.response.status === 400)) {
          console.log(`Handling error for job ${jobId} as not_found (${error.response.status})`);
          const notFoundResponse = {
            data: {
              status: 'not_found',
              error: error.response.data?.message || 'Job not found'
            },
            status: 200 // Fake 200 status so it doesn't trigger error handling
          };
          
          // Cache this result longer to prevent repeated failed calls
          apiService._resultsCache[jobId] = {
            response: notFoundResponse,
            timestamp: Date.now() + 60000 // Cache for 1 minute
          };
          
          return notFoundResponse;
        }
        
        // For network errors or timeouts, create a processing response to keep polling
        if (error.code === 'ECONNABORTED' || !error.response) {
          console.log(`Network error for job ${jobId}, treating as still processing`);
          const processingResponse = {
            data: {
              status: 'processing',
              message: 'Network error, still processing'
            },
            status: 200
          };
          
          // Short cache for network errors
          apiService._resultsCache[jobId] = {
            response: processingResponse,
            timestamp: Date.now() + 5000 // Cache for 5 seconds
          };
          
          return processingResponse;
        }
        
        // For other errors, let the caller handle them
        throw error;
      });
  },
  
  // Throttled version of getResults
  getResults: throttle((jobId) => {
    return apiService._getResultsImpl(jobId);
  }, 2000), // Throttle to once every 2 seconds max
  
  // User profile and history
  getUserProfile: () => {
    return api.get('/user/profile');
  },
  
  updateUserProfile: (profileData) => {
    return api.put('/user/profile', profileData);
  },
  
  getScanHistory: () => {
    return api.get('/user/scans');
  },
  
  deleteScan: (scanId) => {
    return api.delete(`/user/scans/${scanId}`);
  },
  
  // Export results handling
  exportResultsCSV: (jobId) => {
    return api.get(`/export/csv/${jobId}`, {
      responseType: 'blob'
    });
  },
  
  exportResultsPDF: (jobId) => {
    return api.get(`/export/pdf/${jobId}`, {
      responseType: 'blob'
    });
  },
  
  filterScans: (filters = {}) => {
    return api.get('/user/scans/filter', { params: filters });
  },
  
  // Visualization endpoints with customization options
  getVisualizationUrl: (jobId, options = {}, segmentationPath = null) => {
    const {
      type = 'slice',              // 'slice', 'multi-slice', 'projection', 'lesions'
      quality = 'high',            // 'high', 'standard'
      slice = null,                // slice index (null = middle slice)
      numSlices = 5,               // for multi-slice view
      upscale = 1.2,               // upscaling factor for high-res visualizations
      enhanceContrast = true,      // contrast enhancement
      enhanceEdges = true,         // edge enhancement
      brightness = null,           // brightness adjustment
      contrast = null,             // contrast adjustment
      show_original = false,       // whether to show original scan when no segmentation
      colormap = null              // colormap for visualization
    } = options;
    
    // Check if the job is in cache and is in a terminal state
    const cachedData = apiService._resultsCache[jobId];
    if (cachedData?.response?.data?.status === 'not_found') {
      console.log(`Visualization requested for not_found job ${jobId}, returning empty URL`);
      return '#'; // Return a non-functional URL for jobs not found
    }
    
    const params = { 
      quality, 
      upscale, 
      enhance_contrast: enhanceContrast, 
      enhance_edges: enhanceEdges,
      type // Use 'type' as parameter name to match backend expectation
    };
    
    // Add conditional parameters
    if (slice !== null) {
      params.slice_idx = slice;
    } else {
      // Default to middle slice (index 50) instead of using 'middle' string which causes 400 error
      params.slice_idx = 50; // Will be overridden by actual middle slice when dimensions are fetched
    }
    if (type === 'multi-slice') params.num_slices = numSlices;
    if (brightness) params.brightness = brightness;
    if (contrast) params.contrast = contrast;
    if (show_original) params.show_original = show_original;
    if (colormap) params.colormap = colormap;
    
    // Build the URL with query parameters
    const queryString = Object.keys(params)
      .map(key => `${key}=${params[key]}`)
      .join('&');
    
    // Use the path directly from the backend if provided
    if (segmentationPath) {
      return `${segmentationPath}?${queryString}`;
    } else {
      // Fallback to constructed path based on quality
      const endpoint = quality === 'high' ? 'advanced-visualization' : 'visualization';
      return `/${endpoint}/${jobId}?${queryString}`;
    }
  },

  // High resolution data visualization (for direct API calls)
  getVisualization: (jobId, options = {}) => {
    const {
      type = 'slice',
      quality = 'high',
      slice = null,
      numSlices = 5
    } = options;
    
    const params = { 
      type,
      quality
    };
    
    if (slice !== null) params.slice_idx = slice;
    if (type === 'multi-slice') params.num_slices = numSlices;
    
    const endpoint = quality === 'high' ? 'advanced-visualization' : 'visualization';
    return api.get(`/${endpoint}/${jobId}`, { params, responseType: 'blob' });
  },
  
  // 2FA
  setup2FA: () => {
    return api.post('/auth/2fa/setup');
  },
  
  verify2FA: (token) => {
    return api.post('/auth/2fa/verify', { token });
  },
  
  enable2FA: () => {
    return api.post('/auth/2fa/enable');
  },
  
  disable2FA: () => {
    return api.post('/auth/2fa/disable');
  },
  
  // Volume dimensions and information
  _dimensionsCache: {},

  getVolumeDimensions: (jobId) => {
    // If we have recent dimensions for this job in cache, return a Promise with that data
    const cachedData = apiService._dimensionsCache[jobId];
    if (cachedData && (Date.now() - cachedData.timestamp) < 300000) { // 5 minute cache
      console.log("Using cached volume dimensions for job:", jobId);
      return Promise.resolve(cachedData.response);
    }
    
    console.log("Calling getVolumeDimensions API for job:", jobId);
    return api.get(`/api/volume-dimensions/${jobId}`).then(response => {
      // Cache the result
      apiService._dimensionsCache[jobId] = {
        response: response,
        timestamp: Date.now()
      };
      return response;
    });
  }
};

export default apiService;
