import React, { useState, useEffect } from 'react';
import {
  Paper,
  Typography,
  Grid,
  TextField,
  Button,
  Box,
  IconButton,
  Collapse,
  InputAdornment,
  Select,
  MenuItem,
  FormControl,
  InputLabel
} from '@mui/material';
import { 
  FilterList, 
  Clear, 
  ExpandMore, 
  ExpandLess,
  Search
} from '@mui/icons-material';
import apiService from '../services/api';

/**
 * Advanced filter component for scan history
 * 
 * @param {Object} props - Component props
 * @param {Function} props.onFilter - Callback when filters are applied
 * @param {Object} props.initialFilters - Initial filter values
 */
const ScanFilter = ({ onFilter, initialFilters = {} }) => {
  const [expanded, setExpanded] = useState(false);
  const [models, setModels] = useState([]);
  const [filters, setFilters] = useState({
    min_metastasis: initialFilters.min_metastasis || '',
    max_metastasis: initialFilters.max_metastasis || '',
    min_volume: initialFilters.min_volume || '',
    max_volume: initialFilters.max_volume || '',
    start_date: initialFilters.start_date || '',
    end_date: initialFilters.end_date || '',
    model_name: initialFilters.model_name || '',
    ...initialFilters
  });

  // Fetch available models on component mount
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await apiService.getModels();
        setModels(response.data.models || []);
      } catch (error) {
        console.error('Error fetching models:', error);
      }
    };
    
    fetchModels();
  }, []);

  const handleChange = (field) => (event) => {
    setFilters({
      ...filters,
      [field]: event.target.value
    });
  };

  const handleClearFilters = () => {
    setFilters({
      min_metastasis: '',
      max_metastasis: '',
      min_volume: '',
      max_volume: '',
      start_date: '',
      end_date: '',
      model_name: ''
    });
    
    // Apply the cleared filters
    onFilter({});
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    // Filter out empty values
    const appliedFilters = {};
    Object.keys(filters).forEach(key => {
      if (filters[key] !== '') {
        appliedFilters[key] = filters[key];
      }
    });
    
    console.log('ScanFilter: Applying filters:', appliedFilters);
    console.log('ScanFilter: Available models:', models);
    
    onFilter(appliedFilters);
  };

  const toggleExpanded = () => {
    setExpanded(!expanded);
  };

  return (
    <Paper sx={{ p: 2, mb: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <FilterList sx={{ mr: 1 }} />
          <Typography variant="h6">Filter Scans</Typography>
        </Box>
        <IconButton onClick={toggleExpanded} size="small">
          {expanded ? <ExpandLess /> : <ExpandMore />}
        </IconButton>
      </Box>
      
      <Collapse in={expanded}>
        <Box component="form" onSubmit={handleSubmit} sx={{ mt: 2 }}>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={3}>
              <TextField
                label="Min Metastases"
                type="number"
                fullWidth
                value={filters.min_metastasis}
                onChange={handleChange('min_metastasis')}
                size="small"
                InputProps={{
                  inputProps: { min: 0 }
                }}
              />
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <TextField
                label="Max Metastases"
                type="number"
                fullWidth
                value={filters.max_metastasis}
                onChange={handleChange('max_metastasis')}
                size="small"
                InputProps={{
                  inputProps: { min: 0 }
                }}
              />
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <TextField
                label="Min Volume"
                type="number"
                fullWidth
                value={filters.min_volume}
                onChange={handleChange('min_volume')}
                size="small"
                InputProps={{
                  endAdornment: <InputAdornment position="end">mm³</InputAdornment>,
                  inputProps: { min: 0, step: 0.1 }
                }}
              />
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <TextField
                label="Max Volume"
                type="number"
                fullWidth
                value={filters.max_volume}
                onChange={handleChange('max_volume')}
                size="small"
                InputProps={{
                  endAdornment: <InputAdornment position="end">mm³</InputAdornment>,
                  inputProps: { min: 0, step: 0.1 }
                }}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                label="Start Date"
                type="date"
                fullWidth
                value={filters.start_date}
                onChange={handleChange('start_date')}
                size="small"
                InputLabelProps={{
                  shrink: true
                }}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                label="End Date"
                type="date"
                fullWidth
                value={filters.end_date}
                onChange={handleChange('end_date')}
                size="small"
                InputLabelProps={{
                  shrink: true
                }}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth size="small">
                <InputLabel>Model</InputLabel>
                <Select
                  value={filters.model_name}
                  onChange={handleChange('model_name')}
                  label="Model"
                >
                  <MenuItem value="">All Models</MenuItem>
                  {models.map((model) => {
                    // Strip .pth extension for display and filtering
                    const displayName = model.name.endsWith('.pth') 
                      ? model.name.slice(0, -4) 
                      : model.name;
                    return (
                      <MenuItem key={model.name} value={displayName}>
                        {displayName}
                      </MenuItem>
                    );
                  })}
                </Select>
              </FormControl>
            </Grid>
          </Grid>
          
          <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
            <Button 
              variant="outlined" 
              onClick={handleClearFilters} 
              startIcon={<Clear />}
              sx={{ mr: 1 }}
            >
              Clear
            </Button>
            <Button 
              type="submit" 
              variant="contained" 
              startIcon={<Search />}
            >
              Apply Filters
            </Button>
          </Box>
        </Box>
      </Collapse>
    </Paper>
  );
};

export default ScanFilter;
