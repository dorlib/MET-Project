import React, { useState } from 'react';
import {
  Paper,
  Typography,
  Grid,
  TextField,
  Button,
  Box,
  IconButton,
  Collapse,
  InputAdornment
} from '@mui/material';
import { 
  FilterList, 
  Clear, 
  ExpandMore, 
  ExpandLess,
  Search
} from '@mui/icons-material';

/**
 * Advanced filter component for scan history
 * 
 * @param {Object} props - Component props
 * @param {Function} props.onFilter - Callback when filters are applied
 * @param {Object} props.initialFilters - Initial filter values
 */
const ScanFilter = ({ onFilter, initialFilters = {} }) => {
  const [expanded, setExpanded] = useState(false);
  const [filters, setFilters] = useState({
    min_metastasis: initialFilters.min_metastasis || '',
    max_metastasis: initialFilters.max_metastasis || '',
    min_volume: initialFilters.min_volume || '',
    max_volume: initialFilters.max_volume || '',
    start_date: initialFilters.start_date || '',
    end_date: initialFilters.end_date || '',
    ...initialFilters
  });

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
      end_date: ''
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
