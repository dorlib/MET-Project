import React, { useState, useEffect } from 'react';
import {
  Container,
  Paper,
  Typography,
  Box,
  Avatar,
  Divider,
  CircularProgress,
  Alert,
  Button,
  Card,
  CardContent,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Switch
} from '@mui/material';
import { 
  PersonOutline, 
  HistoryOutlined, 
  ErrorOutline, 
  CheckCircleOutline,
  SecurityOutlined, 
  VerifiedUser
} from '@mui/icons-material';
import { useAuth } from '../contexts/AuthContext';
import api from '../services/api';
import { format } from 'date-fns';
import TablePagination from '@mui/material/TablePagination';
import TwoFactorSetup from './TwoFactorSetup';
import ScanFilter from './ScanFilter';

const UserProfile = ({ onViewScan }) => {
  const { currentUser, logout } = useAuth();
  const [scans, setScans] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [show2FASetup, setShow2FASetup] = useState(false);
  const [has2FAEnabled, setHas2FAEnabled] = useState(false);
  const [filters, setFilters] = useState({});
  
  // Pagination state
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(5);

  // Fetch user's scan history and 2FA status
  useEffect(() => {
    const fetchUserScans = async () => {
      try {
        setLoading(true);
        
        // Add pagination params to filters
        const params = {
          ...filters,
          page: page + 1,  // +1 because our API uses 1-based indexing
          per_page: rowsPerPage
        };
        
        // Use the filter endpoint if filters are applied, otherwise use regular endpoint
        let response;
        if (Object.keys(filters).length > 0) {
          response = await api.filterScans(params);
        } else {
          response = await api.getUserScans(page + 1, rowsPerPage);
        }
        
        setScans(response.data.scans || []);
        
        // If pagination data is available in the response
        const pagination = response.data.pagination;
        if (pagination) {
          // Adjust page if out of bounds (e.g., if items were deleted)
          if (pagination.total_pages > 0 && page >= pagination.total_pages) {
            setPage(pagination.total_pages - 1);
          }
        }
      } catch (err) {
        console.error('Error fetching user scans:', err);
        setError('Failed to load your scan history. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    const fetchUserSettings = async () => {
      try {
        const response = await api.getUserSettings();
        setHas2FAEnabled(response.data.two_fa_enabled);
      } catch (err) {
        console.error('Error fetching user settings:', err);
      }
    };

    fetchUserScans();
    fetchUserSettings();
  }, [page, rowsPerPage, filters]);

  // Format date for display
  const formatDate = (dateString) => {
    try {
      return format(new Date(dateString), 'MMM dd, yyyy - HH:mm');
    } catch (error) {
      return 'Unknown date';
    }
  };

  // Get status chip for scan
  const getStatusChip = (status) => {
    switch (status) {
      case 'completed':
        return <Chip icon={<CheckCircleOutline />} label="Completed" color="success" size="small" />;
      case 'processing':
        return <Chip icon={<CircularProgress size={16} />} label="Processing" color="primary" size="small" />;
      case 'failed':
        return <Chip icon={<ErrorOutline />} label="Failed" color="error" size="small" />;
      default:
        return <Chip label={status || 'Unknown'} size="small" />;
    }
  };
  
  const handle2FASetupComplete = (success) => {
    setShow2FASetup(false);
    if (success) {
      setHas2FAEnabled(true);
    }
  };

  // Toggle 2FA setting
  const handleTwoFAToggle = async (event) => {
    const enabled = event.target.checked;
    setHas2FAEnabled(enabled);

    if (enabled) {
      setShow2FASetup(true);
    } else {
      // Disable 2FA
      try {
        await api.updateUserSettings({ two_fa_enabled: false });
        setShow2FASetup(false);
      } catch (err) {
        console.error('Error disabling 2FA:', err);
        setError('Failed to update 2FA setting. Please try again later.');
      }
    }
  };

  // Handle filter change
  const handleFilterChange = (newFilters) => {
    setFilters(newFilters);
    setPage(0); // Reset to first page on filter change
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Grid container spacing={4}>
        {/* User info card */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent sx={{ textAlign: 'center', py: 3 }}>
              <Avatar 
                sx={{ 
                  width: 80, 
                  height: 80, 
                  mx: 'auto', 
                  mb: 2,
                  bgcolor: 'primary.main' 
                }}
              >
                <PersonOutline fontSize="large" />
              </Avatar>
              
              <Typography variant="h5" gutterBottom>
                {currentUser?.name}
              </Typography>
              
              <Typography variant="body2" color="textSecondary">
                {currentUser?.email}
              </Typography>
              
              <Button 
                variant="outlined" 
                color="primary"
                onClick={logout}
                sx={{ mt: 3 }}
              >
                Logout
              </Button>
            </CardContent>
            
            <Divider />
            
            {/* Security Settings */}
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2, display: 'flex', alignItems: 'center' }}>
                <SecurityOutlined sx={{ mr: 1 }} />
                Security Settings
              </Typography>
              
              <List disablePadding>
                <ListItem sx={{ px: 0 }}>
                  <ListItemIcon>
                    <VerifiedUser color={has2FAEnabled ? "success" : "disabled"} />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Two-Factor Authentication" 
                    secondary={has2FAEnabled ? "Enabled" : "Disabled"} 
                  />
                  {has2FAEnabled ? (
                    <Chip 
                      label="Enabled" 
                      color="success" 
                      size="small"
                    />
                  ) : (
                    <Button 
                      variant="outlined" 
                      size="small" 
                      onClick={() => setShow2FASetup(true)}
                    >
                      Enable
                    </Button>
                  )}
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>
        
        {/* Scan history */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <HistoryOutlined color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6">Scan History</Typography>
              </Box>
              
              <Divider sx={{ mb: 3 }} />
              
              {/* Scan filters */}
              <ScanFilter 
                initialFilters={filters}
                onFilter={handleFilterChange}
              />
              
              {loading ? (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <CircularProgress />
                </Box>
              ) : error ? (
                <Alert severity="error">{error}</Alert>
              ) : scans.length === 0 ? (
                <Alert severity="info">
                  No scan history found. Upload a scan to get started!
                </Alert>
              ) : (
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Date</TableCell>
                        <TableCell>File</TableCell>
                        <TableCell>Status</TableCell>
                        <TableCell align="right">Results</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {scans.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage).map((scan) => (
                        <TableRow key={scan.job_id}>
                          <TableCell>{formatDate(scan.created_at)}</TableCell>
                          <TableCell>{scan.file_name}</TableCell>
                          <TableCell>{getStatusChip(scan.status)}</TableCell>
                          <TableCell align="right">
                            {scan.status === 'completed' ? (
                              <Box>
                                <Typography variant="body2">
                                  {scan.metastasis_count || 0} metastases
                                </Typography>
                                <Button 
                                  size="small" 
                                  onClick={() => onViewScan(scan.job_id)}
                                >
                                  View Details
                                </Button>
                              </Box>
                            ) : (
                              '-'
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                  {scans.length > 0 && (
                    <TablePagination
                      rowsPerPageOptions={[5, 10, 25]}
                      component="div"
                      count={scans.length}
                      rowsPerPage={rowsPerPage}
                      page={page}
                      onPageChange={(event, newPage) => setPage(newPage)}
                      onRowsPerPageChange={(event) => {
                        setRowsPerPage(parseInt(event.target.value, 10));
                        setPage(0);
                      }}
                    />
                  )}
                </TableContainer>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      
      {/* 2FA setup dialog */}
      <TwoFactorSetup 
        open={show2FASetup}
        onClose={handle2FASetupComplete}
      />
    </Container>
  );
};

export default UserProfile;
