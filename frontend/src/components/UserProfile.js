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
  Switch,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  IconButton,
  Tooltip,
  TextField
} from '@mui/material';
import { 
  PersonOutline, 
  HistoryOutlined, 
  ErrorOutline, 
  CheckCircleOutline,
  SecurityOutlined, 
  VerifiedUser,
  Delete,
  Warning,
  Edit,
  Save,
  Cancel,
  Person,
  Search
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import api from '../services/api';
import { format } from 'date-fns';
import TablePagination from '@mui/material/TablePagination';
import TwoFactorSetup from './TwoFactorSetup';
import ScanFilter from './ScanFilter';

const UserProfile = () => {
  const { currentUser, logout } = useAuth();
  const navigate = useNavigate();
  const [scans, setScans] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [show2FASetup, setShow2FASetup] = useState(false);
  const [has2FAEnabled, setHas2FAEnabled] = useState(false);
  const [filters, setFilters] = useState({});
  
  // Pagination state
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(5);

  // Delete confirmation state
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [scanToDelete, setScanToDelete] = useState(null);
  const [deleting, setDeleting] = useState(false);
  
  // Patient name editing state
  const [editingPatientName, setEditingPatientName] = useState(null);
  const [patientNameValue, setPatientNameValue] = useState('');
  const [updatingPatientName, setUpdatingPatientName] = useState(false);
  
  // Patient name search state
  const [patientNameSearch, setPatientNameSearch] = useState('');
  const [filteredScans, setFilteredScans] = useState([]);

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
          console.log('UserProfile: Filtering scans with params:', params);
          response = await api.filterScans(params);
          console.log('UserProfile: Filter response:', response);
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

  // Filter scans by patient name search
  useEffect(() => {
    if (!patientNameSearch.trim()) {
      setFilteredScans(scans);
    } else {
      const searchTerm = patientNameSearch.toLowerCase();
      const filtered = scans.filter(scan => {
        if (!scan.patient_name) return false;
        return scan.patient_name.toLowerCase().includes(searchTerm);
      });
      setFilteredScans(filtered);
      // Reset to first page when search results change
      setPage(0);
    }
  }, [scans, patientNameSearch]);

  // Handle search input change
  const handleSearchChange = (event) => {
    setPatientNameSearch(event.target.value);
  };

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

  // Handle delete scan
  const handleDeleteScan = (scan) => {
    setScanToDelete(scan);
    setDeleteDialogOpen(true);
  };

  // Confirm delete scan
  const confirmDeleteScan = async () => {
    if (!scanToDelete) return;

    setDeleting(true);
    try {
      await api.deleteScan(scanToDelete.job_id);
      
      // Remove the scan from the local state
      setScans(prevScans => prevScans.filter(scan => scan.job_id !== scanToDelete.job_id));
      
      // Show success message
      setError(null);
      
      // Reset pagination if necessary
      const newTotalScans = scans.length - 1;
      const newMaxPage = Math.ceil(newTotalScans / rowsPerPage) - 1;
      if (page > newMaxPage && newMaxPage >= 0) {
        setPage(newMaxPage);
      }
      
    } catch (err) {
      console.error('Error deleting scan:', err);
      setError('Failed to delete scan. Please try again later.');
    } finally {
      setDeleting(false);
      setDeleteDialogOpen(false);
      setScanToDelete(null);
    }
  };

  // Cancel delete
  const cancelDelete = () => {
    setDeleteDialogOpen(false);
    setScanToDelete(null);
  };

  // Handle edit patient name
  const handleEditPatientName = (scan) => {
    setEditingPatientName(scan.job_id);
    setPatientNameValue(scan.patient_name || '');
  };

  // Handle save patient name
  const handleSavePatientName = async (jobId) => {
    setUpdatingPatientName(true);
    try {
      await api.updateScanPatientName(jobId, patientNameValue);
      
      // Update the scan in local state
      setScans(prevScans => 
        prevScans.map(scan => 
          scan.job_id === jobId 
            ? { ...scan, patient_name: patientNameValue }
            : scan
        )
      );
      
      setEditingPatientName(null);
      setPatientNameValue('');
      setError(null);
    } catch (err) {
      console.error('Error updating patient name:', err);
      setError('Failed to update patient name. Please try again.');
    } finally {
      setUpdatingPatientName(false);
    }
  };

  // Handle cancel patient name edit
  const handleCancelPatientName = () => {
    setEditingPatientName(null);
    setPatientNameValue('');
  };

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      <Grid container spacing={4}>
        {/* User info card */}
        <Grid item xs={12} md={3}>
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
        <Grid item xs={12} md={9}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <HistoryOutlined color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6">Scan History</Typography>
              </Box>
              
              <Divider sx={{ mb: 3 }} />
              
              {/* Patient name search */}
              <Box sx={{ mb: 3 }}>
                <TextField
                  fullWidth
                  variant="outlined"
                  placeholder="Search by patient name..."
                  value={patientNameSearch}
                  onChange={handleSearchChange}
                  InputProps={{
                    startAdornment: <Search color="disabled" sx={{ mr: 1 }} />
                  }}
                  sx={{ maxWidth: 400 }}
                />
                {patientNameSearch && (
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                    {filteredScans.length} scan(s) found matching "{patientNameSearch}"
                  </Typography>
                )}
              </Box>
              
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
              ) : filteredScans.length === 0 && patientNameSearch ? (
                <Alert severity="info">
                  No scans found matching "{patientNameSearch}". Try a different search term.
                </Alert>
              ) : (
                <TableContainer sx={{ width: '100%' }}>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell sx={{ width: '140px' }}>Date</TableCell>
                        <TableCell sx={{ minWidth: '200px' }}>Patient Name</TableCell>
                        <TableCell sx={{ maxWidth: '200px' }}>File</TableCell>
                        <TableCell sx={{ width: '120px' }}>Model</TableCell>
                        <TableCell sx={{ width: '100px' }}>Status</TableCell>
                        <TableCell align="right" sx={{ width: '120px' }}>Results</TableCell>
                        <TableCell align="center" sx={{ width: '100px' }}>Actions</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {filteredScans.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage).map((scan) => (
                        <TableRow key={scan.job_id}>
                          <TableCell>{formatDate(scan.created_at)}</TableCell>
                          <TableCell>
                            {editingPatientName === scan.job_id ? (
                              <Box display="flex" alignItems="center" gap={1}>
                                <TextField
                                  size="small"
                                  value={patientNameValue}
                                  onChange={(e) => setPatientNameValue(e.target.value)}
                                  placeholder="Enter patient name"
                                  disabled={updatingPatientName}
                                  sx={{ minWidth: '150px', maxWidth: '180px' }}
                                />
                                <IconButton
                                  size="small"
                                  onClick={() => handleSavePatientName(scan.job_id)}
                                  disabled={updatingPatientName}
                                  color="primary"
                                >
                                  {updatingPatientName ? <CircularProgress size={16} /> : <Save />}
                                </IconButton>
                                <IconButton
                                  size="small"
                                  onClick={handleCancelPatientName}
                                  disabled={updatingPatientName}
                                >
                                  <Cancel />
                                </IconButton>
                              </Box>
                            ) : (
                              <Box display="flex" alignItems="center" gap={1}>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                  <Person fontSize="small" color="disabled" />
                                  <Typography variant="body2">
                                    {scan.patient_name || 'Not set'}
                                  </Typography>
                                </Box>
                                <Tooltip title="Edit patient name">
                                  <IconButton
                                    size="small"
                                    onClick={() => handleEditPatientName(scan)}
                                  >
                                    <Edit fontSize="small" />
                                  </IconButton>
                                </Tooltip>
                              </Box>
                            )}
                          </TableCell>
                          <TableCell sx={{ maxWidth: '200px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                            <Tooltip title={scan.file_name}>
                              <span>{scan.file_name}</span>
                            </Tooltip>
                          </TableCell>
                          <TableCell>
                            <Tooltip title={scan.model_name || 'brats_t1ce.pth'}>
                              <Chip 
                                label={(scan.model_name || 'brats_t1ce.pth').replace('.pth', '')} 
                                size="small" 
                                variant="outlined"
                                color="primary"
                                sx={{ maxWidth: '110px', fontSize: '0.75rem' }}
                              />
                            </Tooltip>
                          </TableCell>
                          <TableCell>{getStatusChip(scan.status)}</TableCell>
                          <TableCell align="right">
                            {scan.status === 'completed' ? (
                              <Box>
                                <Typography variant="body2">
                                  {scan.metastasis_count || 0} metastases
                                </Typography>
                                <Button 
                                  size="small" 
                                  onClick={() => navigate(`/scan/${scan.job_id}`)}
                                >
                                  View Details
                                </Button>
                              </Box>
                            ) : (
                              '-'
                            )}
                          </TableCell>
                          <TableCell align="center">
                            <Tooltip title="Delete scan">
                              <IconButton
                                size="small"
                                color="error"
                                onClick={() => handleDeleteScan(scan)}
                                disabled={deleting}
                              >
                                <Delete fontSize="small" />
                              </IconButton>
                            </Tooltip>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                  {scans.length > 0 && (
                    <TablePagination
                      rowsPerPageOptions={[5, 10, 25]}
                      component="div"
                      count={filteredScans.length}
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

      {/* Delete confirmation dialog */}
      <Dialog
        open={deleteDialogOpen}
        onClose={cancelDelete}
        aria-labelledby="delete-dialog-title"
        aria-describedby="delete-dialog-description"
      >
        <DialogTitle id="delete-dialog-title" sx={{ display: 'flex', alignItems: 'center' }}>
          <Warning color="warning" sx={{ mr: 1 }} />
          Confirm Delete
        </DialogTitle>
        <DialogContent>
          <DialogContentText id="delete-dialog-description">
            Are you sure you want to delete this scan?
            {scanToDelete && (
              <Box sx={{ mt: 2, p: 2, bgcolor: 'grey.100', borderRadius: 1 }}>
                <Typography variant="body2"><strong>File:</strong> {scanToDelete.file_name}</Typography>
                <Typography variant="body2"><strong>Date:</strong> {formatDate(scanToDelete.created_at)}</Typography>
                <Typography variant="body2"><strong>Status:</strong> {scanToDelete.status}</Typography>
              </Box>
            )}
            <Box sx={{ mt: 2, p: 2, bgcolor: 'error.50', borderRadius: 1, border: '1px solid', borderColor: 'error.200' }}>
              <Typography variant="body2" color="error.main">
                <strong>Warning:</strong> This action cannot be undone. All scan data and results will be permanently deleted.
              </Typography>
            </Box>
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={cancelDelete} disabled={deleting}>
            Cancel
          </Button>
          <Button 
            onClick={confirmDeleteScan} 
            color="error" 
            variant="contained"
            disabled={deleting}
            startIcon={deleting ? <CircularProgress size={16} /> : <Delete />}
          >
            {deleting ? 'Deleting...' : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default UserProfile;
