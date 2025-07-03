import React from 'react';
import { 
  AppBar, 
  Toolbar, 
  Typography, 
  Box, 
  Button, 
  Avatar, 
  IconButton,
  Menu,
  MenuItem,
  Divider,
  useTheme,
  Tooltip
} from '@mui/material';
import {
  MedicalInformation as MedicalInformationIcon,
  Home as HomeIcon,
  Person as PersonIcon,
  Login as LoginIcon,
  Brightness4 as Brightness4Icon,
  Brightness7 as Brightness7Icon
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { ColorModeContext } from '../App';

const Header = ({ isAuthenticated, userName }) => {
  const [anchorEl, setAnchorEl] = React.useState(null);
  const open = Boolean(anchorEl);
  const { logout } = useAuth();
  const theme = useTheme();
  const colorMode = React.useContext(ColorModeContext);
  const navigate = useNavigate();
  const location = useLocation();
  
  const handleMenu = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };
  
  const handleLogout = async () => {
    handleClose();
    await logout();
    navigate('/auth');
  };

  const navigateTo = (path) => {
    handleClose();
    navigate(path);
  };

  const getInitials = (name) => {
    if (!name) return '?';
    return name
      .split(' ')
      .map(part => part[0])
      .join('')
      .toUpperCase();
  };

  return (
    <AppBar position="static">
      <Toolbar>
        <MedicalInformationIcon sx={{ mr: 2 }} />
        <Typography 
          variant="h6" 
          component="div"
          sx={{ cursor: 'pointer' }}
          onClick={() => navigate('/')}
        >
          Brain Metastasis Analysis
        </Typography>
        
        <Box sx={{ flexGrow: 1 }} />
        
        <Typography variant="subtitle2" sx={{ opacity: 0.7, mr: 2 }}>
          Powered by UNETR
        </Typography>
        
        {/* Navigation buttons */}
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          {/* Theme toggle button */}
          <Tooltip title={theme.palette.mode === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}>
            <IconButton 
              onClick={colorMode.toggleColorMode} 
              color="inherit"
              sx={{ mr: 1 }}
            >
              {theme.palette.mode === 'dark' ? <Brightness7Icon /> : <Brightness4Icon />}
            </IconButton>
          </Tooltip>
          
          <IconButton 
            color="inherit" 
            onClick={() => navigate('/')}
            sx={{ 
              mr: 1, 
              bgcolor: location.pathname === '/' ? 'rgba(255,255,255,0.2)' : 'transparent' 
            }}
          >
            <HomeIcon />
          </IconButton>
          
          {isAuthenticated ? (
            <>
              <IconButton
                onClick={handleMenu}
                color="inherit"
                size="small"
                aria-controls="menu-appbar"
                aria-haspopup="true"
              >
                <Avatar 
                  sx={{ 
                    width: 32, 
                    height: 32,
                    bgcolor: location.pathname === '/profile' ? 'primary.light' : 'primary.dark'
                  }}
                >
                  {getInitials(userName)}
                </Avatar>
              </IconButton>
              
              <Menu
                id="menu-appbar"
                anchorEl={anchorEl}
                anchorOrigin={{
                  vertical: 'bottom',
                  horizontal: 'right',
                }}
                keepMounted
                transformOrigin={{
                  vertical: 'top',
                  horizontal: 'right',
                }}
                open={open}
                onClose={handleClose}
              >
                <MenuItem onClick={() => navigateTo('/profile')}>My Profile</MenuItem>
                <MenuItem onClick={() => navigateTo('/')}>New Analysis</MenuItem>
                <Divider />
                <MenuItem onClick={handleLogout}>Logout</MenuItem>
              </Menu>
            </>
          ) : (
            <Button 
              color="inherit" 
              startIcon={<LoginIcon />} 
              onClick={() => navigate('/auth')}
            >
              Login
            </Button>
          )}
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Header;
