import React from 'react';
import { 
  Box, 
  Typography, 
  Grid, 
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  FormControlLabel,
  Switch,
  Stack
} from '@mui/material';

/**
 * Visualization Controls component for MRI visualization settings
 * 
 * @param {Object} props Component properties
 * @param {string} props.vizType Type of visualization ('slice', 'multi-slice', etc)
 * @param {Function} props.setVizType Function to update visualization type
 * @param {string} props.vizQuality Quality setting ('high', 'standard')
 * @param {Function} props.setVizQuality Function to update quality setting
 * @param {number|null} props.sliceIndex Current slice index (null = middle)
 * @param {Function} props.setSliceIndex Function to update slice index
 * @param {number} props.numSlices Number of slices for multi-slice view
 * @param {Function} props.setNumSlices Function to update number of slices
 * @param {number} props.upscaleFactor Upscale factor for high-resolution mode
 * @param {Function} props.setUpscaleFactor Function to update upscale factor
 * @param {boolean} props.enhanceContrast Whether contrast enhancement is enabled
 * @param {Function} props.setEnhanceContrast Function to toggle contrast enhancement
 * @param {boolean} props.enhanceEdges Whether edge enhancement is enabled
 * @param {Function} props.setEnhanceEdges Function to toggle edge enhancement
 */
const VisualizationControls = ({
  vizType,
  setVizType,
  vizQuality,
  setVizQuality,
  sliceIndex,
  setSliceIndex,
  numSlices,
  setNumSlices,
  upscaleFactor,
  setUpscaleFactor,
  enhanceContrast,
  setEnhanceContrast,
  enhanceEdges,
  setEnhanceEdges,
  maxSliceIndex
}) => {
  // Use local state to track slider position during dragging
  const [localSliceIndex, setLocalSliceIndex] = React.useState(
    sliceIndex === null ? Math.floor((maxSliceIndex || 100) / 2) : sliceIndex
  );
  
  // Update local slice index when the prop changes (from outside)
  React.useEffect(() => {
    if (sliceIndex !== null) {
      setLocalSliceIndex(sliceIndex);
    }
  }, [sliceIndex]);
  
  // Debounce function to limit API calls
  const debouncedSetSliceIndex = React.useCallback(
    React.useMemo(
      () => {
        // Create a debounced version of setSliceIndex that only triggers after 150ms of no changes
        let timeoutId;
        return (newValue) => {
          clearTimeout(timeoutId);
          timeoutId = setTimeout(() => {
            // Only update if the value has changed
            if (newValue !== sliceIndex) {
              console.log(`Updating actual slice index to: ${newValue}`);
              setSliceIndex(newValue);
            }
          }, 150); // 150ms debounce
        };
      },
      [setSliceIndex, sliceIndex]
    ),
    [setSliceIndex, sliceIndex]
  );
  return (
    <Paper sx={{ p: 2, mb: 2 }}>
      <Typography variant="subtitle1" gutterBottom>
        Visualization Controls
      </Typography>
      <Grid container spacing={2} alignItems="center">
        <Grid item xs={12} md={4}>
          <FormControl fullWidth size="small">
            <InputLabel>Type</InputLabel>
            <Select
              value={vizType}
              label="Type"
              onChange={(e) => setVizType(e.target.value)}
            >
              <MenuItem value="slice">Single Slice</MenuItem>
              <MenuItem value="multi-slice">Multiple Slices</MenuItem>
              <MenuItem value="projection">3D Projection</MenuItem>
              <MenuItem value="lesions">Lesions Focus</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <FormControl fullWidth size="small">
            <InputLabel>Quality</InputLabel>
            <Select
              value={vizQuality}
              label="Quality"
              onChange={(e) => setVizQuality(e.target.value)}
            >
              <MenuItem value="high">High Resolution</MenuItem>
              <MenuItem value="standard">Standard</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        
        <Grid item xs={12} md={4}>
          {vizQuality === 'high' && (
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <Typography variant="body2" sx={{ mr: 1, whiteSpace: 'nowrap' }}>
                Upscale:
              </Typography>
              <Slider
                size="small"
                min={1.0}
                max={2.0}
                step={0.1}
                value={upscaleFactor}
                onChange={(e, value) => setUpscaleFactor(value)}
                sx={{ width: '80px', mx: 1 }}
              />
              <Typography variant="body2" sx={{ ml: 1, minWidth: '30px' }}>
                {upscaleFactor}x
              </Typography>
            </Box>
          )}
        </Grid>

        {vizType === 'slice' && (
          <Grid item xs={12}>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <Typography variant="body2" sx={{ mr: 1, whiteSpace: 'nowrap' }}>
                Slice:
              </Typography>
              <Slider
                size="small"
                min={0}
                max={maxSliceIndex || 100}
                value={localSliceIndex}
                onChange={(e, value) => {
                  // Update local state immediately for responsive UI
                  setLocalSliceIndex(value);
                  // Debounce the actual state update to limit API calls
                  debouncedSetSliceIndex(value);
                }}
                onChangeCommitted={(e, value) => {
                  // Ensure the final position is applied
                  console.log(`Slider final position: ${value}`);
                  // Force update the real state when dragging ends
                  setSliceIndex(value);
                }}
                sx={{ width: '150px', mx: 1 }}
              />
              <Typography variant="body2" sx={{ ml: 1, minWidth: '40px' }}>
                {localSliceIndex}/{maxSliceIndex || 100}
              </Typography>
              {/* Debug info */}
              <Typography variant="caption" color="text.secondary" sx={{ ml: 1, display: 'none' }}>
                (max: {maxSliceIndex})
              </Typography>
            </Box>
          </Grid>
        )}
        
        {vizType === 'multi-slice' && (
          <Grid item xs={12}>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <Typography variant="body2" sx={{ mr: 1, whiteSpace: 'nowrap' }}>
                Number of Slices:
              </Typography>
              <Slider
                size="small"
                min={3}
                max={9}
                step={2}
                value={numSlices}
                onChange={(e, value) => setNumSlices(value)}
                marks={[
                  { value: 3, label: '3' },
                  { value: 5, label: '5' },
                  { value: 7, label: '7' },
                  { value: 9, label: '9' },
                ]}
                sx={{ width: '150px', mx: 1 }}
              />
              <Typography variant="body2" sx={{ ml: 1, minWidth: '30px' }}>
                {numSlices}
              </Typography>
            </Box>
          </Grid>
        )}
        
        {vizQuality === 'high' && (
          <Grid item xs={12}>
            <Stack direction="row" spacing={3}>
              <FormControlLabel
                control={
                  <Switch 
                    checked={enhanceContrast} 
                    onChange={(e) => setEnhanceContrast(e.target.checked)}
                    size="small"
                  />
                }
                label="Enhance Contrast"
              />
              <FormControlLabel
                control={
                  <Switch 
                    checked={enhanceEdges} 
                    onChange={(e) => setEnhanceEdges(e.target.checked)}
                    size="small"
                  />
                }
                label="Enhance Edges"
              />
            </Stack>
          </Grid>
        )}
      </Grid>
    </Paper>
  );
};

export default VisualizationControls;
