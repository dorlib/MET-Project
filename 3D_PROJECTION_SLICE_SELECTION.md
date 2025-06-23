# 3D Projection Slice Selection Feature

This document provides information about the 3D Projection Slice Selection feature that allows users to choose which slice to display in the 3D Projection visualization mode.

## Feature Overview

When using the 3D Projection visualization mode, users can now select a specific slice to be used for generating the axial, sagittal, and coronal projections, instead of always using the maximum intensity projection.

## How to Use

1. In the visualization controls, select "3D Projection" as the visualization type.
2. Use the slice slider that appears to select which slice you want to use for the projection.
3. The 3D projection visualization will update to show the projections based on your selected slice.

## Technical Implementation

The feature has been implemented with the following components:

1. **Frontend Implementation**:
   - The `VisualizationControls.js` component shows the slice slider when the visualization type is either 'slice' or 'projection'.
   - The `ResultViewer.js` component passes the selected slice index to the API call.
   - The `api.js` service includes the slice index in the visualization URL parameters.

2. **Backend Implementation**:
   - The `generate_3d_projection` function in `image_processor.py` accepts a `slice_idx` parameter.
   - When a slice index is provided, the function creates projections using that specific slice instead of using the maximum intensity projection.
   - The API gateway passes all query parameters to the image processing service.

## Testing

To test the feature, use the included test script:

```bash
./test_3d_projection_slice_selection.sh
```

This script generates 3D projections with different slice indices and saves them in the `test_3d_projection_slices` folder for visual comparison.

## Verification

You can verify that the feature is correctly implemented by running:

```bash
./verify_3d_projection_slice.sh
```

This script checks all the necessary code components to ensure the feature is properly implemented.
