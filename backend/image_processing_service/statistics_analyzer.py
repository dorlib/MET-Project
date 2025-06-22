#!/usr/bin/env python3
# statistics_analyzer.py - Analyze segmentation masks to extract metastasis statistics

import numpy as np
import scipy.ndimage as ndi
from skimage import measure
import logging

class MetastasisStatisticsAnalyzer:
    """
    Analyzes segmentation masks to extract metrics about brain metastases
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_segmentation(self, segmentation, voxel_spacing=(1.0, 1.0, 1.0)):
        """
        Analyze a segmentation mask to identify individual metastases and calculate statistics.
        
        Args:
            segmentation (numpy.ndarray): Segmentation mask where positive values indicate metastases
            voxel_spacing (tuple): Physical spacing of voxels in mm (x, y, z)
            
        Returns:
            dict: Statistics including:
                - metastasis_count: Number of distinct metastases
                - total_volume: Total volume of all metastases in mm³
                - metastasis_volumes: List of volumes for each metastasis in mm³
                - metastasis_centroids: List of centroids (x,y,z) for each metastasis
                - size_distribution: Distribution of sizes (small, medium, large)
        """
        try:
            # Ensure binary mask (all values > 0 considered metastasis)
            binary_mask = segmentation > 0
            
            # Calculate volume per voxel in mm³
            voxel_volume = voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]
            
            # Label connected components to identify individual metastases
            labeled_mask, num_metastases = ndi.label(binary_mask)
            
            # Get properties for each metastasis
            regions = measure.regionprops(labeled_mask)
            
            # Calculate volumes and centroids
            metastasis_volumes = []
            metastasis_centroids = []
            
            for region in regions:
                # Volume = number of voxels × volume per voxel
                volume = region.area * voxel_volume
                metastasis_volumes.append(float(volume))
                
                # Convert centroid to physical coordinates
                centroid = [
                    region.centroid[0] * voxel_spacing[0],
                    region.centroid[1] * voxel_spacing[1], 
                    region.centroid[2] * voxel_spacing[2]
                ]
                metastasis_centroids.append(centroid)
            
            # Sort metastases by volume (largest first)
            volume_indices = np.argsort(metastasis_volumes)[::-1]
            metastasis_volumes = [metastasis_volumes[i] for i in volume_indices]
            metastasis_centroids = [metastasis_centroids[i] for i in volume_indices]
            
            # Calculate size distribution
            # Define thresholds (customizable based on clinical relevance)
            SMALL_THRESHOLD = 100.0   # mm³
            LARGE_THRESHOLD = 1000.0  # mm³
            
            size_distribution = {
                'small': sum(1 for vol in metastasis_volumes if vol < SMALL_THRESHOLD),
                'medium': sum(1 for vol in metastasis_volumes if SMALL_THRESHOLD <= vol < LARGE_THRESHOLD),
                'large': sum(1 for vol in metastasis_volumes if vol >= LARGE_THRESHOLD)
            }
            
            # Compile results
            total_volume = sum(metastasis_volumes)
            
            results = {
                'metastasis_count': num_metastases,
                'total_volume': total_volume,
                'metastasis_volumes': metastasis_volumes,
                'metastasis_centroids': metastasis_centroids,
                'size_distribution': size_distribution
            }
            
            self.logger.info(f"Analyzed segmentation: found {num_metastases} metastases with total volume {total_volume:.2f} mm³")
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing segmentation: {str(e)}")
            raise
    
    def get_metastasis_statistics(self, segmentation, original_image=None, voxel_spacing=(1.0, 1.0, 1.0)):
        """
        Get comprehensive statistics for metastases in a segmentation mask.
        
        Args:
            segmentation (numpy.ndarray): Segmentation mask
            original_image (numpy.ndarray, optional): Original MRI image for intensity analysis
            voxel_spacing (tuple): Physical spacing of voxels in mm
            
        Returns:
            dict: Comprehensive statistics
        """
        # Get basic statistics
        basic_stats = self.analyze_segmentation(segmentation, voxel_spacing)
        
        # Additional statistics if original image is provided
        if original_image is not None and original_image.shape == segmentation.shape:
            # Calculate additional metrics using both segmentation and original image
            # For example: mean intensity, heterogeneity, etc.
            labeled_mask, _ = ndi.label(segmentation > 0)
            mean_intensities = []
            
            for i in range(1, basic_stats['metastasis_count'] + 1):
                metastasis_mask = labeled_mask == i
                if np.any(metastasis_mask):
                    mean_intensity = np.mean(original_image[metastasis_mask])
                    mean_intensities.append(float(mean_intensity))
            
            basic_stats['mean_intensities'] = mean_intensities
        
        return basic_stats