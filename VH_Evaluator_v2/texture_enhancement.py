#!/usr/bin/env python3
"""
Texture Enhancement Module for Grid Pattern Detection
Provides image preprocessing methods to enhance texture visibility before grid detection.

Supported Methods:
- Laplacian Enhancement: Edge and detail enhancement using Laplacian operator
- CLAHE Enhancement: Contrast Limited Adaptive Histogram Equalization
- Bilateral Filter Enhancement: Edge-preserving smoothing with detail enhancement
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class TextureEnhancer:
    """Texture enhancement class for preprocessing images before grid pattern detection."""
    
    def __init__(self, image_input):
        """
        Initialize the texture enhancer.
        
        Args:
            image_input: Either a file path (str) or a numpy array (BGR image)
        """
        if isinstance(image_input, str):
            self.original = cv2.imread(image_input)
            if self.original is None:
                raise ValueError(f"Failed to load image: {image_input}")
        elif isinstance(image_input, np.ndarray):
            self.original = image_input.copy()
        else:
            raise TypeError("image_input must be a file path or numpy array")
        
        self.original_rgb = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)
        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
    
    def laplacian_enhancement(self, strength: float = 2.0) -> np.ndarray:
        """
        Enhance image using Laplacian operator for edge and detail enhancement.
        
        The Laplacian operator detects edges and fine details by computing
        the second derivative of the image. Adding this back to the original
        image sharpens edges and enhances texture patterns.
        
        Args:
            strength: Enhancement strength multiplier (default: 2.0)
                     Higher values = stronger edge enhancement
                     Recommended range: 1.0 - 5.0
        
        Returns:
            Enhanced BGR image as numpy array
        """
        # Apply Laplacian operator to grayscale image
        laplacian = cv2.Laplacian(self.gray, cv2.CV_64F)
        
        # Create enhanced image by adding weighted Laplacian to original
        enhanced = self.original.astype(np.float64)
        
        # Apply enhancement to each channel
        for i in range(3):
            enhanced[:, :, i] = enhanced[:, :, i] + strength * laplacian
        
        # Clip values and convert back to uint8
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return enhanced
    
    def clahe_enhancement(self, clip_limit: float = 4.0, tile_size: int = 8) -> np.ndarray:
        """
        Enhance image using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        CLAHE divides the image into small tiles and applies histogram equalization
        to each tile independently, with contrast limiting to prevent noise amplification.
        This is particularly effective for revealing subtle texture patterns.
        
        Args:
            clip_limit: Threshold for contrast limiting (default: 4.0)
                       Higher values = more contrast enhancement
                       Recommended range: 2.0 - 10.0
            tile_size: Size of grid tiles for local histogram equalization (default: 8)
                      Smaller values = more local enhancement
                      Recommended range: 4 - 16
        
        Returns:
            Enhanced BGR image as numpy array
        """
        # Convert to LAB color space for better perceptual enhancement
        lab = cv2.cvtColor(self.original, cv2.COLOR_BGR2LAB)
        
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        
        # Apply CLAHE to L channel (luminance)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def bilateral_filter_enhancement(self, d: int = 9, sigma_color: float = 75, 
                                     sigma_space: float = 75, strength: float = 2.0) -> np.ndarray:
        """
        Enhance image using bilateral filter for edge-preserving detail enhancement.
        
        The bilateral filter smooths flat regions while preserving edges. By subtracting
        the filtered image from the original and adding it back with amplification,
        we can enhance texture details while maintaining edge sharpness.
        
        Args:
            d: Diameter of each pixel neighborhood (default: 9)
               Larger values = more smoothing
               Recommended range: 5 - 15
            sigma_color: Filter sigma in the color space (default: 75)
                        Larger values = more colors mixed together
            sigma_space: Filter sigma in the coordinate space (default: 75)
                        Larger values = farther pixels influence each other
            strength: Enhancement strength multiplier (default: 2.0)
                     Higher values = stronger detail enhancement
                     Recommended range: 1.0 - 5.0
        
        Returns:
            Enhanced BGR image as numpy array
        """
        # Apply bilateral filter to smooth the image while preserving edges
        smoothed = cv2.bilateralFilter(self.original, d, sigma_color, sigma_space)
        
        # Calculate detail layer (difference between original and smoothed)
        detail = self.original.astype(np.float64) - smoothed.astype(np.float64)
        
        # Enhance by adding amplified detail back to original
        enhanced = self.original.astype(np.float64) + strength * detail
        
        # Clip values and convert back to uint8
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return enhanced
    
    def apply_all_methods(self, laplacian_strength: float = 2.0,
                         clahe_clip_limit: float = 4.0, clahe_tile_size: int = 8,
                         bilateral_d: int = 9, bilateral_sigma_color: float = 75,
                         bilateral_sigma_space: float = 75, bilateral_strength: float = 2.0) -> dict:
        """
        Apply all enhancement methods and return results.
        
        Args:
            laplacian_strength: Strength for Laplacian enhancement
            clahe_clip_limit: Clip limit for CLAHE
            clahe_tile_size: Tile size for CLAHE
            bilateral_d: Diameter for bilateral filter
            bilateral_sigma_color: Color sigma for bilateral filter
            bilateral_sigma_space: Space sigma for bilateral filter
            bilateral_strength: Strength for bilateral enhancement
        
        Returns:
            Dictionary containing enhanced images for each method
        """
        return {
            'original': self.original,
            'laplacian': self.laplacian_enhancement(strength=laplacian_strength),
            'clahe': self.clahe_enhancement(clip_limit=clahe_clip_limit, tile_size=clahe_tile_size),
            'bilateral': self.bilateral_filter_enhancement(
                d=bilateral_d, sigma_color=bilateral_sigma_color,
                sigma_space=bilateral_sigma_space, strength=bilateral_strength
            )
        }


def enhance_image_for_grid_detection(image_input, method: str = 'all',
                                     **kwargs) -> Tuple[np.ndarray, str]:
    """
    Convenience function to enhance an image for grid pattern detection.
    
    Args:
        image_input: File path or numpy array (BGR image)
        method: Enhancement method to use
               'laplacian' - Laplacian edge enhancement
               'clahe' - CLAHE contrast enhancement
               'bilateral' - Bilateral filter detail enhancement
               'all' - Returns best result from all methods (default)
        **kwargs: Additional parameters for the enhancement method
    
    Returns:
        Tuple of (enhanced_image, method_name)
    """
    enhancer = TextureEnhancer(image_input)
    
    if method == 'laplacian':
        strength = kwargs.get('strength', 2.0)
        return enhancer.laplacian_enhancement(strength=strength), 'laplacian'
    
    elif method == 'clahe':
        clip_limit = kwargs.get('clip_limit', 4.0)
        tile_size = kwargs.get('tile_size', 8)
        return enhancer.clahe_enhancement(clip_limit=clip_limit, tile_size=tile_size), 'clahe'
    
    elif method == 'bilateral':
        d = kwargs.get('d', 9)
        sigma_color = kwargs.get('sigma_color', 75)
        sigma_space = kwargs.get('sigma_space', 75)
        strength = kwargs.get('strength', 2.0)
        return enhancer.bilateral_filter_enhancement(
            d=d, sigma_color=sigma_color, sigma_space=sigma_space, strength=strength
        ), 'bilateral'
    
    elif method == 'all':
        # Return all enhanced versions for comprehensive analysis
        results = enhancer.apply_all_methods(**kwargs)
        return results, 'all'
    
    else:
        raise ValueError(f"Unknown enhancement method: {method}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python texture_enhancement.py <image_path> [output_prefix]")
        print("\nThis script applies three texture enhancement methods:")
        print("  - Laplacian: Edge and detail enhancement")
        print("  - CLAHE: Contrast Limited Adaptive Histogram Equalization")
        print("  - Bilateral: Edge-preserving detail enhancement")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else 'enhanced'
    
    print(f"Processing: {image_path}")
    print("-" * 50)
    
    try:
        enhancer = TextureEnhancer(image_path)
        results = enhancer.apply_all_methods()
        
        for method_name, enhanced_img in results.items():
            if method_name != 'original':
                output_path = f"{output_prefix}_{method_name}.jpg"
                cv2.imwrite(output_path, enhanced_img)
                print(f"✓ Saved: {output_path}")
        
        print("-" * 50)
        print("Enhancement complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
